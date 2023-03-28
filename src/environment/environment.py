import gym
import numpy as np
import math
import os
from rich import print

from agent.sim_agent import SimAgent
from agent.sim_config import create_config
from environment.actions import ACTION_SPACE, Action
from environment.state import State

NORMALIZATION_CONFIG = "normalization_config.json"


class WoWSimsEnv(gym.Env):
    def __init__(
        self,
        sim_duration_seconds,
        sim_step_duration_msec,
        reward_type: str = "final_dps",
        verbose=False,
    ):
        super(WoWSimsEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(len(ACTION_SPACE))
        self.observation_space = State.get_observation_space()

        self._normalizer = State.create_normalizer()
        try:
            self._normalizer.load_normalization_config(NORMALIZATION_CONFIG)
            print("Loaded normalization config")
            print(self._normalizer.normalization_config)
        except FileNotFoundError:
            print("No normalization config found, using unnormalized observations")

        self._verbose = verbose

        # set up reward type
        self.calculate_reward = None
        if reward_type == "delta_dps":
            self.calculate_reward = self.calculate_reward_delta_dps
        elif reward_type == "delta_damage":
            self.calculate_reward = self.calculate_reward_delta_damage
        elif reward_type == "final_dps":
            self.calculate_reward = self.calculate_reward_final_dps
        elif reward_type == "final_damage":
            self.calculate_reward = self.calculate_reward_final_damage
        elif reward_type == "abs_dps":
            self.calculate_reward = self.calculate_reward_abs_dps
        elif reward_type == "abs_damage":
            self.calculate_reward = self.calculate_reward_abs_damage
        elif reward_type == "guided":
            self.calculate_reward = self.calculate_reward_guided

        assert self.calculate_reward is not None, (
            "%s is not a valid reward type" % reward_type
        )

        self._sim_duration_seconds = sim_duration_seconds

        # initialize mutable state
        self.state = None
        self._steps = 0
        self._commands = ""
        self._last_state = None
        self._last_action = None
        self._best_damage = 0
        self._total_reward = 0
        self._sim_agent = SimAgent(
            port="/tmp/sim-agent.sock", step_duration_msec=sim_step_duration_msec
        )

    def step(self, action):
        assert self.action_space.contains(action), "%r invalid" % action

        self._steps += 1
        action: Action = ACTION_SPACE[action]
        assert action.can_do(self.state), "attempted illegal action %r" % action
        action.do(self._sim_agent, self.state)

        new_state = self._sim_agent.get_state()
        self.state = State(new_state, self._normalizer)
        reward = self.calculate_reward()
        self._total_reward += reward

        if hasattr(action, "spell"):
            self._commands += action.spell + " " + str(math.floor(reward)) + " >"

        done = self.state.is_done
        obs = self._get_obs()
        self._last_state = self.state

        if self.state.is_done:
            print(
                self.state.dps,
                self.state.ability_dps,
                self.state.melee_dps,
                self.state.disease_dps,
            )

        if self.state.is_done and self._best_damage < self.state.damage:
            self._best_damage = self.state.damage
            print("New best found:")
            print(self._commands)
            print("total = ", self._best_damage)
            print("dps = ", self.state.dps)
            print(
                "---------------------------------------------------------------------"
            )
            print(" ")
        return obs, reward, done, self.get_metadata()

    def _get_active_diseases(self, disease_state):
        diseases = disease_state.values()
        return [
            d
            for d in diseases
            if d["name"] in ("BloodPlague", "FrostFever") and d["isActive"]
        ]

    def calculate_reward_guided(self):
        if self._last_state is None:
            return 0

        if self.state.is_done:
            if self._verbose:
                print(self._commands)
            return self.state.ability_dps

        # rp
        delta_rp = self.state.runic_power - self._last_state.runic_power

        # diseases
        num_diseases_before = len(self._get_active_diseases(self._last_state.debuffs))
        num_diseases_now = len(self._get_active_diseases(self.state.debuffs))
        delta_diseases = num_diseases_now - num_diseases_before
        num_diseases = num_diseases_now

        rp_reward = max(0, delta_rp) / 10
        disease_delta_reward = delta_diseases * 10
        num_disease_reward = num_diseases * 0.5
        damage_reward = (
            self.state.ability_damage - self._last_state.ability_damage
        ) / 200

        reward = rp_reward + disease_delta_reward + num_disease_reward + damage_reward

        if self._verbose:
            if self._steps % 100 == 0:
                print(
                    "Total: ",
                    reward,
                    "RP reward: ",
                    rp_reward,
                    "dis reward: ",
                    disease_delta_reward,
                    "num dis reward: ",
                    num_disease_reward,
                    "damage reward: ",
                    damage_reward,
                )
        return reward

    def calculate_reward_delta_dps(self):
        if self._last_state is None:
            return self.state.dps
        return self.state.dps - self._last_state.dps

    def calculate_reward_delta_damage(self):
        if self._last_state is None:
            return self.state.damage
        return self.state.damage - self._last_state.damage

    def calculate_reward_final_dps(self):
        if self.state.is_done:
            return self.state.dps
        return 0

    def calculate_reward_final_damage(self):
        if self.state.is_done:
            return self.state.damage - self._last_state.damage
        return 0

    def calculate_reward_abs_dps(self):
        return self.state.dps

    def calculate_reward_abs_damage(self):
        return self.state.damage

    def get_metadata(self):
        return {
            "dps": self.state.dps,
            "is_success": self.state.is_done,
            "steps": self._steps,
        }

    def reset(self, seed=None, options=None):
        sim_config = create_config(
            random_seed=seed,
            duration=self._sim_duration_seconds,
        )
        state = self._sim_agent.reset(sim_config)
        self.state = State(state, self._normalizer)
        self._last_state = None
        self._steps = 0
        self._commands = ""
        self._total_reward = 0

        return self._get_obs()

    def render(self, mode="ascii"):
        pass

    def close(self):
        self._sim_agent.close()

        if not os.path.exists(NORMALIZATION_CONFIG):
            print("Saving normalization config")
            self._normalizer.build_normalization_config()
            self._normalizer.save_normalization_config(NORMALIZATION_CONFIG)

    def sample_possible_actions(self):
        return np.random.choice(
            [i for i, mask in enumerate(self.action_masks()) if mask]
        )

    def action_masks(self):
        return np.array([action.can_do(self.state) for action in ACTION_SPACE])

    def _get_obs(self):
        return self.state.get_observations()
