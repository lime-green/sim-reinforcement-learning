import gym
import numpy as np

from agent.sim_agent import SimAgent
from agent.sim_config import create_config
from sim_gym.actions import ACTION_SPACE, Action
from sim_gym.state import State
from sim_gym.constants import SPELLS


class WoWSimsEnv(gym.Env):
    def __init__(self, sim_duration, reward_type: str = "final_dps", mask_invalid_actions: bool = True):
        self.action_space = gym.spaces.Discrete(len(ACTION_SPACE))
        self.observation_space = State.get_observation_space()
        self.state = None
        self._last_dps = 0
        self._sim_agent = SimAgent(port="/tmp/sim-agent.sock")
        self._rewards = []
        self._reward_type = reward_type
        self._mask_invalid_actions = mask_invalid_actions
        self._steps = 0
        self._bestDPS = 0
        self._render = False
        self._sim_duration = sim_duration

    def step(self, action):
        assert self.action_space.contains(action), "%r invalid" % action
        num = int(action)

        self._steps += 1
        action: Action = ACTION_SPACE[action]

        if self._mask_invalid_actions:
            # If we're masking invalid actions, then we should never get an invalid action
            assert action.can_do(self.state), "%r cannot be done right now" % action
        else:
            if not action.can_do(self.state):
                return self._get_obs(), -1000, False, self.get_metadata()

        action.do(self._sim_agent, self.state)
        new_state = self._sim_agent.get_state()
        self.state = State(new_state)

        #if len(self._rewards) == 0:
        #    self._rewards.append(0)
        #    self._rewards.append(0)
        #    self._rewards.append(0)

        #self._rewards[2] = self.state.damage/100
        #reward = self.calculate_reward()
        self._rewards = self.state.damage/100
        #self._rewards[0] = reward
        #self._rewards = self._rewards / np.linalg.norm(self._rewards)


        reward = self.state.damage/100
        if self._render and num<14:
            print(SPELLS[int(num)])
            print(reward)
        done = self.state.is_done
        obs = self._get_obs()
        self._last_dps = self.state.dps
        # if str(action).find("Cast"):
        #    print(action)
        return obs, reward, done, self.get_metadata()

    def calculate_reward(self):
        reward = 0

        if self._reward_type == "delta_dps":
            reward = self.state.dps - self._last_dps
            self._rewards[1] = self.state.dps/100
        elif self._reward_type == "final_dps" and self.state.is_done:
            reward = self.state.dps
        elif self._reward_type == "just_dps":
            reward = self.state.dps
        #if reward > self._bestDPS:
            #self._bestDPS = reward
            #print("Best reward: " + str(reward))
        return reward

    def get_metadata(self):
        return {"dps": self.state.dps, "is_success": self.state.is_done, "steps": self._steps}

    def reset(self, seed=None, options=None):
        sim_config = create_config(
            random_seed=seed,
            duration=self._sim_duration,
        )
        state = self._sim_agent.reset(sim_config)
        self.state = State(state)
        self._last_dps = 0
        self._rewards = []
        self._steps = 0

        return self._get_obs()

    def render(self, mode='human'):
        self._render = True
        return

    def close(self):
        self._sim_agent.close()

    def sample_possible_actions(self):
        return np.random.choice([i for i, mask in enumerate(self.action_masks()) if mask])

    def action_masks(self):
        return np.array([action.can_do(self.state) for action in ACTION_SPACE])

    def _get_obs(self):
        return self.state.get_observations()
