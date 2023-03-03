import json

import gymnasium as gym

from agent.sim_agent import SimAgent
from agent.sim_config import create_config
from gym_env.actions import ACTION_SPACE, Action
from gym_env.state import State


class WoWSimsEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(len(ACTION_SPACE))
        self.observation_space = State.get_observation_space()
        self.state = None
        self._last_dps = 0
        self._sim_agent = SimAgent(port=1234)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        action: Action = ACTION_SPACE[action]
        state = action.do(self._sim_agent, self.state)
        self.state = State(state)

        reward = self.state.calculate_reward(self._last_dps)
        done = self.state.is_done
        obs = self._get_obs()

        self._last_dps = self.state.dps

        return obs, reward, done, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        sim_config = create_config(random_seed=seed)
        self.state = self._sim_agent.reset(sim_config)
        return self._get_obs()

    def render(self, mode='human'):
        return json.dumps(self.state)

    def close(self):
        self._sim_agent.close()

    def _get_obs(self):
        return self.state.get_observations()
