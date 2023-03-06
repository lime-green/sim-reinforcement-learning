from typing import Union, Dict, Optional, Tuple

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.utils import is_vectorized_observation
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.dqn.policies import QNetwork


"""
This is a modified version of the DQN model from stable-baselines3.
It allows for masking out actions that are not possible.
Most methods are copied from the original implementation.
"""


class MaskedNetwork(QNetwork):
    def __init__(self, **kwargs):
        self.action_masks = kwargs.pop("action_masks")
        super().__init__(**kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values: th.Tensor = self(observation)
        mask_tensor = th.from_numpy(self.action_masks())

        # Mask out impossible actions
        q_values = q_values.masked_fill(mask_tensor == False, -np.inf)

        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action


class MaskedDQN(DQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, policy_kwargs={"action_masks": self.action_masks})

    def action_masks(self):
        if isinstance(self.env, VecEnv):
            return np.stack(self.env.env_method("action_masks"))
        else:
            return getattr(self.env, "action_masks")()

    def sample_possible_actions(self):
        if isinstance(self.env, VecEnv):
            return np.stack(self.env.env_method("sample_possible_actions"))
        else:
            return getattr(self.env, "sample_possible_actions")()

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if not deterministic and np.random.rand() < self.exploration_rate:
            action = self.sample_possible_actions()
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

    def _sample_action(
        self,
        learning_starts: int,
        action_noise = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            unscaled_action = self.sample_possible_actions()
        else:
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action


class MaskedPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        self.action_masks = kwargs.pop("action_masks")
        super().__init__(*args, **kwargs)

    def make_q_net(self) -> MaskedNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskedNetwork(action_masks=self.action_masks, **net_args).to(self.device)
