from typing import Optional, Tuple

import numpy as np
import sb3_contrib
from sb3_contrib.common.maskable.utils import get_action_masks


class MaskablePPO(sb3_contrib.MaskablePPO):
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if action_masks is None:
            action_masks = get_action_masks(self.env)
        return self.policy.predict(
            observation, state, episode_start, deterministic, action_masks=action_masks
        )
