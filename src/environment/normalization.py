import numpy as np
from gym.wrappers import normalize


class NormalizeObservation(normalize.NormalizeObservation):
    def __init__(
        self,
        env,
        epsilon=1e-8,
        scaling="minmax"
    ):
        super().__init__(env, epsilon)

        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = RunningMeanStdMinMax(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStdMinMax(shape=self.observation_space.shape)
        self.epsilon = epsilon
        self.scale_fn = {"standard": self._scale_standard, "minmax": self._scale_minmax}[scaling]

    def normalize(self, obs):
        self.obs_rms.update(obs)
        return self.scale_fn(obs)

    def _scale_standard(self, obs):
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)

    def _scale_minmax(self, obs):
        return (obs - self.obs_rms.min) / (self.obs_rms.max - self.obs_rms.min + self.epsilon)


class RunningMeanStdMinMax(normalize.RunningMeanStd):
    def __init__(self, epsilon=1e-4, shape=()):
        super().__init__(epsilon, shape)
        self.min = np.zeros(shape)
        self.max = np.zeros(shape)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = normalize.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )
        self.min = np.minimum(self.min, batch_var)
        self.max = np.maximum(self.max, batch_var)
