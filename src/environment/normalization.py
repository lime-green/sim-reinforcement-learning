import json

import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class ObservationNormalizer:
    def __init__(self, exclude_keys=None):
        self._value_cache = {}
        self._normalization_config = {}
        self._exclude_keys = set(exclude_keys or [])

    def __call__(self, observations):
        if not self._normalization_config:
            self._record(observations)
            return observations

        normalized = self._normalize(observations)
        return normalized

    def build_normalization_config(self, scaling_type="minmax"):
        for key, values in self._value_cache.items():
            if isinstance(values[0], list):
                self._normalization_config[key] = {
                    "min": [np.min(v) for v in values],
                    "max": [np.max(v) for v in values],
                    "mean": [np.mean(v) for v in values],
                    "std": [np.std(v) for v in values],
                    "scaling_type": scaling_type,
                    "is_list": True,
                }
            else:
                self._normalization_config[key] = {
                    "min": np.min(values),
                    "max": np.max(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "scaling_type": scaling_type,
                    "is_list": False,
                }
        return self._normalization_config

    def load_normalization_config(self, path):
        with open(path, "r") as f:
            self._normalization_config = json.load(f)

    def save_normalization_config(self, path):
        with open(path, "w") as f:
            json.dump(self._normalization_config, f, indent=4, cls=NpEncoder)

    @property
    def normalization_config(self):
        return self._normalization_config

    def _record(self, observations):
        for key, value in observations.items():
            if key not in self._value_cache:
                self._value_cache[key] = []

            if isinstance(value, list):
                for i, v in enumerate(value):
                    if len(self._value_cache[key]) <= i:
                        self._value_cache[key].append([])
                    self._value_cache[key][i].append(v)
            else:
                self._value_cache[key].append(value)

    def _minmax_scale(self, value, config, i=None):
        min, max = config["min"], config["max"]
        if i is not None:
            min, max = config["min"][i], config["max"][i]

        if max - min == 0:
            return 0
        return (value - min) / (max - min)

    def _standard_scale(self, value, config, i=None):
        mean, std = config["mean"], config["std"]
        if i is not None:
            mean, std = config["mean"][i], config["std"][i]

        if std == 0:
            return 0
        return (value - mean) / std

    def _normalize_observation(self, value, config):
        scale = self._minmax_scale
        if config["scaling_type"] == "standard":
            scale = self._standard_scale

        if config["is_list"]:
            for i, v in enumerate(value):
                value[i] = scale(v, config, i)
        else:
            value = scale(value, config)
        return value

    def _normalize(self, observations):
        for key, value in observations.items():
            if key not in self._normalization_config:
                continue
            if key in self._exclude_keys:
                continue

            config = self._normalization_config[key]
            observations[key] = self._normalize_observation(value, config)
        return observations
