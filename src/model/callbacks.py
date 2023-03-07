from stable_baselines3.common import callbacks


class CheckpointCallback(callbacks.CheckpointCallback):
    def __init__(self, exclude, **kwargs):
        self._exclude = exclude
        super().__init__(**kwargs)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path, exclude=self._exclude)
        return True
