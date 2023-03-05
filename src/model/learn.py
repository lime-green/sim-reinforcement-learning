import numpy as np
from gym.wrappers import FlattenObservation
from stable_baselines3.common.evaluation import evaluate_policy

from model.ppo import MaskablePPO
from sim_gym.environment import WoWSimsEnv


def policy_callback(l, g):
    if l["info"]["is_success"]:
         print(l["info"]["dps"])


def learn():
    reward_type = "final_dps"
    # reward_type = "delta_dps"
    env = WoWSimsEnv(reward_type=reward_type, mask_invalid_actions=True)
    env = FlattenObservation(env)

    with env:
        # model = MaskedDQN(MaskedPolicy, env, verbose=1)
        model = MaskablePPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=500000, progress_bar=True)
        print(evaluate_policy(model, model.get_env(), n_eval_episodes=10, deterministic=True, callback=policy_callback))


if __name__ == "__main__":
    learn()
