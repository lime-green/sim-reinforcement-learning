import os

from gym.wrappers import FlattenObservation
from stable_baselines3.common.evaluation import evaluate_policy

from model.dqn import MaskedDQN, MaskedPolicy
from model.ppo import MaskablePPO
from sim_gym.environment import WoWSimsEnv


def policy_callback(l, g):
    if l["info"]["is_success"]:
        print(l["info"]["dps"], l["info"]["steps"])


def learn():
    reward_type = "final_dps"
    # reward_type = "delta_dps"
    env = WoWSimsEnv(reward_type=reward_type, mask_invalid_actions=True)
    env = FlattenObservation(env)

    with env:
        model = MaskedDQN(MaskedPolicy, env, verbose=1)
        # model = MaskablePPO("MlpPolicy", env, verbose=1)

        # check if model path exists, if so, load it
        model_load_path = f"./models/{model.__class__.__name__}"
        if os.path.exists(model_load_path):
            print("Loading existing model")
            model.load(model_load_path, env=env)

        model.learn(total_timesteps=1, progress_bar=True)
        print(evaluate_policy(model, model.get_env(), n_eval_episodes=10, deterministic=True, callback=policy_callback))
        print(f"Saving model to {model_load_path}...")
        model.save(model_load_path)


if __name__ == "__main__":
    learn()
