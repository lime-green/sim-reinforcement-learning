import os

from gym.wrappers import FlattenObservation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from model.dqn import MaskedDQN, MaskedPolicy
from sim_gym.environment import WoWSimsEnv


def policy_callback(l, g):
    if l["info"]["is_success"]:
        print(l["info"]["dps"], l["info"]["steps"])


def create_env(**kwargs):
    env = WoWSimsEnv(**kwargs)
    env = FlattenObservation(env)
    return env


def create_multi_env(num_envs, env_kwargs):
    return make_vec_env(
        create_env,
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs,
    )


def create_single_env(env_kwargs):
    return create_env(**env_kwargs)


def learn():
    env_kwargs = dict(sim_duration=180, reward_type="delta_dps", mask_invalid_actions=True)
    env = create_multi_env(4, env_kwargs)
    # env = create_single_env(env_kwargs)
    model = MaskedDQN(MaskedPolicy, env, verbose=1)
    # model = MaskablePPO("MlpPolicy", env, verbose=1)

    model_load_path = f"./models/{model.__class__.__name__}"
    if os.path.exists(f"{model_load_path}.zip"):
        print("Loading existing model...")
        model.load(model_load_path, env=env)
        print("Done loading model")

    model.learn(total_timesteps=10, progress_bar=True)
    print(evaluate_policy(model, model.get_env(), n_eval_episodes=1, deterministic=True, callback=policy_callback))
    print(f"Saving model to {model_load_path}.zip...")
    model.save(model_load_path, exclude=["policy_kwargs"])
    print("Done saving model")


if __name__ == "__main__":
    learn()
