import math
import os

from gym.wrappers import FlattenObservation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from model.dqn import MaskedDQN, MaskedPolicy
from environment.environment import WoWSimsEnv


def policy_callback(locals, globals_):
    if locals["info"]["is_success"]:
        print("DPS", locals["info"]["dps"], "Steps", locals["info"]["steps"])


def create_env(**kwargs):
    env = WoWSimsEnv(**kwargs)
    env = FlattenObservation(env)
    return env


def create_multi_env(num_envs, env_kwargs):
    return make_vec_env(
        create_env,
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method="fork"),
        env_kwargs=env_kwargs,
    )


def create_single_env(env_kwargs):
    return create_env(**env_kwargs)


def learn():
    best = 0
    j = 0

    episode_duration_seconds = 60
    simulation_step_duration_msec = 50
    steps_per_episode = math.ceil(
        (episode_duration_seconds * 1000) / simulation_step_duration_msec
    )
    episodes_per_training_epoch = 500
    verbose = bool(int(os.environ.get("VERBOSE", 0)))

    for i in range(60):
        env_kwargs = dict(
            sim_duration_seconds=episode_duration_seconds,
            sim_step_duration_msec=simulation_step_duration_msec,
            reward_type="delta_damage",
            verbose=verbose,
        )
        env = create_multi_env(16, env_kwargs)
        # env = create_single_env(env_kwargs)
        model = MaskedDQN(MaskedPolicy, env, verbose=verbose)

        model_load_path = f"./models/{j}-{model.__class__.__name__}"
        if os.path.exists(f"{model_load_path}.zip"):
            print("Loading existing model...")
            model.load(model_load_path, env=env)
            print("Done loading model")

        model.learn(
            total_timesteps=(steps_per_episode * episodes_per_training_epoch),
            progress_bar=True,
        )
        current = evaluate_policy(
            model,
            model.env,
            n_eval_episodes=120,
            deterministic=True,
            callback=policy_callback,
            render=False,
        )
        if current[1] > best:
            best = current[1]
            j = i
            model_load_path = f"./models/{j}-{model.__class__.__name__}"
            print(f"Saving model to {model_load_path}.zip...")
            model.save(model_load_path, exclude=["policy_kwargs"])
            print("Done saving model")

    if j > 0:
        print("---- Done! Best model was number ", j, " ------")
        model_load_path = f"./models/{j}-{model.__class__.__name__}"
        model.load(model_load_path, env=env)
        model_load_path = f"./models/best-{model.__class__.__name__}"
        model.save(model_load_path, exclude=["policy_kwargs"])


if __name__ == "__main__":
    learn()
