import math
import os

from gym.wrappers import FlattenObservation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from model.dqn import MaskedDQN, MaskedPolicy
from model.ppo import MaskablePPO
from environment.environment import WoWSimsEnv
from environment.normalization import NormalizeObservation


def policy_callback(locals, globals_):
    if locals["info"]["is_success"]:
        print("DPS", locals["info"]["dps"], "Steps", locals["info"]["steps"])


def create_env(**kwargs):
    env = WoWSimsEnv(**kwargs)
    env = FlattenObservation(env)
    env = NormalizeObservation(env)
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


def initialize_environment(count, env_kwargs):
    if count == 1:
        return create_single_env(env_kwargs)
    return create_multi_env(count, env_kwargs)


def initialize_model(env, verbose):
    model_type = os.environ.get("MODEL_TYPE", "PPO")
    if model_type == "PPO":
        model = MaskablePPO("MlpPolicy", env, verbose=verbose)
    elif model_type == "DQN":
        model = MaskedDQN(MaskedPolicy, env, verbose=verbose)
    else:
        assert False, "%s is not a valid model type" % model_type
    load_latest_file(model, env)
    return model


def find_highest_file_index(modelName):
    if not os.path.exists(f"./models/{modelName}/"):
        print("no path")
        return -1
    return max(
        [int(f[: f.index(".")]) for f in os.listdir(f"./models/{modelName}/")],
        default=-1,
    )


def save_file(model):
    new_file_index = find_highest_file_index(model.__class__.__name__) + 1
    model_save_path = f"./models/{model.__class__.__name__}/{new_file_index}"
    print(f"Saving model to {model_save_path}.zip...")
    model.save(model_save_path, exclude=["policy_kwargs"])
    print("Done saving model")
    return model_save_path


def load_latest_file(model, env):
    file_index = find_highest_file_index(model.__class__.__name__)
    if file_index == -1:
        print("There is no existing model to load, starting learning from scratch")
    else:
        model_load_path = f"./models/{model.__class__.__name__}/{file_index}"
        assert os.path.exists(f"{model_load_path}.zip")
        print(f"Loading existing model from {model_load_path}.zip...")
        model.load(model_load_path, env=env)
        print("Done loading model")


def learn():
    verbose = bool(int(os.environ.get("VERBOSE", 0)))
    environment_count = int(os.environ.get("ENVIRONMENT_COUNT", 16))
    episode_duration_seconds = int(os.environ.get("EPISODE_DURATION_SECONDS", 60))
    simulation_step_duration_msec = int(
        os.environ.get("SIMULATION_STEP_DURATION_MSEC", 50)
    )
    episodes_per_training_iteration = int(
        os.environ.get("EPISODES_PER_TRAINING_ITERATION", 400)
    )
    reward_type = os.environ.get("REWARD_TYPE", "delta_damage")
    steps_per_episode = math.ceil(
        (episode_duration_seconds * 1000) / simulation_step_duration_msec
    )
    env_kwargs = dict(
        sim_duration_seconds=episode_duration_seconds,
        sim_step_duration_msec=simulation_step_duration_msec,
        reward_type=reward_type,
        verbose=verbose,
    )
    env = initialize_environment(environment_count, env_kwargs)
    model = initialize_model(env, verbose)

    model.learn(
        total_timesteps=(steps_per_episode * episodes_per_training_iteration),
        progress_bar=True,
    )
    evaluate_policy(
        model,
        env,
        n_eval_episodes=20,
        deterministic=True,
        callback=policy_callback,
        render=False,
    )
    save_file(model)
    env.close()


if __name__ == "__main__":
    learn()
