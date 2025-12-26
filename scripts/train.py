from typing import Callable
import random
import time
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, asdict
from yappo.ppo import ProximalPolicyOptimizer
from yappo.rollout import collect_rollout


@dataclass
class Params:
    gym_id = "HumanoidStandup-v5"
    num_envs = 16
    seed = 1
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lamb: float = 0.95
    epsilon: float = 0.2
    entropy_coef: float = 0.0
    critic_coef: float = 0.25
    max_grad_norm: float = 0.5
    minibatch_size: int = 512
    num_epochs: int = 10
    total_timesteps: int = 10000000
    steps_per_rollout: int = 2048


def make_env(gym_id: str, idx: int, run_name: str) -> Callable:
    def thunk() -> gym.Env:
        env = gym.make(gym_id, render_mode="rgb_array")

        assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

        env = gym.wrappers.RecordEpisodeStatistics(env)
        if idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


if __name__ == "__main__":
    params = Params()
    global_step = 0

    run_name = f"runs/{params.gym_id}_{int(time.time())}"
    writer = SummaryWriter(run_name)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in asdict(params).items()])),
    )

    envs = gym.vector.SyncVectorEnv([make_env(params.gym_id, i, run_name) for i in range(8)])
    ppo = ProximalPolicyOptimizer(envs, **asdict(params))

    envs.reset(seed=params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = True

    for _ in range(ppo.num_updates):

        def log_episode_stats(timestep: int, returns: float, length: int, global_step: int = global_step) -> None:
            print(f"Timestep: {global_step + timestep} return: {returns}")
            writer.add_scalar("charts/episodic_return", returns, global_step + timestep)
            writer.add_scalar("charts/episodic_length", length, global_step + timestep)

        rollout = collect_rollout(envs, ppo.actor, params.steps_per_rollout, log_episode_stats)

        global_step += params.steps_per_rollout

        def log_training_stats(
            learning_rate: float,
            critic_loss: float,
            policy_loss: float,
            entropy_loss: float,
            global_step: int = global_step,
        ) -> None:
            writer.add_scalar("charts/learning_rate", learning_rate, global_step)
            writer.add_scalar("losses/value_loss", critic_loss, global_step)
            writer.add_scalar("losses/policy_loss", policy_loss, global_step)
            writer.add_scalar("losses/entropy_loss", entropy_loss, global_step)

        ppo.train(rollout, log_training_stats)

    envs.close()
    writer.close()
