from typing import Callable
import argparse
import random
import time
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from yappo.ppo import ProximalPolicyOptimizer
from yappo.rollout import collect_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gym-id", type=str, default="BipedalWalker-v3")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamb", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--critic-coef", type=float, default=0.25)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--total-timesteps", type=int, default=2000000)
    parser.add_argument("--steps-per-rollout", type=int, default=2048)
    return parser.parse_args()


def make_env(gym_id: str, idx: int, run_name: str) -> Callable:
    def thunk() -> gym.Env:
        env = gym.make(gym_id, render_mode="rgb_array")

        assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

        env = gym.wrappers.RecordEpisodeStatistics(env)
        if idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


if __name__ == "__main__":
    args = parse_args()
    global_step = 0

    run_name = f"runs/{args.gym_id}_{int(time.time())}"
    writer = SummaryWriter(run_name)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id, i, run_name) for i in range(8)])
    ppo = ProximalPolicyOptimizer(envs, **vars(args))

    envs.reset(seed=args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    for _ in range(ppo.num_updates):

        def log_episode_stats(timestep: int, returns: float, length: int, global_step: int = global_step) -> None:
            print(f"Timestep: {global_step + timestep} return: {returns}")
            writer.add_scalar("charts/episodic_return", returns, global_step + timestep)
            writer.add_scalar("charts/episodic_length", length, global_step + timestep)

        rollout = collect_rollout(envs, ppo.actor, args.steps_per_rollout, log_episode_stats)

        global_step += len(rollout)

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
