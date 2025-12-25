import argparse
import os
import random
import time
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from yappo.ppo import ProximalPolicyOptimizer, PPOParams
from yappo.rollout import collect_rollout


def make_env(gym_id: str, idx: int, run_name: str):
    def thunk():
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--gym-id", type=str, default="BipedalWalker-v3", help="the id of the gym environment")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=2000000, help="total timesteps of the experiments")
    parser.add_argument("--num-envs", type=int, default=8, help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048, help="the number of steps to run per rollout")
    args = parser.parse_args()

    global_step = 0
    run_name = f"runs/{os.path.basename(__file__).rstrip(".py")}_{int(time.time())}"
    writer = SummaryWriter(run_name)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id, i, run_name) for i in range(args.num_envs)])
    ppo = ProximalPolicyOptimizer(envs, PPOParams())

    num_updates = args.total_timesteps // (args.num_envs * args.num_steps)  # todo: this

    for update in range(num_updates):

        def log_episode_stats(timestep: int, returns: float, length: int):
            print(f"timestep: {global_step + timestep} return: {returns}")
            writer.add_scalar("charts/episodic_return", returns, global_step + timestep)
            writer.add_scalar("charts/episodic_length", length, global_step + timestep)

        rollout = collect_rollout(envs, ppo.actor, args.num_steps, log_episode_stats)

        global_step += args.num_steps

        def log_training_stats(learning_rate: float, critic_loss: float, policy_loss: float, entropy_loss: float):
            writer.add_scalar("charts/learning_rate", learning_rate, global_step)
            writer.add_scalar("losses/value_loss", critic_loss, global_step)
            writer.add_scalar("losses/policy_loss", policy_loss, global_step)
            writer.add_scalar("losses/entropy_loss", entropy_loss, global_step)

        ppo.train(rollout, log_training_stats)

    envs.close()
    writer.close()
