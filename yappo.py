import argparse
import os
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

global_step = 0
run_name = f"runs/{os.path.basename(__file__).rstrip(".py")}_{int(time.time())}"
writer = SummaryWriter(run_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gym-id", type=str, default="BipedalWalker-v3", help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=2000000, help="total timesteps of the experiments")
    parser.add_argument("--num-envs", type=int, default=8, help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048, help="the number of steps to run per rollout")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="lambda for general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10, help="the K epochs to update the policy")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def make_env(gym_id, seed, idx, run_name):
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
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        # env.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Critic(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()
        self._critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, states: torch.Tensor) -> torch.Tensor:
        return self._critic(states)


class Actor(nn.Module):
    def __init__(self, envs: gym.vector.SyncVectorEnv) -> None:
        super().__init__()
        self._mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self._logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_action_distribution(self, states: torch.Tensor | np.ndarray) -> Normal:
        mean = self._mean(states)
        logstd = self._logstd.expand_as(mean)
        std = torch.exp(logstd)

        return Normal(mean, std)


class RolloutDataset(Dataset):
    def __init__(self, states, actions, log_probs, advantages, returns):
        self.states = states[:-1].reshape(-1, states.shape[2])
        self.actions = actions.reshape(-1, actions.shape[2])
        self.log_probs = log_probs.reshape(-1)
        self.advantages = advantages[:-1].reshape(-1)
        self.returns = returns[:-1].reshape(-1)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (self.states[idx], self.actions[idx], self.log_probs[idx], self.advantages[idx], self.returns[idx])


def collect_rollout(envs: gym.vector.SyncVectorEnv, actor: Actor, num_steps: int):
    states = torch.zeros((num_steps + 1, envs.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, envs.num_envs) + envs.single_action_space.shape).to(device)
    log_probs = torch.zeros((num_steps, envs.num_envs)).to(device)
    rewards = torch.zeros((num_steps, envs.num_envs)).to(device)
    dones = torch.zeros((num_steps + 1, envs.num_envs)).to(device)

    states[0] = torch.Tensor(envs._observations).to(device)

    for t in range(num_steps):
        global global_step
        global_step += envs.num_envs

        with torch.no_grad():
            dist = actor.get_action_distribution(states[t])
            actions[t] = dist.sample()
            log_probs[t] = dist.log_prob(actions[t]).sum(1)

        state, reward, done, _, info = envs.step(actions[t].cpu().numpy())

        rewards[t] = torch.tensor(reward).to(device).view(-1)
        states[t + 1] = torch.Tensor(state).to(device)
        dones[t + 1] = torch.Tensor(done).to(device)

        if "episode" in info:
            print(f"global_step={global_step}, episodic_return={info['episode']['r'].mean()}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"].mean(), global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"].mean(), global_step)

    return states, actions, log_probs, rewards, dones


def compute_gae(values, dones, rewards, gamma, gae_lambda):
    advantages = torch.zeros_like(values).to(device)

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t + 1]) - values[t]
        advantages[t] = delta + gamma * gae_lambda * (1.0 - dones[t + 1]) * advantages[t + 1]

    return advantages


if __name__ == "__main__":
    args = parse_args()
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id, args.seed + i, i, run_name) for i in range(args.num_envs)])

    actor = Actor(envs).to(device)
    critic = Critic(envs).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.learning_rate, eps=1e-5)

    start_time = time.time()
    envs.reset()

    num_updates = args.total_timesteps // args.batch_size

    for update in range(num_updates):
        optimizer.param_groups[0]["lr"] = (1.0 - update / num_updates) * args.learning_rate

        states, actions, log_probs, rewards, dones = collect_rollout(envs, actor, args.num_steps)

        with torch.no_grad():
            values = critic.get_value(states).squeeze()

        advantages = compute_gae(values, dones, rewards, args.gamma, args.gae_lambda)

        returns = advantages + values

        ds = RolloutDataset(states, actions, log_probs, advantages, returns)
        loader = DataLoader(ds, args.minibatch_size, shuffle=True)

        # Optimizing the policy and value network
        clipfracs = []

        for epoch in range(args.update_epochs):
            for b_states, b_actions, b_log_probs, b_advantages, b_returns in loader:

                action_dist = actor.get_action_distribution(b_states)
                newlogprob = action_dist.log_prob(b_actions).sum(1)
                entropy = action_dist.entropy().sum(1)
                newvalue = critic.get_value(b_states)

                logratio = newlogprob - b_log_probs
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred = values.reshape(-1).cpu().numpy()
        y_true = returns.reshape(-1).cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
