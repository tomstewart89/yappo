import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import mse_loss
from typing import Callable, Optional
from yappo.rollout import Rollout
from yappo.network import Actor, Critic


class ProximalPolicyOptimizerDataset(Dataset):
    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> None:
        self.states = states[:-1].reshape(-1, states.shape[2])
        self.actions = actions.reshape(-1, actions.shape[2])
        self.log_probs = log_probs.reshape(-1)
        self.advantages = advantages[:-1].reshape(-1)
        self.returns = returns[:-1].reshape(-1)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> tuple:
        return (
            self.states[idx],
            self.actions[idx],
            self.log_probs[idx],
            self.advantages[idx],
            self.returns[idx],
        )


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + 1e-8)


class ProximalPolicyOptimizer:
    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        learning_rate: float,
        gamma: float,
        lamb: float,
        epsilon: float,
        entropy_coef: float,
        critic_coef: float,
        max_grad_norm: float,
        minibatch_size: int,
        num_epochs: int,
        total_timesteps: int,
        steps_per_rollout: int,
    ) -> None:
        self.gamma = gamma
        self.lamb = lamb
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.max_grad_norm = max_grad_norm
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        self.num_updates = total_timesteps // (envs.num_envs * steps_per_rollout)

        self.actor = Actor(envs)
        self.critic = Critic(envs)
        self.params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(self.params, lr=learning_rate, eps=1e-5)

        def lr_schedule(step: int) -> float:
            return max(0.0, 1 - step / self.num_updates)

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_schedule)

    def compute_gae(self, values: torch.Tensor, dones: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        advantages = torch.zeros_like(values)

        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t + 1]
            delta = rewards[t] + self.gamma * values[t + 1] * nonterminal - values[t]
            advantages[t] = delta + self.gamma * self.lamb * nonterminal * advantages[t + 1]

        return advantages

    def train(self, rollout: Rollout, train_cb: Optional[Callable]) -> None:
        with torch.no_grad():
            values = self.critic.get_value(rollout.states)

        advantages = self.compute_gae(values, rollout.dones, rollout.rewards)
        returns = advantages + values

        dataset = ProximalPolicyOptimizerDataset(
            rollout.states, rollout.actions, rollout.log_probs, advantages, returns
        )

        for _ in range(self.num_epochs):
            for states, actions, log_probs, advantages, returns in DataLoader(
                dataset, self.minibatch_size, shuffle=True
            ):
                action_dist = self.actor.get_action_distribution(states)

                ratio = torch.exp(action_dist.log_prob(actions).sum(1) - log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

                advantages = normalize(advantages)

                policy_loss = -torch.min(advantages * ratio, advantages * clipped_ratio).mean()
                entropy_loss = -action_dist.entropy().sum(1).mean()
                critic_loss = mse_loss(self.critic.get_value(states), returns)
                loss = policy_loss + self.entropy_coef * entropy_loss + critic_loss * self.critic_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
                self.optimizer.step()

        self.scheduler.step()

        if train_cb is not None:
            train_cb(
                self.scheduler.get_last_lr()[0],
                critic_loss.item(),
                policy_loss.item(),
                entropy_loss.item(),
            )
