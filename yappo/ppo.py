import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import mse_loss
from typing import Callable, Optional
from yappo.rollout import Rollout
from yappo.network import Actor, Critic


@dataclass
class PPOParams:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lamb: float = 0.95
    epsilon: float = 0.2
    entropy_coef: float = 0.0
    critic_coef: float = 0.25
    max_grad_norm: float = 0.5
    minibatch_size: int = 512
    num_epochs: int = 10
    num_updates: int = 122


class PPODataset(Dataset):
    def __init__(self, states, actions, log_probs, advantages, returns):
        self.states = states[:-1].reshape(-1, states.shape[2])
        self.actions = actions.reshape(-1, actions.shape[2])
        self.log_probs = log_probs.reshape(-1)
        self.advantages = advantages[:-1].reshape(-1)
        self.returns = returns[:-1].reshape(-1)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.log_probs[idx], self.advantages[idx], self.returns[idx]


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + 1e-8)


class ProximalPolicyOptimizer:
    def __init__(self, envs: gym.vector.SyncVectorEnv, params: PPOParams):
        self.params = params
        self.actor = Actor(envs)
        self.critic = Critic(envs)
        self.trainable_params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(self.trainable_params, lr=self.params.learning_rate, eps=1e-5)

        def lr_schedule(step: int) -> float:
            return max(0.0, 1 - step / params.num_updates)

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_schedule)

    def _compute_gae(self, values: torch.Tensor, dones: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        advantages = torch.zeros_like(values)

        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t + 1]
            delta = rewards[t] + self.params.gamma * values[t + 1] * nonterminal - values[t]
            advantages[t] = delta + self.params.gamma * self.params.lamb * nonterminal * advantages[t + 1]

        return advantages

    def train(self, rollout: Rollout, train_cb: Optional[Callable]) -> None:
        with torch.no_grad():
            values = self.critic.get_value(rollout.states)

        advantages = self._compute_gae(values, rollout.dones, rollout.rewards)
        returns = advantages + values

        dataset = PPODataset(rollout.states, rollout.actions, rollout.log_probs, advantages, returns)

        for _ in range(self.params.num_epochs):
            for states, actions, log_probs, advantages, returns in DataLoader(
                dataset, self.params.minibatch_size, shuffle=True
            ):
                action_dist = self.actor.get_action_distribution(states)

                ratio = torch.exp(action_dist.log_prob(actions).sum(1) - log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.params.epsilon, 1 + self.params.epsilon)

                advantages = normalize(advantages)

                policy_loss = -torch.min(advantages * ratio, advantages * clipped_ratio).mean()
                entropy_loss = -action_dist.entropy().sum(1).mean()
                critic_loss = mse_loss(self.critic.get_value(states), returns)

                loss = policy_loss + self.params.entropy_coef * entropy_loss + critic_loss * self.params.critic_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.trainable_params, self.params.max_grad_norm)
                self.optimizer.step()

        self.scheduler.step()

        if train_cb is not None:
            train_cb(
                self.scheduler.get_last_lr()[0],
                critic_loss.item(),
                policy_loss.item(),
                entropy_loss.item(),
            )
