import gymnasium
import numpy as np
import torch.nn as nn
import torch
from typing import List
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.distributions.normal import Normal
from torch.distributions import Distribution
from torch.nn.functional import mse_loss
from dataclasses import dataclass, field


class Critic(nn.Module):
    def __init__(self, env: gymnasium.Env):
        super().__init__()
        self._critic = nn.Sequential(
            nn.Linear(np.prod(env.observation_space.shape), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def get_value(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 1:
            states = states[None]

        return self._critic(torch.Tensor(states))[..., 0]


class Actor(nn.Module):
    def __init__(self, env: gymnasium.Env):
        super().__init__()

        self._mean = nn.Sequential(
            nn.Linear(np.prod(env.observation_space.shape), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, np.prod(env.action_space.shape)),
        )
        self._logstd = nn.Parameter(torch.zeros(1, np.prod(env.action_space.shape)))

    def get_action_distribution(
        self, states: torch.Tensor | np.ndarray
    ) -> Distribution:
        if states.ndim == 1:
            states = states[None]

        mean = self._mean(torch.Tensor(states))
        logstd = self._logstd.expand_as(mean)
        std = torch.exp(logstd)

        return Normal(mean, std)


@dataclass
class Rollout:
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def collect_sars(env: gymnasium.Env, actor: Actor):
    terminal, truncated = False, False
    state, _ = env.reset()

    states: List[np.ndarray] = [state]
    actions: List[np.ndarray] = []
    rewards: List[float] = []

    while not (terminal or truncated):
        with torch.inference_mode():
            action_dist = actor.get_action_distribution(states[-1])
            action = action_dist.sample()
            action = action.numpy().squeeze()

            state, reward, truncated, terminal, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

    return torch.Tensor(states), torch.Tensor(actions), torch.Tensor(rewards)


def compute_advantage(
    states: torch.Tensor,
    rewards: torch.Tensor,
    critic: Critic,
    lamb: float,
    gamma: float,
) -> torch.Tensor:
    with torch.inference_mode():
        values = critic.get_value(states)
        td_error = rewards + gamma * values[1:] - values[:-1]

        advantage = torch.zeros_like(values)

        for t in reversed(range(len(td_error))):
            advantage[t] = td_error[t] + lamb * gamma * advantage[t + 1]

        return advantage[:-1]


def compute_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    with torch.inference_mode():
        returns = torch.zeros(len(rewards) + 1)

        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + gamma * returns[t + 1]

        return returns[:-1]


def collect_rollout(
    env: gymnasium.Env,
    actor: Actor,
    critic: Critic,
    LAMBDA: float = 0.95,
    GAMMA: float = 0.99,
) -> Rollout:
    states, actions, rewards = collect_sars(env, actor)
    log_probs = actor.get_action_distribution(states[:-1]).log_prob(actions)
    advantages = compute_advantage(states, rewards, critic, LAMBDA, GAMMA)
    returns = compute_returns(rewards, GAMMA)

    return Rollout(states[:-1], actions, log_probs, rewards, advantages, returns)


class RolloutDataset(Dataset):
    def __init__(self, rollouts: List[Rollout]):
        self.states = torch.vstack([r.states for r in rollouts])
        self.actions = torch.vstack([r.actions for r in rollouts])
        self.log_probs = torch.vstack([r.log_probs for r in rollouts])
        self.advantages = torch.hstack([r.advantages for r in rollouts])
        self.returns = torch.hstack([r.returns for r in rollouts])

    def __len__(self):
        return len(self.advantages)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.log_probs[idx],
            self.advantages[idx],
            self.returns[idx],
        )


def main() -> None:
    EPSILON = 0.1
    N_EPOCHS = 10
    N_ROLLOUTS = 3
    K_CRITIC = 0.1
    K_ENTROPY = 0.2

    env = gymnasium.make("BipedalWalker-v3", render_mode="rgb_array")

    actor = Actor(env)
    critic = Critic(env)
    optimizer = Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)

    dataset = RolloutDataset(
        [collect_rollout(env, actor, critic) for _ in range(N_ROLLOUTS)]
    )

    for _ in range(N_EPOCHS):
        for states, actions, log_probs, advantage, returns in DataLoader(
            dataset, batch_size=32
        ):

            action_dist = actor.get_action_distribution(states)
            values = critic.get_value(states)

            ratio = torch.exp(action_dist.log_prob(actions) - log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON)

            loss_clip = -torch.min(
                ratio * advantage[:, None], clipped_ratio * advantage[:, None]
            )
            loss_entropy = -action_dist.entropy().mean()
            loss_critic = mse_loss(values, returns)

            loss = loss_clip + K_ENTROPY * loss_entropy + K_CRITIC * loss_critic

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
