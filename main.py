import gymnasium
import numpy as np
import torch.nn as nn
import torch
from typing import List
from torch.optim import Adam
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

        return self._critic(torch.Tensor(states))


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
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    log_probs: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)


def collect_rollout(env: gymnasium.Env, actor: Actor) -> Rollout:
    terminal, truncated = False, False
    state, _ = env.reset()
    rollout = Rollout(states=[state])

    while not (terminal or truncated):
        with torch.inference_mode():
            action_dist = actor.get_action_distribution(rollout.states[-1])
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            action = action.numpy().squeeze()

            state, reward, truncated, terminal, _ = env.step(action)

            rollout.states.append(state)
            rollout.actions.append(action)
            rollout.rewards.append(reward)
            rollout.log_probs.append(log_prob)

    return rollout


def compute_gae(
    rollout: Rollout, critic: Critic, lamb: float, gamma: float
) -> torch.Tensor:
    with torch.inference_mode():
        values = critic.get_value(np.stack(rollout.states))
        td_error = (
            torch.Tensor(rollout.rewards)[:, None] + gamma * values[1:] - values[:-1]
        )

        advantage = torch.zeros_like(values)

        for t in reversed(range(len(td_error))):
            advantage[t] = td_error[t] + lamb * gamma * advantage[t + 1]

        return advantage[:-1]


def compute_returns(rollout: Rollout, gamma: float) -> torch.Tensor:
    with torch.inference_mode():
        returns = torch.zeros(len(rollout.rewards) + 1)

        for t in reversed(range(len(rollout.rewards))):
            returns[t] = rollout.rewards[t] + gamma * returns[t + 1]

        return returns[:-1]


class RolloutDataLoader(torch.utils.data.DataLoader):
    def __init__(self, rollouts: List[Rollout]):
        pass


def main() -> None:
    EPSILON = 0.1
    LAMBDA = 0.95
    GAMMA = 0.99
    EPOCHS = 10
    K_critic = 0.1
    K_entropy = 0.2

    env = gymnasium.make("BipedalWalker-v3", render_mode="rgb_array")

    actor = Actor(env)
    critic = Critic(env)

    optimizer = Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)

    rollout = collect_rollout(env, actor, critic)
    advantage = compute_gae(rollout, critic, LAMBDA, GAMMA)
    returns = compute_returns(rollout, GAMMA)

    for _ in range(EPOCHS):

        action_dist = actor.get_action_distribution(rollout.states)
        values = critic.get_value(rollout.states)

        ratio = torch.exp(action_dist.log_prob(rollout.actions) - rollout.log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON)

        loss_clip = -torch.min(ratio * advantage, clipped_ratio * advantage)
        loss_entropy = -action_dist.entropy().mean()
        loss_critic = mse_loss(values, returns)

        loss = loss_clip + K_entropy * loss_entropy + K_critic * loss_critic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
