from typing import Callable, Optional
import gymnasium as gym
import torch
from yappo.network import Actor


class Rollout:
    def __init__(self, envs: gym.vector.SyncVectorEnv, num_steps: int) -> None:
        self.states = torch.zeros((num_steps + 1, envs.num_envs) + envs.single_observation_space.shape)
        self.actions = torch.zeros((num_steps, envs.num_envs) + envs.single_action_space.shape)
        self.log_probs = torch.zeros((num_steps, envs.num_envs))
        self.rewards = torch.zeros((num_steps, envs.num_envs))
        self.dones = torch.zeros((num_steps + 1, envs.num_envs))

    def __len__(self) -> int:
        return len(self.actions)


def collect_rollout(
    envs: gym.vector.SyncVectorEnv, actor: Actor, num_steps: int, episode_cb: Optional[Callable] = None
) -> Rollout:

    rollout = Rollout(envs, num_steps)
    rollout.states[0] = torch.Tensor(envs._observations)

    low, high = torch.Tensor(envs.action_space.low), torch.Tensor(envs.action_space.high)

    for t in range(num_steps):
        with torch.no_grad():
            dist = actor.get_action_distribution(rollout.states[t])
            rollout.actions[t] = torch.clip(dist.sample(), low, high)
            rollout.log_probs[t] = dist.log_prob(rollout.actions[t]).sum(1)

        state, reward, done, _, info = envs.step(rollout.actions[t].cpu().numpy())

        rollout.rewards[t] = torch.tensor(reward)
        rollout.states[t + 1] = torch.Tensor(state)
        rollout.dones[t + 1] = torch.Tensor(done)

        if episode_cb is not None and "episode" in info:
            for i in info["_episode"].nonzero()[0]:
                episode_cb(t, info["episode"]["r"][i], info["episode"]["l"][i])

    return rollout
