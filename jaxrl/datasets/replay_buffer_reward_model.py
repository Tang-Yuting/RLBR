from typing import Optional, Union

import gym
import numpy as np

from jaxrl.datasets.dataset_reward_model import Dataset_Reward_Model


class ReplayBuffer_Reward_Model(Dataset_Reward_Model):

    def __init__(self, observation_space: gym.spaces.Box,
                 action_space: Union[gym.spaces.Discrete,
                                     gym.spaces.Box], capacity: int):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, *action_space.shape),
                           dtype=action_space.dtype)
        real_rewards = np.empty((capacity,), dtype=np.float32)
        rewards = np.empty((capacity, ), dtype=np.float32)
        rewards_pred = np.empty((capacity,), dtype=np.float32)
        masks = np.empty((capacity, ), dtype=np.float32)
        dones_float = np.empty((capacity, ), dtype=np.float32)
        bag_end = np.empty((capacity,), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        super().__init__(observations=observations,
                         actions=actions,
                         real_rewards=real_rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         rewards=rewards,
                         rewards_pred = rewards_pred,
                         bag_end=bag_end,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity
        print("init_replay_buffer")

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray, real_reward: float,
               bag_end: float):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation
        self.real_rewards[self.insert_index] = real_reward
        self.rewards_pred[self.insert_index] = 0
        self.bag_end[self.insert_index] = bag_end

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def insert_traj(self, trajectory):
        for transition in trajectory:
            self.insert(
                observation=transition['obs'],
                action=transition['acts'],
                reward=transition['rews'],
                mask=transition['mask'],
                done_float=transition['done'],
                next_observation=transition['obs_next'],
                real_reward=transition['real_rews'],
                bag_end=transition['bag_end']
            )

    def index_batch(self, batch, indices):
        indexed = {}
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                indexed[key] = value[indices, ...]
            elif isinstance(value, list):
                indexed[key] = [value[i] for i in indices]
            else:
                raise ValueError(f"Unsupported data type for key '{key}': {type(value)}")
        return indexed
