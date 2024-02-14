import collections
from typing import Tuple, Union

import numpy as np
from tqdm import tqdm

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)


class Dataset(object):

    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])

    # TT: ircr, rewards modify
    def ircr_rewards(self, batch: Batch, min_reward, max_reward) -> Batch:
        # print("batch.rewards", batch.rewards)
        # print("min_reward, max_reward", min_reward, max_reward)
        modified_rewards = (batch.rewards - min_reward) / (max_reward - min_reward)
        return batch._replace(rewards=modified_rewards)

    # TT: sample sequence
    def sample_sequences(self, num_query, len_query, sampled_indices):
        observation_dim = self.observations.shape[-1]
        action_dim = self.actions.shape[-1]

        total_reward_seq = np.zeros((num_query, len_query))
        total_obs_seq = np.zeros((num_query, len_query, observation_dim))
        total_next_obs_seq = np.zeros((num_query, len_query, observation_dim))
        total_act_seq = np.zeros((num_query, len_query, action_dim))
        total_done_seq = np.zeros((num_query, len_query), dtype=np.int32)
        total_timestep = np.zeros((num_query, len_query), dtype=np.int32)
        query_range = np.arange(num_query)

        total_idx = np.zeros(num_query, dtype=np.int32)
        for query_count, i in enumerate(tqdm(query_range, desc="get queries from saved indices")):
            start_idx = sampled_indices[i]
            end_idx = start_idx + len_query

            # print("start_idx", start_idx, "end_idx", end_idx)

            reward_seq = self.rewards[start_idx:end_idx]
            obs_seq = self.observations[start_idx:end_idx]
            next_obs_seq = self.next_observations[start_idx:end_idx]
            act_seq = self.actions[start_idx:end_idx]
            done_seq = self.dones_float[start_idx:end_idx]
            timestep_seq = np.arange(1, len_query + 1)

            # print("start_idx", start_idx, "end_idx", end_idx)

            total_reward_seq[query_count] = reward_seq
            total_obs_seq[query_count] = obs_seq
            total_next_obs_seq[query_count] = next_obs_seq
            total_act_seq[query_count] = act_seq
            total_done_seq[query_count] = done_seq
            total_timestep[query_count] = timestep_seq

            total_idx[query_count] = start_idx

        seg_reward = total_reward_seq.copy()
        seg_obs = total_obs_seq.copy()
        seg_next_obs = total_next_obs_seq.copy()
        seq_act = total_act_seq.copy()
        seq_done = total_done_seq.copy()
        seq_timestep = total_timestep.copy()

        # print("ave_rewards", seg_ave_reward.shape, seg_ave_reward[0])
        # print("rewards", seg_reward.shape, seg_reward[0])

        batch = {}

        batch['rewards'] = seg_reward
        batch['observations'] = seg_obs
        batch['next_observations'] = seg_next_obs
        batch['actions'] = seq_act
        batch['dones'] = seq_done
        batch['timestep'] = seq_timestep
        batch['start_indices'] = sampled_indices

        return batch


    def get_initial_states(
        self,
        and_action: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        states = []
        if and_action:
            actions = []
        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)

        def compute_returns(traj):
            episode_return = 0
            for _, _, rew, _, _, _ in traj:
                episode_return += rew

            return episode_return

        trajs.sort(key=compute_returns)

        for traj in trajs:
            states.append(traj[0][0])
            if and_action:
                actions.append(traj[0][1])

        states = np.stack(states, 0)
        if and_action:
            actions = np.stack(actions, 0)
            return states, actions
        else:
            return states

    def get_monte_carlo_returns(self, discount) -> np.ndarray:
        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)
        mc_returns = []
        for traj in trajs:
            mc_return = 0.0
            for i, (_, _, reward, _, _, _) in enumerate(traj):
                mc_return += reward * (discount**i)
            mc_returns.append(mc_return)

        return np.asarray(mc_returns)

    def take_top(self, percentile: float = 100.0):
        assert percentile > 0.0 and percentile <= 100.0

        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)

        def compute_returns(traj):
            episode_return = 0
            for _, _, rew, _, _, _ in traj:
                episode_return += rew

            return episode_return

        trajs.sort(key=compute_returns)

        N = int(len(trajs) * percentile / 100)
        N = max(1, N)

        trajs = trajs[-N:]

        (self.observations, self.actions, self.rewards, self.masks,
         self.dones_float, self.next_observations) = merge_trajectories(trajs)

        self.size = len(self.observations)

    def take_random(self, percentage: float = 100.0):
        assert percentage > 0.0 and percentage <= 100.0

        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)
        np.random.shuffle(trajs)

        N = int(len(trajs) * percentage / 100)
        N = max(1, N)

        trajs = trajs[-N:]

        (self.observations, self.actions, self.rewards, self.masks,
         self.dones_float, self.next_observations) = merge_trajectories(trajs)

        self.size = len(self.observations)

    def train_validation_split(self,
                               train_fraction: float = 0.8
                               ) -> Tuple['Dataset', 'Dataset']:
        trajs = split_into_trajectories(self.observations, self.actions,
                                        self.rewards, self.masks,
                                        self.dones_float,
                                        self.next_observations)
        train_size = int(train_fraction * len(trajs))

        np.random.shuffle(trajs)

        (train_observations, train_actions, train_rewards, train_masks,
         train_dones_float,
         train_next_observations) = merge_trajectories(trajs[:train_size])

        (valid_observations, valid_actions, valid_rewards, valid_masks,
         valid_dones_float,
         valid_next_observations) = merge_trajectories(trajs[train_size:])

        train_dataset = Dataset(train_observations,
                                train_actions,
                                train_rewards,
                                train_masks,
                                train_dones_float,
                                train_next_observations,
                                size=len(train_observations))
        valid_dataset = Dataset(valid_observations,
                                valid_actions,
                                valid_rewards,
                                valid_masks,
                                valid_dones_float,
                                valid_next_observations,
                                size=len(valid_observations))

        return train_dataset, valid_dataset
