import collections
from typing import Tuple, Union
import jax
import jax.numpy as jnp

import numpy as np
from tqdm import tqdm, trange

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations, size):
    trajs = [[]]
    # count = 0
    for i in tqdm(range(size), desc="split"):
        # count = count + 1
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])
            # print("terminals", i, count)
            # count = 0
    return trajs


@jax.jit
def batch_to_jax(batch):
    return jax.tree_util.tree_map(jax.device_put, batch)

class Dataset_Reward_Model(object):

    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 real_rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 rewards: np.ndarray, rewards_pred: np.ndarray, bag_end: np.ndarray, size: int):
        self.observations = observations
        self.actions = actions
        self.real_rewards = real_rewards  # real reward
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.rewards = rewards  # bagged reward
        self.rewards_pred = rewards_pred  # reward from reward model, for training agent
        self.bag_end = bag_end
        self.size = size

    # agent
    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards_pred[indx],  # reward from reward model
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])

    # reward model
    def sample_sequences(self, num_query, len_query, sampled_indices):
        observation_dim = self.observations.shape[-1]
        action_dim = self.actions.shape[-1]

        total_real_reward_seq = np.zeros((num_query, len_query))
        total_reward_seq = np.zeros((num_query, len_query))
        total_obs_seq = np.zeros((num_query, len_query, observation_dim))
        total_next_obs_seq = np.zeros((num_query, len_query, observation_dim))
        total_act_seq = np.zeros((num_query, len_query, action_dim))
        total_done_seq = np.zeros((num_query, len_query), dtype=np.int32)
        total_bag_end_seq = np.zeros((num_query, len_query), dtype=np.int32)
        total_timestep = np.zeros((num_query, len_query), dtype=np.int32)
        query_range = np.arange(num_query)

        total_idx = np.zeros(num_query, dtype=np.int32)
        for query_count, i in enumerate(tqdm(query_range, desc="get queries from saved indices")):
            start_idx = sampled_indices[i]
            end_idx = start_idx + len_query

            # print("start_idx", start_idx, "end_idx", end_idx)

            real_reward_seq = self.real_rewards[start_idx:end_idx]
            reward_seq = self.rewards[start_idx:end_idx]
            obs_seq = self.observations[start_idx:end_idx]
            next_obs_seq = self.next_observations[start_idx:end_idx]
            act_seq = self.actions[start_idx:end_idx]
            done_seq = self.dones_float[start_idx:end_idx]
            bag_end_seq = self.bag_end[start_idx:end_idx]
            timestep_seq = np.arange(1, len_query + 1)

            # print("start_idx", start_idx, "end_idx", end_idx)

            total_real_reward_seq[query_count] = real_reward_seq
            total_reward_seq[query_count] = reward_seq
            total_obs_seq[query_count] = obs_seq
            total_next_obs_seq[query_count] = next_obs_seq
            total_act_seq[query_count] = act_seq
            total_done_seq[query_count] = done_seq
            total_bag_end_seq[query_count] = bag_end_seq
            total_timestep[query_count] = timestep_seq

            total_idx[query_count] = start_idx

        seg_real_reward = total_real_reward_seq.copy()
        seg_reward = total_reward_seq.copy()
        seg_obs = total_obs_seq.copy()
        seg_next_obs = total_next_obs_seq.copy()
        seq_act = total_act_seq.copy()
        seq_done = total_done_seq.copy()
        seq_bag_end = total_bag_end_seq.copy()
        seq_timestep = total_timestep.copy()

        # print("ave_rewards", seg_ave_reward.shape, seg_ave_reward[0])
        # print("rewards", seg_reward.shape, seg_reward[0])

        batch = {}

        batch['real_rewards'] = seg_real_reward
        batch['rewards'] = seg_reward
        batch['observations'] = seg_obs
        batch['next_observations'] = seg_next_obs
        batch['actions'] = seq_act
        batch['dones'] = seq_done
        batch['bag_end'] = seq_bag_end
        batch['timestep'] = seq_timestep
        batch['start_indices'] = sampled_indices

        return batch

    def recompute_rewards(self,
                          reward_model,
                          seq_len: int,
                          batch_size: int = 256,
                          with_attn_weights: bool = False):
        # see dataset_utils.reward_from_preference_transformer
        # print("self.size", self.size)
        trajs = split_into_trajectories(
            self.observations,
            self.actions,
            self.rewards,
            self.masks,
            self.dones_float,
            self.next_observations,
            self.size
        )
        trajectories = []
        trj_mapper = []
        observation_dim = self.observations.shape[-1]
        action_dim = self.actions.shape[-1]

        for trj_idx, traj in tqdm(enumerate(trajs), total=len(trajs), desc="chunk trajectories"):
            _obs, _act, _reward, _mask, _done, _next_obs = [], [], [], [], [], []
            for _o, _a, _r, _m, _d, _no in traj:
                _obs.append(_o)
                _act.append(_a)
                _reward.append(_r)
                _mask.append(_m)
                _done.append(_d)
                _next_obs.append(_no)

            traj_len = len(traj)
            _obs, _act = np.asarray(_obs), np.asarray(_act)
            trajectories.append((_obs, _act))

            for seg_idx in range(traj_len):
                trj_mapper.append((trj_idx, seg_idx))

        data_size = self.size
        # print("data_size", data_size, "batch_size", batch_size)
        # interval = int(data_size / batch_size) + 1
        if data_size % batch_size == 0:
            interval = data_size // batch_size
        else:
            interval = data_size // batch_size + 1
        new_r = np.zeros_like(self.rewards)
        pts = []
        attn_weights = []
        for i in trange(interval, desc="relabel reward"):
            start_pt = i * batch_size
            end_pt = min((i + 1) * batch_size, data_size)

            _input_obs, _input_act, _input_timestep, _input_attn_mask, _input_pt = [], [], [], [], []
            for pt in range(start_pt, end_pt):
                _trj_idx, _seg_idx = trj_mapper[pt]
                if _seg_idx < seq_len - 1:
                    __input_obs = np.concatenate([np.zeros((seq_len - 1 - _seg_idx, observation_dim)),
                                                  trajectories[_trj_idx][0][:_seg_idx + 1, :]], axis=0)
                    __input_act = np.concatenate([np.zeros((seq_len - 1 - _seg_idx, action_dim)),
                                                  trajectories[_trj_idx][1][:_seg_idx + 1, :]], axis=0)
                    __input_timestep = np.concatenate([np.zeros(seq_len - 1 - _seg_idx, dtype=np.int32),
                                                       np.arange(1, _seg_idx + 2, dtype=np.int32)], axis=0)
                    __input_attn_mask = np.concatenate(
                        [np.zeros(seq_len - 1 - _seg_idx, dtype=np.int32), np.ones(_seg_idx + 1, dtype=np.float32)],
                        axis=0)
                    __input_pt = np.concatenate(
                        [np.zeros(seq_len - 1 - _seg_idx), np.arange(pt - _seg_idx, pt + 1)], axis=0)
                else:
                    __input_obs = trajectories[_trj_idx][0][_seg_idx - seq_len + 1:_seg_idx + 1, :]
                    __input_act = trajectories[_trj_idx][1][_seg_idx - seq_len + 1:_seg_idx + 1, :]
                    __input_timestep = np.arange(1, seq_len + 1, dtype=np.int32)
                    __input_attn_mask = np.ones((seq_len), dtype=np.float32)
                    __input_pt = np.arange(pt - seq_len + 1, pt + 1)

                _input_obs.append(__input_obs)
                _input_act.append(__input_act)
                _input_timestep.append(__input_timestep)
                _input_attn_mask.append(__input_attn_mask)
                _input_pt.append(__input_pt)

            _input_obs = np.asarray(_input_obs)
            _input_act = np.asarray(_input_act)
            _input_timestep = np.asarray(_input_timestep)
            _input_attn_mask = np.asarray(_input_attn_mask)
            _input_pt = np.asarray(_input_pt)

            input = dict(
                observations=_input_obs,
                actions=_input_act,
                timestep=_input_timestep,
                attn_mask=_input_attn_mask,
            )

            jax_input = batch_to_jax(input)
            if with_attn_weights:
                new_reward, attn_weight = reward_model.get_reward(jax_input)
                attn_weights.append(np.array(attn_weight))
                pts.append(_input_pt)
            else:
                new_reward, _ = reward_model.get_reward(jax_input)
            new_reward = new_reward.reshape(end_pt - start_pt, seq_len) * _input_attn_mask

            # print("new_reward1", new_reward.shape, new_reward[0])

            # if label_mode == "mean":
            #     new_reward = jnp.sum(new_reward, axis=1) / jnp.sum(_input_attn_mask, axis=1)
            #     new_reward = new_reward.reshape(-1, 1)
            # elif label_mode == "last":
            #     new_reward = new_reward[:, -1].reshape(-1, 1)

            new_reward = new_reward[:, -1].reshape(-1, 1)

            # print("new_reward2", new_reward.shape, new_reward[0])

            new_reward = np.asarray(list(new_reward))
            new_r[start_pt:end_pt, ...] = new_reward.squeeze(-1)

            # print("new_reward3", new_reward.shape, new_reward[0])
            # print("new_reward4", start_pt, end_pt, end_pt-start_pt, new_r.shape, new_r[start_pt:end_pt, ...])

        self.rewards_pred = new_r.copy()

        # rewards_pred_last = self.rewards_pred[-100:]
        # real_rewards_last = self.real_rewards[-100:]
        # rewards_last = self.rewards[-100:]
        #
        # with open('output_100.txt', 'w') as f:
        #     f.write(' '.join(map(str, rewards_pred_last)) + '\n')
        #     f.write(' '.join(map(str, real_rewards_last)) + '\n')
        #     f.write(' '.join(map(str, rewards_last)) + '\n')
        #
        # return 0


        # if with_attn_weights:
        #     return dataset, (attn_weights, pts)
        # return dataset