import os
import h5py
import pickle
from tqdm import tqdm
import numpy as np
import ujson as json
import jax.numpy as jnp
from itertools import chain


def get_goal(name):
    if 'large' in name:
        return (32.0, 24.0)
    elif 'medium' in name:
        return (20.0, 20.0)
    elif 'umaze' in name:
        return (0.0, 8.0)
    return None


def new_get_trj_idx(env, terminate_on_end=False, **kwargs):
    if not hasattr(env, 'get_dataset'):
        dataset = kwargs['dataset']
    else:
        dataset = env.get_dataset()
    N = dataset['rewards'].shape[0]

    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    start_idx, data_idx = 0, 0
    trj_idx_list = []
    for i in range(N - 1):
        if env.spec and 'maze' in env.spec.id:
            done_bool = sum(dataset['infos/goal'][i + 1] - dataset['infos/goal'][i]) > 0
        else:
            done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            trj_idx_list.append([start_idx, data_idx - 1])
            start_idx = data_idx
            continue
        if done_bool or final_timestep:
            episode_step = 0
            trj_idx_list.append([start_idx, data_idx])
            start_idx = data_idx + 1

        episode_step += 1
        data_idx += 1

    trj_idx_list.append([start_idx, data_idx])

    return trj_idx_list


def find_time_idx(trj_idx_list, idx):
    for (start, end) in trj_idx_list:
        if start <= idx <= end:
            return idx - start


def load_queries(env, dataset, num_query, len_query, sampled_indices, bag_len):
    trj_idx_list = new_get_trj_idx(env, dataset=dataset)
    trj_idx_list = np.array(trj_idx_list)
    trj_len_list = trj_idx_list[:, 1] - trj_idx_list[:, 0] + 1
    assert max(trj_len_list) > len_query

    observation_dim = dataset["observations"].shape[-1]
    action_dim = dataset["actions"].shape[-1]

    total_reward_seq = np.zeros((num_query, len_query))
    total_ave_reward_seq = np.zeros((num_query, len_query))
    total_sum_reward_seq = np.zeros((num_query, len_query))
    total_obs_seq = np.zeros((num_query, len_query, observation_dim))
    total_next_obs_seq = np.zeros((num_query, len_query, observation_dim))
    total_act_seq = np.zeros((num_query, len_query, action_dim))
    total_done_seq = np.zeros((num_query, len_query), dtype=np.int32)
    total_timestep = np.zeros((num_query, len_query), dtype=np.int32)
    query_range = np.arange(num_query)

    total_idx = np.zeros(num_query, dtype=np.int32)

    if bag_len == -1:
        dataset_sum_reward = sum_traj_reward(dataset, env, bag_len)
        dataset_ave_reward = ave_traj_reward(dataset_sum_reward)
    else:
        dataset_sum_reward = sum_bagged_reward(dataset, env, bag_len)
        dataset_ave_reward = ave_bagged_reward(dataset_sum_reward)

    for query_count, i in enumerate(tqdm(query_range, desc="get queries from saved indices")):

        start_idx = sampled_indices[i]
        end_idx = start_idx + len_query

        reward_seq = dataset['rewards'][start_idx:end_idx]
        ave_reward_seq = dataset_ave_reward[start_idx:end_idx]
        sum_reward_seq = dataset_sum_reward[start_idx:end_idx]
        obs_seq = dataset['observations'][start_idx:end_idx]
        next_obs_seq = dataset['next_observations'][start_idx:end_idx]
        act_seq = dataset['actions'][start_idx:end_idx]
        done_seq = dataset['dones'][start_idx:end_idx]

        total_reward_seq[query_count] = reward_seq
        total_ave_reward_seq[query_count] = ave_reward_seq
        total_sum_reward_seq[query_count] = sum_reward_seq
        total_obs_seq[query_count] = obs_seq
        total_next_obs_seq[query_count] = next_obs_seq
        total_act_seq[query_count] = act_seq
        total_done_seq[query_count] = done_seq

        total_idx[query_count] = start_idx

    seg_reward = total_reward_seq.copy()
    seg_ave_reward = total_ave_reward_seq.copy()
    seg_sum_reward = total_sum_reward_seq.copy()
    seg_obs = total_obs_seq.copy()
    seg_next_obs = total_next_obs_seq.copy()
    seq_act = total_act_seq.copy()
    seq_done = total_done_seq.copy()
    seq_timestep = total_timestep.copy()

    batch = {}

    batch['rewards'] = seg_reward
    batch['ave_rewards'] = seg_ave_reward
    batch['sum_rewards'] = seg_sum_reward
    batch['observations'] = seg_obs
    batch['next_observations'] = seg_next_obs
    batch['actions'] = seq_act
    batch['dones'] = seq_done
    batch['timestep'] = seq_timestep
    batch['start_indices'] = sampled_indices

    return batch

def load_all_queries(env, dataset, num_query, len_query, bag_len):
    trj_idx_list = new_get_trj_idx(env, dataset=dataset)
    trj_idx_list = np.array(trj_idx_list)
    trj_len_list = trj_idx_list[:, 1] - trj_idx_list[:, 0] + 1
    assert max(trj_len_list) > len_query

    if bag_len == -1:
        dataset_sum_reward = sum_traj_reward(dataset, env, bag_len)
        dataset_ave_reward = ave_traj_reward(dataset_sum_reward)
    else:
        dataset_sum_reward = sum_bagged_reward(dataset, env, bag_len)
        dataset_ave_reward = ave_bagged_reward(dataset_sum_reward)

    return dataset_sum_reward, dataset_ave_reward


def load_batch_samples(env, dataset, num_query, len_query, dataset_sum_reward, dataset_ave_reward, sampled_indices):
    observation_dim = dataset["observations"].shape[-1]
    action_dim = dataset["actions"].shape[-1]

    total_reward_seq = np.zeros((num_query, len_query))
    total_ave_reward_seq = np.zeros((num_query, len_query))
    total_sum_reward_seq = np.zeros((num_query, len_query))
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

        reward_seq = dataset['rewards'][start_idx:end_idx]
        ave_reward_seq = dataset_ave_reward[start_idx:end_idx]
        sum_reward_seq = dataset_sum_reward[start_idx:end_idx]
        obs_seq = dataset['observations'][start_idx:end_idx]
        next_obs_seq = dataset['next_observations'][start_idx:end_idx]
        act_seq = dataset['actions'][start_idx:end_idx]
        done_seq = dataset['dones'][start_idx:end_idx]

        total_reward_seq[query_count] = reward_seq
        total_ave_reward_seq[query_count] = ave_reward_seq
        total_sum_reward_seq[query_count] = sum_reward_seq
        total_obs_seq[query_count] = obs_seq
        total_next_obs_seq[query_count] = next_obs_seq
        total_act_seq[query_count] = act_seq
        total_done_seq[query_count] = done_seq

        total_idx[query_count] = start_idx

    seg_reward = total_reward_seq.copy()
    seg_ave_reward = total_ave_reward_seq.copy()
    seg_sum_reward = total_sum_reward_seq.copy()
    seg_obs = total_obs_seq.copy()
    seg_next_obs = total_next_obs_seq.copy()
    seq_act = total_act_seq.copy()
    seq_done = total_done_seq.copy()
    seq_timestep = total_timestep.copy()

    batch = {}

    batch['rewards'] = seg_reward
    batch['ave_rewards'] = seg_ave_reward
    batch['sum_rewards'] = seg_sum_reward
    batch['observations'] = seg_obs
    batch['next_observations'] = seg_next_obs
    batch['actions'] = seq_act
    batch['dones'] = seq_done
    batch['timestep'] = seq_timestep
    batch['start_indices'] = sampled_indices

    return batch


def split_into_trajectories(observations, actions, rewards, dones,
                            next_observations):
    trajs = [[]]
    count_num = 0

    for i in tqdm(range(len(observations)), desc="split"):
        trajs[-1].append((observations[i], actions[i], rewards[i], dones[i], next_observations[i]))
        if dones[i] == 1 and i + 1 < len(observations):
            trajs.append([])
        if dones[i] != 0:
            count_num = count_num + 1
    print("count_num", count_num)
    # print("len(trajs)", len(trajs))
    return trajs


def sum_bagged_reward(dataset, env_name, bag_len):
    trajs = split_into_trajectories(
        dataset['observations'],
        dataset['actions'],
        dataset['rewards'],
        dataset['dones'],
        dataset['next_observations']
    )

    bagged_rewards = []

    for trj_idx, traj in tqdm(enumerate(trajs), total=len(trajs), desc="sum bagged reward trajectories"):
        traj_len = len(traj)
        rewards_for_traj = []
        for item in traj:
            reward = item[2]
            rewards_for_traj.append(reward)
        bagged_data_sum = np.copy(rewards_for_traj)
        # print("rewards", bagged_data_sum, len(bagged_data_sum))
        print("bagged_rewards1", bagged_data_sum[-200:])

        b_sum = 0
        if len(bagged_data_sum) % bag_len == 0:
            b_sum = (int(len(bagged_data_sum) / bag_len) - 1) * bag_len
        else:
            b_sum = (int(len(bagged_data_sum) / bag_len)) * bag_len
        bagged_data_sum[-1] = max(rewards_for_traj[b_sum:])

        for rwd in range(len(bagged_data_sum) - 1):
            if (rwd % bag_len) == (bag_len - 1):
                bagged_data_sum[rwd] = np.max(
                    bagged_data_sum[rwd - bag_len + 1:rwd + 1])
        for rwd in range(len(bagged_data_sum) - 1):
            if (rwd % bag_len) != (bag_len - 1):
                bagged_data_sum[rwd] = 0

        bagged_rewards.append(bagged_data_sum)

    bagged_rewards = list(chain(*bagged_rewards))
    print("bagged_rewards2", bagged_rewards[-200:], len(bagged_rewards))
    return np.array(bagged_rewards)


def ave_bagged_reward(x):
    ave = np.zeros_like(x)
    bag_len, i = 0, 0
    end = len(x)
    while(i < end):
        if x[i] == 0:
            bag_len = bag_len + 1
        else:
            bag_len = bag_len + 1
            ave[i - bag_len + 1:i + 1] = x[i] / bag_len
            bag_len = 0
        i = i + 1
    return ave


def sum_traj_reward(dataset, env_name):
    # print("sum_bagged_reward", dataset.keys())
    # sum_bagged_reward dict_keys(['observations', 'actions', 'next_observations', 'rewards', 'dones'])
    trajs = split_into_trajectories(
        dataset['observations'],
        dataset['actions'],
        dataset['rewards'],
        dataset['dones'],
        dataset['next_observations']
    )

    bagged_rewards = []

    for trj_idx, traj in tqdm(enumerate(trajs), total=len(trajs), desc="sum traj reward trajectories"):
        traj_len = len(traj)
        rewards_for_traj = []
        for item in traj:
            reward = item[2]
            rewards_for_traj.append(reward)
        bagged_data_sum = np.copy(rewards_for_traj)
        bagged_data_sum[-1] = np.max(bagged_data_sum)
        bagged_data_sum[:-1] = 0

        bagged_rewards.append(bagged_data_sum)
    bagged_rewards = list(chain(*bagged_rewards))
    return np.array(bagged_rewards)


def ave_traj_reward(x):
    ave = np.zeros_like(x)
    bag_len, i = 0, 0
    end = len(x)
    while(i < end):
        if x[i] == 0:
            bag_len = bag_len + 1
        else:
            bag_len = bag_len + 1
            ave[i - bag_len + 1:i + 1] = x[i] / bag_len
            bag_len = 0
        i = i + 1
    return ave