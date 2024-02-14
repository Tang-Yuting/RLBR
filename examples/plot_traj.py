import gym
from gym.wrappers.record_video import RecordVideo
from PIL import Image
import pickle
import os
import numpy as np
from jaxrl.datasets import ReplayBuffer_Reward_Model
import tqdm
import time
import glob

max_step = 1000
seq_len = 100
bag_lens = {100}
env_name = 'Hopper-v2'
seed = 0

# work on ke5

for bag_len in bag_lens:

    print(env_name, "bag_len", bag_len)

    current_time = int(time.time())

    # save_path_model = f'/workspace/jaxrl-main/result_kserver/tmp_result_next_state_sum/{env_name}/{bag_len}/0_*'
    file_paths = glob.glob(f'/workspace/jaxrl-main/result_kserver/tmp_result_next_state_sum/{env_name}/{bag_len}/{seed}_*')

    if not file_paths:
        print("File not found. Please check the path and try again.")
    else:
        save_path_model = file_paths[0]
        try:
            with open(os.path.join(save_path_model, 'model.pkl'), 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            print("File not found. Please check the path and try again.")
            exit()
    #
    # try:
    #     with open(os.path.join(save_path_model, 'model.pkl'), 'rb') as f:
    #         data = pickle.load(f)
    # except FileNotFoundError:
    #     print("File not found. Please check the path and try again.")
    #     exit()

    reward_model = data['reward_model']
    agent = data['agent_model']

    env = gym.make(env_name)

    # save file
    save_file_path = f'traj_video/{env_name}/{bag_len}_{current_time}'
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)

    # video
    video_path = os.path.join(save_file_path, 'video')
    env = RecordVideo(env, video_path,  episode_trigger = lambda episode_number: True)
    env.reset()

    observation = env.reset()
    done = False

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    traj = ReplayBuffer_Reward_Model(env.observation_space, env.action_space, max_step)
    bag = []

    for i in tqdm.tqdm(range(1, max_step + 1),
                       smoothing=0.1,
                       disable=not True):
        action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        img = env.render(mode='rgb_array')
        Image.fromarray(img).save(f"{save_file_path}/frame_{i}.png")

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0
        transition = {
            'obs': observation,
            'obs_next': next_observation,
            'acts': action,
            'real_rews': reward,
            'done': float(done),
            'mask': mask,
            'bag_end': 0,
            'rews': reward
        }
        observation = next_observation
        bag.append(transition)

        if (done or (i == max_step)) and (len(bag) != 0):

            bag[-1]['bag_end'] = 1
            mean_reward = sum([t['real_rews'] for t in bag]) / len(bag)

            for trans in bag:
                trans['rews'] = mean_reward

            traj.insert_traj(bag)
            break

        if (not done) and (len(bag) == bag_len):
            # print("add bag", i)
            # print("reward_1", [t['rews'] for t in bag])

            bag[-1]['bag_end'] = 1
            mean_reward = sum([t['real_rews'] for t in bag]) / len(bag)

            for trans in bag:
                trans['rews'] = mean_reward

            traj.insert_traj(bag)
            # print("reward_2", [t['rews'] for t in bag])
            bag = []

    traj.recompute_rewards(reward_model, seq_len)

    # print("reward", traj.rewards)
    # print("real_reward", traj.real_rewards)
    # print("pred_reward", traj.rewards_pred)

    buffer_path = os.path.join(save_file_path, 'replay_buffer.pkl')
    try:
        with open(buffer_path, 'wb') as f:
            pickle.dump(traj, f)
    except IOError:
        print("Error while writing the replay buffer file.")

    env.close()










