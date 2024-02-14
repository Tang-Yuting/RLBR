import os
import random
import time

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from jaxrl.agents import (AWACLearner, DDPGLearner, REDQLearner, SACLearner,
                          SACV1Learner)
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate
from jaxrl.utils import make_env

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp_sac_1000/', 'Tensorboard logging dir.')  # TT: file name of saving results
flags.DEFINE_integer('seed', 45, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
# flags.DEFINE_integer('updates_per_step', 1, 'Gradient updates per step.')  # original
flags.DEFINE_integer('updates_per_step', 1000, 'Gradient updates per step.')  # TT: updates_per_step = bag_len
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('track', False, 'Track experiments with Weights and Biases.')
flags.DEFINE_string('wandb_project_name', "jaxrl", "The wandb's project name.")
flags.DEFINE_string('wandb_entity', None, "the entity (team) of wandb's project")

# TT_modification (bagged reward)
flags.DEFINE_boolean('bagged_algorithm', True, "run bagged reward algorithm od not (True: our algorithm; False: original algorithm.)")
flags.DEFINE_integer('bag_len', 100, "length of reward bag in learning from reward bag setting")
flags.DEFINE_boolean('traj_feedback', False, "True: learning from traj, False: learning from bagged reward")
flags.DEFINE_string('bag_type', None, "bag type: ave, sum, ircr")

config_flags.DEFINE_config_file(
    'config',
    'examples/configs/sac_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    kwargs = dict(FLAGS.config)
    algo = kwargs.pop('algo')
    # TT: file name of Tensorboard
    current_time = int(time.time())
    run_name = f"{algo}_{FLAGS.env_name}_{FLAGS.bag_type}_{FLAGS.bag_len}_{FLAGS.seed}_{current_time}"
    if FLAGS.track:
        import wandb

        wandb.init(
            project=FLAGS.wandb_project_name,
            entity=FLAGS.wandb_entity,
            sync_tensorboard=True,
            config=FLAGS,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.config.update({"algo": algo})
    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, f'{FLAGS.env_name}', f'{FLAGS.bag_type}', f'{FLAGS.bag_len}', run_name))

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    env = make_env(FLAGS.env_name, FLAGS.seed, video_train_folder)
    eval_env = make_env(FLAGS.env_name, FLAGS.seed + 42, video_eval_folder)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)


    replay_buffer_size = kwargs.pop('replay_buffer_size')
    if algo == 'sac':
        agent = SACLearner(FLAGS.seed,
                           env.observation_space.sample()[np.newaxis],
                           env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'redq':
        agent = REDQLearner(FLAGS.seed,
                            env.observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis],
                            policy_update_delay=FLAGS.updates_per_step,
                            **kwargs)
    elif algo == 'sac_v1':
        agent = SACV1Learner(FLAGS.seed,
                             env.observation_space.sample()[np.newaxis],
                             env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'awac':
        agent = AWACLearner(FLAGS.seed,
                            env.observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'ddpg':
        agent = DDPGLearner(FLAGS.seed,
                            env.observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis], **kwargs)
    else:
        raise NotImplementedError()

    replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                 replay_buffer_size or FLAGS.max_steps)

    eval_returns = []
    observation, done = env.reset(), False

    # TT_modification (bagged reward)
    trajectory = []
    bag = []

    min_reward, max_reward = 50000, -50000

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):

        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        # IPython.embed()

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        if FLAGS.bagged_algorithm == False:
            replay_buffer.insert(observation, action, reward, mask, float(done),
                                 next_observation)
            observation = next_observation

            if done:
                observation, done = env.reset(), False
                for k, v in info['episode'].items():
                    summary_writer.add_scalar(f'training/{k}', v,
                                              info['total']['timesteps'])

                if 'is_success' in info:
                    summary_writer.add_scalar(f'training/success',
                                              info['is_success'],
                                              info['total']['timesteps'])

            if i >= FLAGS.start_training:
                for _ in range(FLAGS.updates_per_step):
                    batch = replay_buffer.sample(FLAGS.batch_size)
                    update_info = agent.update(batch)

                if i % FLAGS.log_interval == 0:
                    for k, v in update_info.items():
                        summary_writer.add_scalar(f'training/{k}', v, i)
                    summary_writer.flush()

            if i % FLAGS.eval_interval == 0:
                eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

                for k, v in eval_stats.items():
                    summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                              info['total']['timesteps'])
                summary_writer.flush()

                eval_returns.append(
                    (info['total']['timesteps'], eval_stats['return']))
                np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                           eval_returns,
                           fmt=['%d', '%.1f'])

        # TT_modification (bagged reward)
        else:
            transition = {
                'obs': observation,
                'obs_next': next_observation,
                'acts': action,
                'rews': reward,
                'done': float(done),
                'mask': mask
            }
            observation = next_observation

            if FLAGS.traj_feedback:  # learning from traj feedback
                trajectory.append(transition)
                # print("trajectory", len(trajectory), [t['rews'] for t in trajectory])

                if done or (i == FLAGS.max_steps):
                    # (done) TODO: add ave or sum bagged reward information
                    if FLAGS.bag_type == 'ave':
                        mean_reward = sum([t['rews'] for t in trajectory]) / len(trajectory)
                        for trans in trajectory:
                            trans['rews'] = mean_reward
                    elif FLAGS.bag_type == 'sum':
                        total_reward = sum([t['rews'] for t in trajectory])
                        for trans in trajectory[:-1]:
                            trans['rews'] = 0
                        trajectory[-1]['rews'] = total_reward

                    replay_buffer.insert_traj(trajectory)

                    # print("traj_reward", len(trajectory), [t['rews'] for t in trajectory])
                    # print("add traj step:", i)

                    trajectory = []
                    observation, done = env.reset(), False

                    if 'episode' in info:
                        for k, v in info['episode'].items():
                            summary_writer.add_scalar(f'training/{k}', v,
                                                      info['total']['timesteps'])

                    if 'is_success' in info:
                        summary_writer.add_scalar(f'training/success',
                                                  info['is_success'],
                                                  info['total']['timesteps'])

            else:  # learning from bagged reward
                bag.append(transition)
                # print("bag", len(bag), [t['rews'] for t in bag])

                if (not done) and (len(bag) == FLAGS.bag_len):
                    # print("add bag", i)
                    # print("reward_1", [t['rews'] for t in bag])

                    # (done) TODO: add ave or sum bagged reward information
                    if FLAGS.bag_type == 'ave':
                        # sum
                        mean_reward = sum([t['rews'] for t in bag]) / len(bag)
                        # max
                        # mean_reward = max([t['rews'] for t in bag]) / len(bag)
                        # max * length
                        # mean_reward = max([t['rews'] for t in bag])
                        for trans in bag:
                            trans['rews'] = mean_reward
                    elif FLAGS.bag_type == 'sum':
                        # sum
                        total_reward = sum([t['rews'] for t in bag])
                        # max
                        # total_reward = max([t['rews'] for t in bag])
                        # max * length
                        # total_reward = max([t['rews'] for t in bag]) * len(bag)

                        for trans in bag[:-1]:
                            trans['rews'] = 0
                        bag[-1]['rews'] = total_reward
                    elif FLAGS.bag_type == 'ircr':
                        # sum
                        mean_reward = sum([t['rews'] for t in bag])
                        # max
                        # mean_reward = max([t['rews'] for t in bag])
                        # max * length
                        # mean_reward = max([t['rews'] for t in bag]) * len(bag)

                        min_reward = min(min_reward, mean_reward)
                        max_reward = max(max_reward, mean_reward)

                        # print("reward", [t['rews'] for t in bag])
                        # print("min_reward", min_reward, "max_reward", max_reward, "mean_reward", mean_reward)
                        for trans in bag:
                            trans['rews'] = mean_reward
                            # print("trans['rews']", trans['rews'])


                    replay_buffer.insert_traj(bag)
                    # print("reward_2", [t['rews'] for t in bag])
                    bag = []

                if done or (i == FLAGS.max_steps):
                    # (done) TODO: add ave or sum bagged reward information
                    if len(bag) != 0:
                        if FLAGS.bag_type == 'ave':
                            # sum
                            mean_reward = sum([t['rews'] for t in bag]) / len(bag)
                            # max
                            # mean_reward = max([t['rews'] for t in bag]) / len(bag)
                            # max * length
                            # mean_reward = max([t['rews'] for t in bag])

                            for trans in bag:
                                trans['rews'] = mean_reward
                        elif FLAGS.bag_type == 'sum':
                            # sum
                            total_reward = sum([t['rews'] for t in bag])
                            # max
                            # total_reward = max([t['rews'] for t in bag])
                            # max * length
                            # total_reward = max([t['rews'] for t in bag]) * len(bag)

                            for trans in bag[:-1]:
                                trans['rews'] = 0
                            bag[-1]['rews'] = total_reward
                        elif FLAGS.bag_type == 'ircr':
                            # sum
                            mean_reward = sum([t['rews'] for t in bag])
                            # max
                            # mean_reward = max([t['rews'] for t in bag])
                            # max * length
                            # mean_reward = max([t['rews'] for t in bag]) * len(bag)

                            min_reward = min(min_reward, mean_reward)
                            max_reward = max(max_reward, mean_reward)

                            # print("reward", [t['rews'] for t in bag])
                            # print("min_reward", min_reward, "max_reward", max_reward, "mean_reward", mean_reward)
                            for trans in bag:
                                trans['rews'] = mean_reward
                                # print("trans['rews']", trans['rews'])

                    replay_buffer.insert_traj(bag)
                    # print("reward_2", [t['rews'] for t in bag])
                    bag = []
                    observation, done = env.reset(), False

                    if 'episode' in info:
                        for k, v in info['episode'].items():
                            summary_writer.add_scalar(f'training/{k}', v,
                                                      info['total']['timesteps'])

                    if 'is_success' in info:
                        summary_writer.add_scalar(f'training/success',
                                                  info['is_success'],
                                                  info['total']['timesteps'])

            if (i >= FLAGS.start_training) and (i % FLAGS.updates_per_step == 0):
                # print("agent update", i)
                # count = 0
                for _ in range(FLAGS.updates_per_step):
                    # print("count", count)
                    # count = count + 1
                    batch = replay_buffer.sample(FLAGS.batch_size)
                    if FLAGS.bag_type == 'ircr':
                        # print("min_reward, max_reward", min_reward, max_reward)
                        # print("reward", batch.rewards)
                        ircr_batch = replay_buffer.ircr_rewards(batch, min_reward, max_reward)
                        # print("ircr_reward", ircr_batch.rewards)
                        update_info = agent.update(ircr_batch)
                    else:
                        update_info = agent.update(batch)

                if i % FLAGS.log_interval == 0:
                    for k, v in update_info.items():
                        summary_writer.add_scalar(f'training/{k}', v, i)
                    summary_writer.flush()

            if i % FLAGS.eval_interval == 0:
                # print("eval:", i)
                eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

                for k, v in eval_stats.items():
                    summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                              info['total']['timesteps'])
                summary_writer.flush()

                eval_returns.append(
                    (info['total']['timesteps'], eval_stats['return']))

                # TT: txt file, eval return
                save_path = os.path.join(FLAGS.save_dir, f'{FLAGS.env_name}', f'{FLAGS.bag_type}', f'{FLAGS.bag_len}',
                                         'return_file', f'{FLAGS.seed}_{current_time}.txt')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savetxt(save_path,
                           eval_returns,
                           fmt=['%d', '%.1f'])



if __name__ == '__main__':
    app.run(main)
