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
from jaxrl.datasets import ReplayBuffer_Reward_Model
from jaxrl.evaluation import evaluate
from jaxrl.utils import make_env

from collections import defaultdict
import transformers
from BaggedRewardModel.Transformer import BaggedRewardsTransformer
from flaxmodels.flaxmodels.gpt2.trajectory_gpt2 import TransRewardModel
from BaggedRewardModel.utils import Timer, define_flags_with_default, set_random_seed, get_user_flags, prefix_metrics, WandBLogger, save_pickle
from BaggedRewardModel.jax_utils import batch_to_jax


FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp_test/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('updates_per_step', 1000, 'Gradient updates per step.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('track', False, 'Track experiments with Weights and Biases.')
flags.DEFINE_string('wandb_project_name', "jaxrl", "The wandb's project name.")
flags.DEFINE_string('wandb_entity', None, "the entity (team) of wandb's project")

flags.DEFINE_boolean('bagged_algorithm', True, "run bagged reward algorithm od not (True: our algorithm; False: original algorithm.)")
flags.DEFINE_integer('bag_len', 100, "length of reward bag in learning from reward bag setting")
flags.DEFINE_boolean('traj_feedback', False, "True: learning from traj, False: learning from bagged reward")

flags.DEFINE_integer('start_training_reward_model', int(1e4), "Number of training steps of reward model to start training.")
flags.DEFINE_integer('reward_model_interval', 4, "training intervals per update step")
flags.DEFINE_integer('sequence_length', 500, "length of sampled sequence")
flags.DEFINE_integer('reward_model_epochs', 2000, "reward model epochs")
flags.DEFINE_integer('update_reward_model', 1000, "frequency of updating reward model")
flags.DEFINE_integer('batch_size_reward_model', 64, "batch size of updating reward model")
flags.DEFINE_string('activations', 'relu', "activations for reward model")
flags.DEFINE_string('activation_final', 'none', "activation_final for reward model")
flags.DEFINE_integer('seq_len', 100, 'sequence length for relabeling reward in Transformer.')

config_flags.DEFINE_config_file(
    'config',
    'examples/configs/sac_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    kwargs = dict(FLAGS.config)
    algo = kwargs.pop('algo')
    current_time = int(time.time())
    run_name = f"{algo}_{FLAGS.env_name}_{FLAGS.bag_len}_{FLAGS.seed}_{current_time}"
    save_path = os.path.join(FLAGS.save_dir, f'{FLAGS.env_name}', f'{FLAGS.bag_len}',
                             f'sequence_length_{FLAGS.sequence_length}^seq_len_{FLAGS.seq_len}',
                             'return_file', f'{FLAGS.seed}_{current_time}.txt')
    save_path_model = os.path.join(FLAGS.save_dir, f'{FLAGS.env_name}', f'{FLAGS.bag_len}',
                             f'{FLAGS.seed}_{current_time}')

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
        os.path.join(FLAGS.save_dir, f'{FLAGS.env_name}', f'{FLAGS.bag_len}',
                     f'sequence_length_{FLAGS.sequence_length}^seq_len_{FLAGS.seq_len}', run_name))

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
    set_random_seed(FLAGS.seed)

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

    replay_buffer = ReplayBuffer_Reward_Model(env.observation_space, env.action_space,
                                 replay_buffer_size or FLAGS.max_steps)

    eval_returns = []
    observation, done = env.reset(), False
    trajectory = []
    bag = []

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    total_epochs = FLAGS.reward_model_epochs
    interval = FLAGS.reward_model_interval
    get_config_t = BaggedRewardsTransformer.get_default_config()
    config_t = transformers.GPT2Config(**get_config_t)
    config_t.warmup_steps = int(total_epochs * 0.1 * interval)
    config_t.total_steps = total_epochs * interval
    trans = TransRewardModel(config=config_t, observation_dim=observation_dim, action_dim=action_dim,
                             activation=FLAGS.activations, activation_final=FLAGS.activation_final)
    reward_model = BaggedRewardsTransformer(config_t, trans)

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):

        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

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

        else:
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

            if FLAGS.traj_feedback:
                trajectory.append(transition)
                if done or (i == FLAGS.max_steps):
                    trajectory[-1]['bag_end'] = 1
                    mean_reward = sum([t['real_rews'] for t in trajectory]) / len(trajectory)
                    for trans in trajectory:
                        trans['rews'] = mean_reward

                    replay_buffer.insert_traj(trajectory)
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

            else:
                bag.append(transition)

                if (not done) and (len(bag) == FLAGS.bag_len):

                    bag[-1]['bag_end'] = 1
                    mean_reward = sum([t['real_rews'] for t in bag]) / len(bag)
                    for trans in bag:
                        trans['rews'] = mean_reward

                    replay_buffer.insert_traj(bag)
                    bag = []

                if (done or (i == FLAGS.max_steps)) and (len(bag) != 0):

                    bag[-1]['bag_end'] = 1
                    mean_reward = sum([t['real_rews'] for t in bag]) / len(bag)

                    for trans in bag:
                        trans['rews'] = mean_reward

                    replay_buffer.insert_traj(bag)
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

            if (i >= FLAGS.start_training_reward_model) and (i % FLAGS.update_reward_model == 0):
                metrics = defaultdict(list)

                if i == FLAGS.start_training_reward_model:
                    update_reward_model = int(FLAGS.update_reward_model / 10)
                else:
                    update_reward_model = int(FLAGS.update_reward_model / 100)

                for _ in range(update_reward_model):

                    samples_index = replay_buffer.size - FLAGS.sequence_length
                    num_samples = int(FLAGS.batch_size_reward_model * FLAGS.reward_model_interval)
                    training_indices = random.sample(range(samples_index), num_samples)
                    training_dataset = replay_buffer.sample_sequences(num_samples, FLAGS.sequence_length, training_indices)
                    shuffled_idx = np.random.permutation(training_dataset["observations"].shape[0])

                    for j in range(FLAGS.reward_model_interval):
                        start_rt = j * FLAGS.batch_size_reward_model
                        end_rt = min((j + 1) * FLAGS.batch_size_reward_model, training_dataset["observations"].shape[0])

                        batch = batch_to_jax(replay_buffer.index_batch(training_dataset, shuffled_idx[start_rt:end_rt]))
                        for key, val in prefix_metrics(reward_model.train(batch), 'reward').items():
                            metrics[key].append(val)


            if (i >= FLAGS.start_training) and (i % FLAGS.updates_per_step == 0):
                replay_buffer.recompute_rewards(reward_model, seq_len=FLAGS.seq_len)

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
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savetxt(save_path,
                           eval_returns,
                           fmt=['%d', '%.1f'])

            if i == FLAGS.max_steps:
                os.makedirs(save_path_model, exist_ok=True)
                save_data = {"reward_model": reward_model, "agent_model": agent}
                save_pickle(save_data, 'model.pkl', save_path_model)
                return 0



if __name__ == '__main__':
    app.run(main)
