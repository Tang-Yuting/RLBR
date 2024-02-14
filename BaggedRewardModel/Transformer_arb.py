from functools import partial
import os

# import IPython
from ml_collections import ConfigDict

import jax
import jax.numpy as jnp

import optax
import numpy as np
from flax.training.train_state import TrainState

from .jax_utils import next_rng, value_and_multi_grad, mse_loss, cross_ent_loss, kld_loss, bagged_reward_loss, state_loss_func, bagged_reward_arbitrary_loss


@jax.jit
def numpy_callback(x):
  result_shape = jax.core.ShapedArray(x.shape, x.dtype)
  return jax.custom_jvp(np.sin, result_shape, x)


class BaggedRewardsTransformer(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.trans_lr = 1e-4
        config.optimizer_type = 'adamw'
        config.scheduler_type = 'CosineDecay'
        config.vocab_size = 1
        config.n_layer = 3
        config.embd_dim = 256
        config.n_embd = config.embd_dim
        config.n_head = 1
        config.pref_attn_n_head = config.n_head
        config.n_positions = 1024
        config.resid_pdrop = 0.1
        config.attn_pdrop = 0.1
        config.pref_attn_embd_dim = 256
        config.train_type = "mean"
        config.use_weighted_sum = True
        config.file_bag_len = 1
        config.file_name = 'hopper-medium-replay'
        config.file_seed = 0


        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, trans):
        self.config = config
        self.trans = trans
        self.observation_dim = trans.observation_dim
        self.action_dim = trans.action_dim

        self._train_states = {}

        optimizer_class = {
            'adam': optax.adam,
            'adamw': optax.adamw,
            'sgd': optax.sgd,
        }[self.config.optimizer_type]

        scheduler_class = {
            'CosineDecay': optax.warmup_cosine_decay_schedule(
                init_value=self.config.trans_lr,
                peak_value=self.config.trans_lr * 10,
                warmup_steps=self.config.warmup_steps,
                decay_steps=self.config.total_steps,
                end_value=self.config.trans_lr
            ),
            "OnlyWarmup": optax.join_schedules(
                [
                    optax.linear_schedule(
                        init_value=0.0,
                        end_value=self.config.trans_lr,
                        transition_steps=self.config.warmup_steps,
                    ),
                    optax.constant_schedule(
                        value=self.config.trans_lr
                    )
                ],
                [self.config.warmup_steps]
            ),
            'none': None
        }[self.config.scheduler_type]

        if scheduler_class:
            tx = optimizer_class(scheduler_class)
        else:
            tx = optimizer_class(learning_rate=self.config.trans_lr)

        trans_params = self.trans.init(
            {"params": next_rng(), "dropout": next_rng()},
            jnp.zeros((10, 100, self.observation_dim)),
            jnp.zeros((10, 100, self.action_dim)),
            jnp.ones((10, 100), dtype=jnp.int32)
        )
        self._train_states['trans'] = TrainState.create(
            params=trans_params,
            tx=tx,
            apply_fn=None
        )

        model_keys = ['trans']
        self._model_keys = tuple(model_keys)
        self._total_steps = 0

    def get_reward(self, batch):
        return self._get_reward_step(self._train_states, batch)

    @partial(jax.jit, static_argnames=('self'))
    def _get_reward_step(self, train_states, batch):
        obs = batch['observations']
        act = batch['actions']
        timestep = batch['timestep']
        attn_mask = batch['attn_mask']

        train_params = {key: train_states[key].params for key in self.model_keys}
        trans_pred, attn_weights = self.trans.apply(train_params['trans'], obs, act, timestep, attn_mask=attn_mask,
                                                    reverse=False)
        return trans_pred["weighted_sum"], attn_weights[-1]

    def train(self, batch):
        self._total_steps += 1
        self._train_states, metrics = self._train_pref_step(
            self._train_states, next_rng(), batch
        )
        return metrics

    @partial(jax.jit, static_argnames=('self'))
    def _train_pref_step(self, train_states, rng, batch):

        def loss_fn_arbitrary(train_params, rng):
            obs = batch['observations']
            next_obs = batch['next_observations']
            act = batch['actions']
            real_reward = batch['real_rewards']
            bag_label = batch['bag_label']
            bag_end = batch['bag_end']
            timestep = batch['timestep']

            B, T, _ = batch['observations'].shape
            B, T, _ = batch['actions'].shape

            rng, _ = jax.random.split(rng)

            trans_pred, _ = self.trans.apply(train_params['trans'], obs, act, timestep, training=True,
                                             attn_mask=None, rngs={"dropout": rng})

            if self.config.use_weighted_sum:
                trans_pred_reward = trans_pred["weighted_sum"]
            else:
                trans_pred_reward = trans_pred["value"]
            bagged_reward_pred = trans_pred_reward

            state_pred = trans_pred["state_pred"]
            loss_collection = {}
            rng, split_rng = jax.random.split(rng)

            """ reward function loss """
            bagged_reward_pred = jnp.squeeze(bagged_reward_pred, axis=-1)
            bagged_reward_real = jax.lax.stop_gradient(real_reward)
            bag_end = jax.lax.stop_gradient(bag_end)
            trans_loss = bagged_reward_arbitrary_loss(bagged_reward_pred, bagged_reward_real, bag_end, bag_label)
            cse_loss = trans_loss

            """ state loss """
            state_pred = jnp.squeeze(state_pred)
            state_target = jax.lax.stop_gradient(next_obs)
            state_loss = state_loss_func(state_pred, state_target)

            cse_loss = cse_loss + state_loss

            loss_collection['trans'] = trans_loss + state_loss

            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn_arbitrary, len(self.model_keys), has_aux=True)(train_params, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            cse_loss=aux_values['cse_loss'],
            trans_loss=aux_values['trans_loss'],
        )

        return new_train_states, metrics


    @partial(jax.jit, static_argnames=('self'))
    def _train_regression_step(self, train_states, rng, batch):

        def loss_fn(train_params, rng):
            observations = batch['observations']
            next_observations = batch['next_observations']
            actions = batch['actions']
            rewards = batch['rewards']

            in_obs = jnp.concatenate([observations, next_observations], axis=-1)

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)

            """ reward function loss """
            rf_pred = self.rf.apply(train_params['rf'], observations, actions)
            reward_target = jax.lax.stop_gradient(rewards)
            rf_loss = mse_loss(rf_pred, reward_target)

            loss_collection['rf'] = rf_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            rf_loss=aux_values['rf_loss'],
            average_rf=aux_values['rf_pred'].mean(),
        )

        return new_train_states, metrics


    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    def total_steps(self):
        return self._total_steps
