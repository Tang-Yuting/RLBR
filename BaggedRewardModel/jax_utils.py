from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import scan
import optax

from jax import grad, jit, vmap, lax
from flax import linen as nn

class JaxRNG(object):
    def __init__(self, seed):
        self.rng = jax.random.PRNGKey(seed)

    def __call__(self):
        self.rng, next_rng = jax.random.split(self.rng)
        return next_rng


def init_rng(seed):
    global jax_utils_rng
    jax_utils_rng = JaxRNG(seed)


def next_rng():
    global jax_utils_rng
    return jax_utils_rng()


def extend_and_repeat(tensor, axis, repeat):
    return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)


def mse_loss(val, target):
    val = jnp.squeeze(val)
    target = jnp.squeeze(target)
    return jnp.mean(jnp.square(val - target))

def cross_ent_loss(logits, target):
    
    if len(target.shape) == 1:
        label = jax.nn.one_hot(target, num_classes=2)
    else:
        label = target
        
    loss = jnp.mean(optax.softmax_cross_entropy(
        logits=logits, 
        labels=label))
    return loss


@partial(jit)
def bagged_reward_loss(bagged_reward_pred, bagged_reward_average, bagged_reward_target):
    loss_func = mse_loss

    # Create masks of non-zero elements and row ends
    non_zero_mask = (bagged_reward_target != 0)
    end_mask = jnp.full_like(bagged_reward_target, False, dtype=jnp.bool_)
    end_mask = end_mask.at[:, -1].set(True)

    # Combine masks into one
    reset_mask = non_zero_mask | end_mask

    # Define scan function
    def scan_func(carry, inputs):
        reset, ave, pred = inputs
        running_sum_ave, running_sum_predict = carry
        running_sum_ave += ave
        running_sum_predict += pred
        out_ave = jnp.where(reset, running_sum_ave, 0.0)
        out_predict = jnp.where(reset, running_sum_predict, 0.0)
        next_carry_ave = jnp.where(reset, 0.0, running_sum_ave)
        next_carry_predict = jnp.where(reset, 0.0, running_sum_predict)
        return (next_carry_ave, next_carry_predict), (out_ave, out_predict)

    # Calculate cumulative sums with reset points
    reward_ave_1 = jnp.zeros_like(bagged_reward_average)
    reward_predict_1 = jnp.zeros_like(bagged_reward_pred)
    for i in range(bagged_reward_average.shape[0]):  # for each row
        (_, _), (reward_ave_1_row, reward_predict_1_row) = lax.scan(scan_func, (0.0, 0.0), (
        reset_mask[i], bagged_reward_average[i], bagged_reward_pred[i]))
        reward_ave_1 = reward_ave_1.at[i, :].set(reward_ave_1_row)
        reward_predict_1 = reward_predict_1.at[i, :].set(reward_predict_1_row)

    # Calculate loss
    loss = loss_func(reward_predict_1, reward_ave_1)

    return loss


@partial(jit)
def bagged_reward_arbitrary_loss(bagged_reward_pred, bagged_reward_real, bag_end, bag_label):
    # bag_end: 0 or 1.
    # bag_label: 0, 1, or 2.
    loss_func = mse_loss

    def scan_func(carry, inputs):
        label, end, ave, pred = inputs
        running_sum_ave_1, running_sum_predict_1, running_sum_ave_2, running_sum_predict_2 = carry
        running_sum_ave_1 += jnp.where(label==1, ave, 0.0)
        running_sum_predict_1 += jnp.where(label==1, pred, 0.0)
        running_sum_ave_1 += jnp.where(label == 2, ave, 0.0)
        running_sum_predict_1 += jnp.where(label == 2, pred, 0.0)
        running_sum_ave_2 += jnp.where(label == 2, ave, 0.0)
        running_sum_predict_2 += jnp.where(label == 2, pred, 0.0)

        out_ave = jnp.where(end, running_sum_ave_1, 0.0)
        out_predict = jnp.where(end, running_sum_predict_1, 0.0)

        next_running_sum_ave_1 = jnp.where(end, running_sum_ave_2, running_sum_ave_1)
        next_running_sum_predict_1 = jnp.where(end, running_sum_predict_2, running_sum_predict_1)
        next_running_sum_ave_2 = jnp.where(end, 0.0, running_sum_ave_2)
        next_running_sum_predict_2 = jnp.where(end, 0.0, running_sum_predict_2)
        return (next_running_sum_ave_1, next_running_sum_predict_1, next_running_sum_ave_2, next_running_sum_predict_2), (out_ave, out_predict)

    # Calculate cumulative sums with reset points
    reward_ave_1 = jnp.zeros_like(bagged_reward_real)
    reward_predict_1 = jnp.zeros_like(bagged_reward_pred)
    for i in range(bagged_reward_real.shape[0]):
        final_carry, (reward_ave_1_row, reward_predict_1_row) = lax.scan(scan_func, (0.0, 0.0, 0.0, 0.0), (
        bag_label[i], bag_end[i], bagged_reward_real[i], bagged_reward_pred[i]))
        reward_ave_1 = reward_ave_1.at[i, :].set(reward_ave_1_row)
        reward_predict_1 = reward_predict_1.at[i, :].set(reward_predict_1_row)

    # Calculate loss
    loss = loss_func(reward_predict_1, reward_ave_1)

    return loss


@partial(jit)
def state_loss_func(state_pred, state_target):
    loss_func = mse_loss
    loss = loss_func(state_pred, state_target)
    return loss


def kld_loss(p, q):
    return jnp.mean(jnp.sum(jnp.where(p != 0, p * (jnp.log(p) - jnp.log(q)), 0), axis=-1))

def custom_softmax(array, axis=-1, temperature=1.0):
    array = array / temperature
    return jax.nn.softmax(array, axis=axis)


def pref_accuracy(logits, target):
    predicted_class = jnp.argmax(logits, axis=1)
    target_class = jnp.argmax(target, axis=1)
    return jnp.mean(predicted_class == target_class)

def value_and_multi_grad(fun, n_outputs, argnums=0, has_aux=False):
    def select_output(index):
        def wrapped(*args, **kwargs):
            if has_aux:
                x, *aux = fun(*args, **kwargs)
                return (x[index], *aux)
            else:
                x = fun(*args, **kwargs)
                return x[index]
        return wrapped

    grad_fns = tuple(
        jax.value_and_grad(select_output(i), argnums=argnums, has_aux=has_aux)
        for i in range(n_outputs)
    )
    def multi_grad_fn(*args, **kwargs):
        grads = []
        values = []
        for grad_fn in grad_fns:
            (value, *aux), grad = grad_fn(*args, **kwargs)
            values.append(value)
            grads.append(grad)
        return (tuple(values), *aux), tuple(grads)
    return multi_grad_fn


@jax.jit
def batch_to_jax(batch):
    return jax.tree_util.tree_map(jax.device_put, batch)
