import jax
from jax import vmap
import jax.numpy as jnp
import optax
import haiku as hk
import rlax

import gym
import numpy as np

import random

from haiku import nets
# from IPython.display import clear_output
from collections import deque
from typing import Callable, Mapping, NamedTuple, Tuple, Sequence

import matplotlib.pyplot as plt


COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR

NUM_EPISODES = 10000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-3
SEED = 1729
HIDDEN_UNITS = 50
REPLAY_CAPACITY = 2000

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500


def epsilon_by_frame(frame_idx):
    return epsilon_final + (epsilon_start - epsilon_final) * jnp.exp(-1. * frame_idx / epsilon_decay)


def build_network(num_actions: int) -> hk.Transformed:
    """Factory for a simple MLP network for approximating Q-values."""

    def q(obs):
        network = hk.Sequential(
            [hk.Flatten(),
             nets.MLP([HIDDEN_UNITS, num_actions])])
        return network(obs)

    return hk.without_apply_rng(hk.transform(q))


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = jnp.expand_dims(state, 0)
        next_state = jnp.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (jnp.concatenate(state), jnp.concatenate(next_state), 
                jnp.asarray(action), jnp.asarray(reward), 
                (1.-jnp.asarray(done, dtype=jnp.float32))*GAMMA)

    def __len__(self):
        return len(self.buffer)


def plot(frame_idx, rewards, losses):
    # clear_output(True)
    plt.close('all')
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title(f'frame {frame_idx}. reward: {np.mean(rewards[-10:])}')
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


class DQN:
    """A simple DQN agent."""

    def __init__(self, num_actions):
        self._optimizer = optax.adam(LEARNING_RATE)
        self._network = build_network(num_actions)
        self.select_action = jax.jit(self.select_action)
        self.update_parameters = jax.jit(self.update_parameters)

    def initial_params(self, sample_input, rng):
        net_params = self._network.init(next(rng), sample_input)
        opt_state = self._optimizer.init(net_params)
        return net_params, opt_state

    def select_action(self, net_params, key, obs, epsilon):
        """Sample action from epsilon-greedy policy."""
        q = self._network.apply(net_params, obs)
        action = rlax.epsilon_greedy(epsilon).sample(key, q)
        return action

    def update_parameters(self, net_params, target_params, opt_state, batch):
        """Update network weights wrt Q-learning loss."""

        loss, dloss_dtheta = jax.value_and_grad(self._loss)(net_params, 
                                                            target_params, batch)
        updates, opt_state = self._optimizer.update(dloss_dtheta, opt_state)
        net_params = optax.apply_updates(net_params, updates)
        return net_params, opt_state, loss

    def _loss(self, net_params, target_params, batch):
        obs_tm1, obs_t, a_tm1, r_t, discount_t  = batch
        q_tm1 = self._network.apply(net_params, obs_tm1)
        q_t_value = self._network.apply(target_params, obs_t)
        q_t_selector = self._network.apply(net_params, obs_t)

        batched_loss = vmap(rlax.double_q_learning)
        td_error = batched_loss(q_tm1, a_tm1, r_t, discount_t,
                                q_t_value, q_t_selector)
        return jnp.mean(rlax.l2_loss(td_error))


def main():
    env_id = "CartPole-v0"
    env = gym.make(env_id)

    replay_buffer = ReplayBuffer(REPLAY_CAPACITY)

    losses = []
    all_rewards = []
    episode_reward = 0

    num_actions = env.action_space.n
    agent = DQN(num_actions=num_actions)
    sample_input = env.reset()
    rng = hk.PRNGSequence(jax.random.PRNGKey(SEED))
    net_params, opt_state = agent.initial_params(sample_input, rng)
    target_params = net_params

    state = env.reset()
    print(f"Training agent for {NUM_EPISODES} episodes...")
    for idx in range(1, NUM_EPISODES+1):
        epsilon = epsilon_by_frame(idx)

        action = agent.select_action(net_params, next(rng), state, epsilon)
        next_state, reward, done, _ = env.step(int(action))
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > BATCH_SIZE:
            batch = replay_buffer.sample(BATCH_SIZE)
            net_params, opt_state, loss = agent.update_parameters(net_params, target_params, 
                                                                  opt_state, batch)
            losses.append(float(loss))

        if idx % 200 == 0:
            plot(idx, all_rewards, losses)

        if idx % 100 == 0:
            target_params = net_params


if __name__ == '__main__':
    main()
