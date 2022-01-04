import random
import jax.numpy as jnp

from collections import deque


class ReplayBuffer(deque):
    def __init__(self, capacity, gamma):
        super().__init__(maxlen=capacity)
        self.gamma = gamma

    def push(self, state, action, reward, next_state, done):
        state = jnp.expand_dims(state, 0)
        next_state = jnp.expand_dims(next_state, 0)

        self.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self, batch_size))
        return (jnp.concatenate(state), jnp.concatenate(next_state),
                jnp.asarray(action), jnp.asarray(reward),
                (1.-jnp.asarray(done, dtype=jnp.float32))*self.gamma)
