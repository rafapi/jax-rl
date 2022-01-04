import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR


def epsilon_by_frame(frame_idx: int, epsilon_start: float,
                     epsilon_decay: float, epsilon_end: float):
    return epsilon_end + (epsilon_start - epsilon_end) * jnp.exp(-1. * frame_idx / epsilon_decay)


def plot(frame_idx, rewards, losses):
    # clear_output(True)
    plt.close('all')
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title(f'frame {frame_idx}. reward: {np.mean(rewards[-10:])}')
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()
