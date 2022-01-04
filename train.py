import jax
import haiku as hk

import gym

from memory import ReplayBuffer
from agent import DQN
from utils import epsilon_by_frame, plot
# from IPython.display import clear_output


NUM_EPISODES = 1000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-3
SEED = 1729
HIDDEN_UNITS = 50
REPLAY_CAPACITY = 2000

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500


def main():
    env_id = "CartPole-v0"
    env = gym.make(env_id)

    replay_buffer = ReplayBuffer(REPLAY_CAPACITY, GAMMA)

    losses = []
    all_rewards = []
    episode_reward = 0

    num_actions = env.action_space.n
    agent = DQN(num_actions=num_actions, hidden_units=HIDDEN_UNITS, learning_rate=LEARNING_RATE)
    sample_input = env.reset()
    rng = hk.PRNGSequence(jax.random.PRNGKey(SEED))
    net_params, opt_state = agent.param_init(sample_input, rng)
    target_params = net_params

    state = env.reset()
    print(f"Training agent for {NUM_EPISODES} episodes...")
    for idx in range(1, NUM_EPISODES+1):
        epsilon = epsilon_by_frame(idx, epsilon_start, epsilon_decay, epsilon_final)

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
