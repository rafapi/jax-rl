import jax
import haiku as hk

import gym
import hydra

from omegaconf import DictConfig

from src.memory import ReplayBuffer
from src.agent import DQN
from src.utils import epsilon_by_frame, plot


@hydra.main(config_path="config", config_name="conf")
def main(config: DictConfig):
    env_id = "CartPole-v0"
    env = gym.make(env_id)

    replay_buffer = ReplayBuffer(config.replay_capacity, config.gamma)

    losses = []
    all_rewards = []
    episode_reward = 0

    num_actions = env.action_space.n
    agent = DQN(num_actions=num_actions, hidden_units=config.hidden_units, learning_rate=config.learning_rate)
    sample_input = env.reset()
    rng = hk.PRNGSequence(jax.random.PRNGKey(config.seed))
    net_params, opt_state = agent.param_init(sample_input, rng)
    target_params = net_params

    state = env.reset()
    print(f"Training agent for {config.num_episodes} episodes...")
    for idx in range(1, config.num_episodes+1):
        epsilon = epsilon_by_frame(idx, config.epsilon_start, config.epsilon_decay, config.epsilon_final)

        action = agent.select_action(net_params, next(rng), state, epsilon)
        next_state, reward, done, _ = env.step(int(action))
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > config.batch_size:
            batch = replay_buffer.sample(config.batch_size)
            net_params, opt_state, loss = agent.update_parameters(net_params, target_params,
                                                                  opt_state, batch)
            losses.append(float(loss))

        if idx % 200 == 0:
            plot(idx, all_rewards, losses)

        if idx % 100 == 0:
            target_params = net_params


if __name__ == '__main__':
    main()
