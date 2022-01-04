import jax
import optax
import rlax

from jax import vmap

from network import build_network


class DQN:
    """A simple DQN agent."""

    def __init__(self, num_actions, hidden_units, learning_rate):
        self._optimizer = optax.adam(learning_rate)
        self._network = build_network(num_actions, hidden_units)
        self.select_action = jax.jit(self.select_action)
        self.update_parameters = jax.jit(self.update_parameters)

    def param_init(self, sample_input, rng):
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
        obs_tm1, obs_t, a_tm1, r_t, discount_t = batch
        q_tm1 = self._network.apply(net_params, obs_tm1)
        q_t_value = self._network.apply(target_params, obs_t)
        q_t_selector = self._network.apply(net_params, obs_t)

        batched_loss = vmap(rlax.double_q_learning)
        td_error = batched_loss(q_tm1, a_tm1, r_t, discount_t,
                                q_t_value, q_t_selector)
        return jax.numpy.mean(rlax.l2_loss(td_error))
