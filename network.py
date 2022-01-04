import haiku as hk


def build_network(num_actions: int, hidden_units: int) -> hk.Transformed:
    """Factory for a simple MLP network for approximating Q-values."""

    def q(obs):
        network = hk.Sequential(
            [hk.Flatten(),
             hk.nets.MLP([hidden_units, num_actions])])
        return network(obs)

    return hk.without_apply_rng(hk.transform(q))
