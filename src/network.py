import haiku as hk
import jax
import jax.numpy as jnp


class FullyConnectedAZNet(hk.Module):
    """Fully-connected AlphaZero NN architecture."""

    def __init__(
        self,
        num_actions,
        num_hidden_layers: int = 3,
        layer_size: int = 64,
        name="fc_az_net",
    ):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = layer_size

    def __call__(self, x, is_training, test_local_stats):
        # body
        x = x.astype(jnp.float32)
        for i in range(self.num_hidden_layers):
            x = hk.Linear(self.hidden_layer_size)(x)
            x = jax.nn.relu(x)

        # value head
        v = hk.Flatten()(x)
        v = hk.Linear(self.hidden_layer_size)(v)
        v = jax.nn.relu(v)
        v = hk.Linear(1)(v)
        v = jnp.tanh(v)
        v = v.reshape((-1,))

        # ube head
        u = hk.Flatten()(x)
        u = hk.Linear(self.hidden_layer_size)(u)
        u = jax.nn.relu(u)
        u = hk.Linear(1)(u)
        u = jnp.exp2(u)
        u = u.reshape((-1,))

        # exploitation_policy head
        main_policy_logits = hk.Flatten()(x)
        main_policy_logits = hk.Linear(self.hidden_layer_size)(main_policy_logits)
        main_policy_logits = jax.nn.relu(main_policy_logits)
        main_policy_logits = hk.Linear(self.num_actions)(main_policy_logits)

        # exploration_policy head
        exploration_policy_logits = hk.Flatten()(x)
        exploration_policy_logits = hk.Linear(self.hidden_layer_size)(exploration_policy_logits)
        exploration_policy_logits = jax.nn.relu(exploration_policy_logits)
        exploration_policy_logits = hk.Linear(self.num_actions)(exploration_policy_logits)

        return main_policy_logits, exploration_policy_logits, v, u
