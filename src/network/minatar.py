from typing import Type, Any

import haiku as hk
import jax
import jax.numpy as jnp

from hashes import SimHash


class EpistemicMinatarAZNet(hk.Module):
    """Epistemic AlphaZero NN architecture for Minatar. Linear layers hidden sizes are all the same, defaults to 64.
    Body:
        One Conv2D with 16 channels followed by two linear layers.
    Policy heads:
        Two linear layers, Relu activations, no activate on last layer.
    Value heads:
        Two linear layers, Relu activations, tanh activate on value and exp2 activate on ube.
    """

    def __init__(
        self,
        num_actions,
        num_channels: int = 16,
        hidden_layers_size: int = 64,
        max_u: float = 1.0,
        max_epistemic_variance_reward: float = 1.0,
        discount: float = 0.9997,
        hash_class: Type = SimHash,
        hash_args: dict[str, Any] | None = None,
        name="minatar_az_net",
    ):
        """
        num_actions = env action space size
        num_channels = num_channels for the conv2d layer
        hidden_layers_size = num of units in each hidden layer
        max_u = if passed, clips the UBE prediction <= max_u. In board games for example, max_u = 1
        max_epistemic_variance_reward = used to scale the hash to max_reward ** 2 := max V[R]
        discount = the bellman discount, used to scale the reward uncertainty for novel states
        hash_class = SimHash, LCGHash, or XXHash
        """
        super().__init__(name=name)
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.hidden_layers_size = hidden_layers_size
        self.hash_class = hash_class
        self.hash_args = hash_args if hash_args is not None else dict()
        self.max_u = max_u
        self.local_unc_to_max_value_unc_scale = 1.0 / (1 - discount**2)
        self.max_reward_epistemic_variance = max_epistemic_variance_reward

    def __call__(
        self, x, is_training, test_local_stats, update_hash: bool = False
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        x = x.astype(jnp.float32)

        # Exploitation net body
        x1 = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x1 = jax.nn.relu(x1)
        x1 = hk.Flatten()(x1)
        x1 = hk.Linear(self.hidden_layers_size)(x1)
        x1 = jax.nn.relu(x1)
        x1 = hk.Linear(self.hidden_layers_size)(x1)
        x1 = jax.nn.relu(x1)

        # Exploitation policy head
        main_policy_logits = hk.Linear(self.hidden_layers_size)(x1)
        main_policy_logits = jax.nn.relu(main_policy_logits)
        main_policy_logits = hk.Linear(self.num_actions)(main_policy_logits)

        # Value head
        v = hk.Linear(self.hidden_layers_size)(x1)
        v = jax.nn.relu(v)
        v = hk.Linear(1)(v)
        v = v.reshape((-1,))

        # Exploration net body
        x2 = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x2 = jax.nn.relu(x2)
        x2 = hk.Flatten()(x2)
        x2 = hk.Linear(self.hidden_layers_size)(x2)
        x2 = jax.nn.relu(x2)
        x2 = hk.Linear(self.hidden_layers_size)(x2)
        x2 = jax.nn.relu(x2)

        # exploration policy head
        exploration_policy_logits = hk.Linear(self.hidden_layers_size)(x2)
        exploration_policy_logits = jax.nn.relu(exploration_policy_logits)
        exploration_policy_logits = hk.Linear(self.num_actions)(exploration_policy_logits)

        # ube head
        u = hk.Linear(self.hidden_layers_size)(x2)
        u = jax.nn.relu(u)
        u = hk.Linear(1)(u)
        # Note that u is a scalar between 0 and 1, 1 representing max unc. This is done for stability and learning speed
        u = 0.5 * (jnp.tanh(u) + 1)
        u = u.reshape((-1,))

        # local uncertainty
        hash_obj = self.hash_class(**self.hash_args)
        scaled_state_novelty = (~hash_obj(x)) * self.max_reward_epistemic_variance

        if not is_training:
            # We need to rescale u to the right output scale
            u = u * self.max_u
            # The UBE prediction for AZ is max(attainable sum of reward_unc speculated from local reward_unc, ube)
            u = jnp.maximum(scaled_state_novelty * self.local_unc_to_max_value_unc_scale, u)
            u = u.clip(min=0, max=self.max_u)

        if update_hash:
            hash_obj.update(x)

        return main_policy_logits, exploration_policy_logits, v, u, scaled_state_novelty
