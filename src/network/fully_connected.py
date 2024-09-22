from typing import Type, Any

import haiku as hk
import jax
import jax.numpy as jnp

from network.hashes import SimHash
from type_aliases import Observation, NetworkOutput


class EpistemicFullyConnectedAZNet(hk.Module):
    """Fully-connected AlphaZero NN architecture."""

    def __init__(
        self,
        num_actions,
        num_hidden_layers: int = 3,
        layer_size: int = 64,
        max_ube: float = 1.0,
        max_epistemic_variance_reward: float = 1.0,
        discount: float = 0.9997,
        hash_class: Type = SimHash,
        hash_args: dict[str, Any] | None = None,
        name="fc_az_net",
        word_size: int = 16,
        hash_io: bool = False
    ):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = layer_size
        self.hash_class = hash_class
        self.hash_args = hash_args if hash_args is not None else dict()
        self.max_u = max_ube
        discount = min(discount, 0.9997)
        self.local_unc_to_max_value_unc_scale = 1.0 / (1 - discount**2)
        self.max_reward_epistemic_variance = max_epistemic_variance_reward
        self.word_size = word_size
        self.hash_io = hash_io

    def __call__(
        self, x: Observation, is_training: bool, test_local_stats: bool, update_hash: bool = False
    ) -> NetworkOutput:
        # body
        x = x.astype(jnp.float32)
        unflattened_x = x
        x = hk.Flatten()(x)

        # value head
        v = hk.Linear(256)(x)
        v = jax.nn.relu(v)
        v = hk.Linear(256)(v)
        v = jax.nn.relu(v)
        v = hk.Linear(1)(v)
        v = jnp.tanh(v)
        v = v.reshape((-1,))

        # ube head
        u = hk.Linear(256)(x)
        u = jax.nn.relu(u)
        u = hk.Linear(256)(u)
        u = jax.nn.relu(u)
        u = hk.Linear(1)(u)
        u = 0.5 * (jnp.tanh(u) + 1)
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

        # local uncertainty
        hash_obj = self.hash_class(**self.hash_args)
        if self.hash_io:
            hash_input = unflattened_x[:, self.word_size:, :]
            hash_input = hk.Flatten()(hash_input)
        else:
            hash_input = x
        scaled_state_novelty = ~hash_obj(hash_input) * self.max_reward_epistemic_variance

        if not is_training:
            u = u * self.max_u
            # The UBE prediction for AZ is max(attainable sum of reward_unc speculated from local reward_unc, ube)
            u = jnp.maximum(scaled_state_novelty, u)
            u = u.clip(min=0, max=self.max_u)

        if update_hash:
            hash_obj.update(hash_input)

        return main_policy_logits, exploration_policy_logits, v, u, scaled_state_novelty
