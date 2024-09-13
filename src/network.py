from typing import Type, Any

import haiku as hk
import jax
import jax.numpy as jnp

from hashes import LCGHash, SimHash, XXHash


class BlockV1(hk.Module):
    def __init__(self, num_channels, name="BlockV1"):
        super(BlockV1, self).__init__(name=name)
        self.num_channels = num_channels

    def __call__(self, x, is_training, test_local_stats):
        i = x
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        return jax.nn.relu(x + i)


class BlockV2(hk.Module):
    def __init__(self, num_channels, name="BlockV2"):
        super(BlockV2, self).__init__(name=name)
        self.num_channels = num_channels

    def __call__(self, x, is_training, test_local_stats):
        i = x
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        return x + i


class EpistemicAZNet(hk.Module):
    """AlphaZero NN architecture."""

    def __init__(
        self,
        num_actions,
        num_channels: int = 64,
        num_blocks: int = 5,
        resnet_v2: bool = True,
        hash_class: Type = SimHash,
        max_u: float = 1.0, # assumes boardgame
        max_epistemic_variance_reward: float = 1.0,
        discount: float = 0.9997,
        hash_args: dict[str, Any] | None = None,
        name="az_net",
    ):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.resnet_v2 = resnet_v2
        self.resnet_class = BlockV2 if resnet_v2 else BlockV1
        self.hash_class = hash_class
        self.hash_args = hash_args if hash_args is not None else dict()
        self.max_u = max_u
        self.local_unc_to_max_value_unc_scale = 1.0 / (1 - discount**2)
        self.max_reward_epistemic_variance = max_epistemic_variance_reward

    def __call__(
        self, x, is_training, test_local_stats, update_hash: bool = False
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        x = x.astype(jnp.float32)
        x1 = hk.Conv2D(self.num_channels, kernel_shape=3)(x)

        if not self.resnet_v2:
            x1 = hk.BatchNorm(True, True, 0.9)(x1, is_training, test_local_stats)
            x1 = jax.nn.relu(x1)

        for i in range(self.num_blocks):
            x1 = self.resnet_class(self.num_channels, name=f"block_{i}")(x1, is_training, test_local_stats)  # type: ignore

        if self.resnet_v2:
            x1 = hk.BatchNorm(True, True, 0.9)(x1, is_training, test_local_stats)
            x1 = jax.nn.relu(x1)

        # policy head
        main_policy_logits = hk.Conv2D(output_channels=2, kernel_shape=1)(x1)
        main_policy_logits = hk.BatchNorm(True, True, 0.9)(main_policy_logits, is_training, test_local_stats)
        main_policy_logits = jax.nn.relu(main_policy_logits)
        main_policy_logits = hk.Flatten()(main_policy_logits)
        main_policy_logits = hk.Linear(self.num_actions)(main_policy_logits)

        # exploration policy head
        exploration_policy_logits = jax.lax.stop_gradient(x1)
        exploration_policy_logits = hk.Conv2D(output_channels=2, kernel_shape=1)(exploration_policy_logits)
        exploration_policy_logits = hk.BatchNorm(True, True, 0.9)(
            exploration_policy_logits, is_training, test_local_stats
        )
        exploration_policy_logits = jax.nn.relu(exploration_policy_logits)
        exploration_policy_logits = hk.Flatten()(exploration_policy_logits)
        exploration_policy_logits = hk.Linear(self.num_actions)(exploration_policy_logits)

        # value head
        v = hk.Conv2D(output_channels=1, kernel_shape=1)(x1)
        v = hk.BatchNorm(True, True, 0.9)(v, is_training, test_local_stats)
        v = jax.nn.relu(v)
        v = hk.Flatten()(v)
        v = hk.Linear(self.num_channels)(v)
        v = jax.nn.relu(v)
        v = hk.Linear(1)(v)
        v = jnp.tanh(v)
        v = v.reshape((-1,))

        # ube head
        u = jax.lax.stop_gradient(x1)
        u = hk.Conv2D(output_channels=1, kernel_shape=1)(u)
        u = hk.BatchNorm(True, True, 0.9)(u, is_training, test_local_stats)
        u = jax.nn.relu(u)
        u = hk.Flatten()(u)
        u = hk.Linear(self.num_channels)(u)
        u = jax.nn.relu(u)
        u = hk.Linear(1)(u)
        u = 0.5 * (jnp.tanh(u) + 1)
        u = u.reshape((-1,))

        # local uncertainty
        hash_obj = self.hash_class(**self.hash_args)
        scaled_state_novelty = (~hash_obj(x)) * self.max_reward_epistemic_variance

        if not is_training:
            u = u * self.max_u
            # The UBE prediction for AZ is max(attainable sum of reward_unc speculated from local reward_unc, ube)
            u = jnp.maximum(scaled_state_novelty * self.local_unc_to_max_value_unc_scale, u)
            u.clip(min=0, max=self.max_u)

        if update_hash:
            hash_obj.update(x)

        return main_policy_logits, exploration_policy_logits, v, u, scaled_state_novelty


class MinatarEpistemicAZNet(hk.Module):
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


class FullyConnectedAZNet(hk.Module):
    """Fully-connected AlphaZero NN architecture."""

    def __init__(
        self,
        num_actions,
        num_hidden_layers: int = 3,
        layer_size: int = 64,
        max_u: float = 1.0,
        max_epistemic_variance_reward: float = 1.0,
        discount: float = 0.9997,
        hash_class: Type = SimHash,
        hash_args: dict[str, Any] | None = None,
        name="fc_az_net",
    ):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = layer_size
        self.hash_class = hash_class
        self.hash_args = hash_args if hash_args is not None else dict()
        self.max_u = max_u
        self.local_unc_to_max_value_unc_scale = 1.0 / (1 - discount**2)
        self.max_reward_epistemic_variance = max_epistemic_variance_reward

    def __call__(
        self, x, is_training, test_local_stats, update_hash: bool = False
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        # body
        x = x.astype(jnp.float32)
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
        scaled_state_novelty = ~hash_obj(x) * self.max_reward_epistemic_variance

        if not is_training:
            u = u * self.max_u
            # The UBE prediction for AZ is max(attainable sum of reward_unc speculated from local reward_unc, ube)
            u = jnp.maximum(scaled_state_novelty, u)
            u = u.clip(min=0, max=self.max_u)

        if update_hash:
            hash_obj.update(x)

        return main_policy_logits, exploration_policy_logits, v, u, scaled_state_novelty
