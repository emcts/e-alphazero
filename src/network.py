from typing import Type, Any

import haiku as hk
import jax
import jax.numpy as jnp

from hashes import LCGHash, SimHash


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
        num_channels: int = 16,  # FIXME: Make 64 again
        num_blocks: int = 2,  # FIXME: Make 5 again
        resnet_v2: bool = True,
        hash_class: Type = SimHash,
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

    def __call__(
        self, x, is_training, test_local_stats, update_hash: bool = False
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        x = x.astype(jnp.float32)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)

        if not self.resnet_v2:
            x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)

        for i in range(self.num_blocks):
            x = self.resnet_class(self.num_channels, name=f"block_{i}")(x, is_training, test_local_stats)  # type: ignore

        if self.resnet_v2:
            x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)

        # policy head
        main_policy_logits = hk.Conv2D(output_channels=2, kernel_shape=1)(x)
        main_policy_logits = hk.BatchNorm(True, True, 0.9)(main_policy_logits, is_training, test_local_stats)
        main_policy_logits = jax.nn.relu(main_policy_logits)
        main_policy_logits = hk.Flatten()(main_policy_logits)
        main_policy_logits = hk.Linear(self.num_actions)(main_policy_logits)

        # exploration policy head
        exploration_policy_logits = hk.Conv2D(output_channels=2, kernel_shape=1)(x)
        exploration_policy_logits = hk.BatchNorm(True, True, 0.9)(
            exploration_policy_logits, is_training, test_local_stats
        )
        exploration_policy_logits = jax.nn.relu(exploration_policy_logits)
        exploration_policy_logits = hk.Flatten()(exploration_policy_logits)
        exploration_policy_logits = hk.Linear(self.num_actions)(exploration_policy_logits)

        # value head
        v = hk.Conv2D(output_channels=1, kernel_shape=1)(x)
        v = hk.BatchNorm(True, True, 0.9)(v, is_training, test_local_stats)
        v = jax.nn.relu(v)
        v = hk.Flatten()(v)
        v = hk.Linear(self.num_channels)(v)
        v = jax.nn.relu(v)
        v = hk.Linear(1)(v)
        v = jnp.tanh(v)
        v = v.reshape((-1,))

        # ube head
        u = hk.Conv2D(output_channels=1, kernel_shape=1)(x)
        u = hk.BatchNorm(True, True, 0.9)(u, is_training, test_local_stats)
        u = jax.nn.relu(u)
        u = hk.Flatten()(u)
        u = hk.Linear(self.num_channels)(u)
        u = jax.nn.relu(u)
        u = hk.Linear(1)(u)
        u = jnp.exp2(u)
        u = u.reshape((-1,))

        # local uncertainty
        hash_obj = self.hash_class(**self.hash_args)
        seen = hash_obj(x)

        if update_hash:
            hash_obj.update(x)

        return main_policy_logits, exploration_policy_logits, v, u, ~seen


class FullyConnectedAZNet(hk.Module):
    """Fully-connected AlphaZero NN architecture."""

    def __init__(
        self,
        num_actions,
        num_hidden_layers: int = 3,
        layer_size: int = 64,
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

    def __call__(
        self, x, is_training, test_local_stats, update_hash: bool = False
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
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

        # local uncertainty
        hash_obj = self.hash_class(**self.hash_args)
        seen = hash_obj(x)

        if update_hash:
            hash_obj.update(x)

        return main_policy_logits, exploration_policy_logits, v, u, ~seen
