import haiku as hk
import jax
import jax.numpy as jnp
import chex


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
        name="az_net",
    ):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.resnet_v2 = resnet_v2
        self.resnet_cls = BlockV2 if resnet_v2 else BlockV1

    def __call__(self, x, is_training, test_local_stats):
        x = x.astype(jnp.float32)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)

        if not self.resnet_v2:
            x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)

        for i in range(self.num_blocks):
            x = self.resnet_cls(self.num_channels, name=f"block_{i}")(x, is_training, test_local_stats)

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

        return main_policy_logits, exploration_policy_logits, v, u


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


# https://stackoverflow.com/a/77213071
class LCGHash(hk.Module):
    def __init__(
        self,
        bits_per_hash: int = 24,
        multiplier: int = 6_364_136_223_846_793_005,
        increment: int = 1,
        name="lcg-hash",
    ):
        super().__init__(name=name)
        self.bits_per_hash = bits_per_hash
        self.binary_set = jnp.zeros(2 ** (bits_per_hash - 3), dtype="uint8")
        self.multiplier = multiplier
        self.increment = increment

    def __call__(self, x, is_training, test_local_stats) -> jax.Array:
        # x: [batch_size, ...]
        x = jnp.asarray(x, dtype="uint64")
        while len(x.shape) > 1:
            accumulator = jnp.zeros(x.shape[:-1])
            for section in jnp.split(x, x.shape[-1], axis=-1):
                accumulator *= self.multiplier
                accumulator += self.increment
                accumulator += section
            x = accumulator
        # x: [batch_size]

        indices = x >> (63 - self.bits_per_hash)

        # Get the bit corresponding to the index.
        byte_indices = indices // 8
        bit_indices = indices % 8
        # bytes_from_set: [batch_size]
        bytes_from_set = self.binary_set[byte_indices]
        # seen: [batch_size]
        seen = (bytes_from_set & (1 << bit_indices)) > 0

        return seen


class SimHash(hk.Module):
    def __init__(self, bits_per_hash: int = 24, name="sim-hash"):
        super().__init__(name=name)
        self.bits_per_hash = bits_per_hash
        self.binary_set = jnp.zeros(2 ** (bits_per_hash - 3), dtype="uint8")

    def __call__(self, x, is_training, test_local_stats) -> jax.Array:
        # x: [batch_size, vector_size]
        x = jnp.reshape(x, (x.shape[0], -1))
        vector_size = x.shape[-1]

        # random_matrix: [vector_size, bits]
        random_matrix = hk.get_state(
            "random_matrix", [vector_size, self.bits_per_hash], init=hk.initializers.RandomNormal()
        )
        random_matrix = jax.lax.stop_gradient(random_matrix)

        # product: [batch_size, bits]
        product = jnp.matmul(x, random_matrix)

        # Get the hash index corresponding to the matrix product of input and random matrix.
        # See SimHash paper for details.
        powers_of_two = 2 ** jnp.arange(self.bits_per_hash, dtype=jnp.uint64)
        masked_powers = jnp.where(product < 0.0, powers_of_two, 0)
        # indices: [batch_size]
        indices = jnp.sum(masked_powers, axis=1)

        # Get the bit corresponding to the index.
        byte_indices = indices // 8
        bit_indices = indices % 8
        # bytes_from_set: [batch_size]
        bytes_from_set = self.binary_set[byte_indices]
        # seen: [batch_size]
        seen = (bytes_from_set & (1 << bit_indices)) > 0

        return seen
