from abc import ABC, abstractmethod
import haiku as hk
import jax
import jax.numpy as jnp


# We can't actually use this for inheritance, but we can still reuse the functions.
class BaseHash(ABC):
    def __init__(self, bits_per_hash: int):
        self.bits_per_hash = bits_per_hash

    @abstractmethod
    def get_indices(self, x) -> tuple[jax.Array, jax.Array]: ...

    def __call__(self, x) -> jax.Array:
        binary_set = self.get_binary_set()
        byte_indices, bit_indices = self.get_indices(x)
        # bytes_from_set: [batch_size]
        bytes_from_set = binary_set[byte_indices]
        # seen: [batch_size]
        seen = (bytes_from_set & (1 << bit_indices)) > 0

        return seen

    def get_binary_set(self) -> jax.Array:
        return hk.get_state(
            "binary_set", [2 ** (self.bits_per_hash - 3)], dtype=jnp.uint8, init=hk.initializers.Constant(0)
        )

    def update(self, x) -> None:
        binary_set = self.get_binary_set()
        byte_indices, bit_indices = self.get_indices(x)
        new_bytes = binary_set[byte_indices] | (1 << bit_indices)
        new_bytes = jnp.asarray(new_bytes, dtype=jnp.uint8)
        binary_set = binary_set.at[byte_indices].set(new_bytes)
        hk.set_state("binary_set", binary_set)


# https://stackoverflow.com/a/77213071
class LCGHash(hk.Module):
    def __init__(
        self,
        bits_per_hash: int = 24,
        name="lcg_hash",
    ):
        super().__init__(name=name)
        assert 0 < bits_per_hash <= 32
        self.bits_per_hash = bits_per_hash

    __call__ = BaseHash.__call__
    get_binary_set = BaseHash.get_binary_set
    update = BaseHash.update

    def get_indices(self, x) -> tuple[jax.Array, jax.Array]:
        # TODO: Maybe try larger constants later
        # REMINDER: Set environment variable JAX_ENABLE_X64=True
        # FIXME: scatter inputs have incompatible types: cannot safely cast value from dtype=int64 to dtype=int32
        MULTIPLIER: int = 29943829
        INCREMENT: int = 1
        TOP_BIT: int = 32
        MODULUS: int = 1 << TOP_BIT

        # x: [batch_size, ...]
        x = jnp.asarray(x, dtype=jnp.uint64)
        while len(x.shape) > 1:
            accumulator = jnp.zeros(x.shape[:-1], dtype=jnp.uint64)
            for section in jnp.split(x, x.shape[-1], axis=-1):
                accumulator *= MULTIPLIER
                accumulator += INCREMENT
                accumulator += section.squeeze()
                accumulator %= MODULUS
            x = accumulator
        # x: [batch_size]

        indices = x >> (TOP_BIT - self.bits_per_hash)
        byte_indices = indices // 8
        bit_indices = indices % 8

        return byte_indices, bit_indices


class SimHash(hk.Module):
    def __init__(self, bits_per_hash: int = 24, name="sim_hash"):
        super().__init__(name=name)
        assert 0 < bits_per_hash <= 32
        self.bits_per_hash = bits_per_hash

    __call__ = BaseHash.__call__
    get_binary_set = BaseHash.get_binary_set
    update = BaseHash.update

    def get_indices(self, x) -> tuple[jax.Array, jax.Array]:
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
        powers_of_two = 2 ** jnp.arange(self.bits_per_hash, dtype=jnp.uint32)
        masked_powers = jnp.where(product < 0.0, powers_of_two, 0)
        # indices: [batch_size]
        indices = jnp.sum(masked_powers, axis=1)

        byte_indices = indices // 8
        bit_indices = indices % 8

        return byte_indices, bit_indices
