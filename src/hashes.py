from abc import ABC, abstractmethod
import haiku as hk
import jax
import jax.numpy as jnp


# We can't actually use this for inheritance, but we can still reuse the functions.
class BaseHash(ABC):
    def __init__(self, bits_per_hash: int):
        self.bits_per_hash = bits_per_hash

    @abstractmethod
    def get_indices(self, x) -> jax.Array: ...

    def get_byte_and_bit_indices(self, x) -> tuple[jax.Array, jax.Array]:
        indices = self.get_indices(x)
        byte_indices = (indices // 8).astype(jnp.uint32)
        bit_indices = (indices % 8).astype(jnp.uint8)
        return byte_indices, bit_indices

    def __call__(self, x) -> jax.Array:
        binary_set = self.get_binary_set()
        byte_indices, bit_indices = self.get_byte_and_bit_indices(x)
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
        byte_indices, bit_indices = self.get_byte_and_bit_indices(x)
        new_bytes = binary_set[byte_indices] | (1 << bit_indices)
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
    get_byte_and_bit_indices = BaseHash.get_byte_and_bit_indices
    get_binary_set = BaseHash.get_binary_set
    update = BaseHash.update

    def get_indices(self, x) -> jax.Array:
        # TODO: Maybe try larger constants later
        # REMINDER: Set environment variable JAX_ENABLE_X64=True
        # FIXME: Setting the environment variable makes pgx break...
        MULTIPLIER: int = 29943829
        INCREMENT: int = 1
        TOP_BIT: int = 32
        MODULUS: int = 1 << TOP_BIT

        # x: [batch_size, ...]
        x = jax.lax.bitcast_convert_type(jnp.asarray(x, jnp.float64), jnp.uint64)
        while len(x.shape) > 1:
            accumulator = jnp.zeros(x.shape[:-1], dtype=jnp.uint64)
            for section in jnp.split(x, x.shape[-1], axis=-1):
                accumulator *= MULTIPLIER
                accumulator += INCREMENT
                accumulator += section.squeeze()
                accumulator %= MODULUS
            x = accumulator
        # x: [batch_size]

        return x >> (TOP_BIT - self.bits_per_hash)


class SimHash(hk.Module):
    def __init__(self, bits_per_hash: int = 24, name="sim_hash"):
        super().__init__(name=name)
        assert 0 < bits_per_hash <= 32
        self.bits_per_hash = bits_per_hash

    __call__ = BaseHash.__call__
    get_byte_and_bit_indices = BaseHash.get_byte_and_bit_indices
    get_binary_set = BaseHash.get_binary_set
    update = BaseHash.update

    def get_indices(self, x) -> jax.Array:
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
        return jnp.sum(masked_powers, axis=1)


# https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md#xxh32-algorithm-description
# https://create.stephan-brumme.com/xxhash/
# https://github.com/google/jax/discussions/10475#discussioncomment-2656590
class XXHash(hk.Module):
    def __init__(self, bits_per_hash: int = 24, name="xxhash32"):
        super().__init__(name=name)
        assert 0 < bits_per_hash <= 32
        self.bits_per_hash = bits_per_hash

    __call__ = BaseHash.__call__
    get_byte_and_bit_indices = BaseHash.get_byte_and_bit_indices
    get_binary_set = BaseHash.get_binary_set
    update = BaseHash.update

    def get_indices(self, x) -> jax.Array:
        # https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md#overview
        PRIME_1 = jnp.uint32(0x9E3779B1)  # 0b10011110001101110111100110110001
        PRIME_2 = jnp.uint32(0x85EBCA77)  # 0b10000101111010111100101001110111
        PRIME_3 = jnp.uint32(0xC2B2AE3D)  # 0b11000010101100101010111000111101
        PRIME_4 = jnp.uint32(0x27D4EB2F)  # 0b00100111110101001110101100101111
        PRIME_5 = jnp.uint32(0x165667B1)  # 0b00010110010101100110011110110001
        SEED = jnp.uint32(1)  # Optional, can be zero
        BITS = 32

        def rotate_left(x: jax.Array, n: int) -> jax.Array:
            return (x << n) | (x >> (BITS - n))

        # https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md#step-2-process-stripes
        def round(acc: jax.Array, lane: jax.Array) -> jax.Array:
            # acc, lane, return: [4, batch_size]
            acc = acc + (lane * PRIME_2)
            acc = rotate_left(acc, 13)
            acc = acc * PRIME_1
            return acc

        # https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md#step-3-accumulator-convergence
        def convergence(acc: jax.Array) -> jax.Array:
            # acc: [4, batch_size]
            # return: [batch_size]
            return rotate_left(acc[0], 1) + rotate_left(acc[1], 7) + rotate_left(acc[2], 12) + rotate_left(acc[3], 18)

        # https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md#step-6-final-mix-avalanche
        def avalanche(acc: jax.Array) -> jax.Array:
            # acc, return: [batch_size]
            acc = acc ^ (acc >> 15)
            acc = acc * PRIME_2
            acc = acc ^ (acc >> 13)
            acc = acc * PRIME_3
            acc = acc ^ (acc >> 16)
            return acc

        def loop_fn(acc: jax.Array, stripe: jax.Array) -> tuple[jax.Array, None]:
            # acc, stripe: [4, batch_size]
            # return: ([4, batch_size], None)
            acc = round(acc, stripe)
            return acc, None

        # Convert input to 4 lanes of u32 (per batch element).
        batch_size = x.shape[0]
        x = jax.lax.bitcast_convert_type(x, jnp.uint32)
        # Assumption/simplification: data is a multiple of 4.
        # TODO: Pad data so that it is a multiple of 4 or implement it like described in the docs.
        x = jnp.reshape(x, [batch_size, 4, -1])  # x: [batch_size, 4, L]
        x = jnp.swapaxes(x, 0, 2)  # x: [L, 4, batch_size]
        input_length = x.shape[0]  # = L

        # Initialize accumulators
        # https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md#step-1-initialize-internal-accumulators
        acc1 = SEED + PRIME_1 + PRIME_2
        acc2 = SEED + PRIME_2
        acc3 = SEED + 0
        acc4 = SEED - PRIME_1
        acc = jnp.tile(jnp.array([[acc1], [acc2], [acc3], [acc4]]), (1, batch_size))  # acc: [4, batch_size]

        # Do xxhash.
        acc, _ = jax.lax.scan(loop_fn, acc, x)
        acc = convergence(acc)
        acc = acc + input_length
        acc = avalanche(acc)

        return acc >> (BITS - self.bits_per_hash)
