from typing import Type, Any

import haiku as hk
import jax
import jax.numpy as jnp

from network.hashes import SimHash
from type_aliases import Array, Observation, NetworkOutput


class BlockV1(hk.Module):
    def __init__(self, num_channels: int, name="BlockV1"):
        super(BlockV1, self).__init__(name=name)
        self.num_channels = num_channels

    def __call__(self, x: Array, is_training: bool, test_local_stats: bool):
        i = x
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        return jax.nn.relu(x + i)


class BlockV2(hk.Module):
    def __init__(self, num_channels: int, name="BlockV2"):
        super(BlockV2, self).__init__(name=name)
        self.num_channels = num_channels

    def __call__(self, x: Array, is_training: bool, test_local_stats: bool):
        i = x
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        return x + i


class EpistemicResidualAZNet(hk.Module):
    """Epistemic Residual AlphaZero NN architecture."""

    def __init__(
        self,
        num_actions,
        num_channels: int = 64,
        num_blocks: int = 5,
        resnet_v2: bool = True,
        hash_class: Type = SimHash,
        max_u: float = 1.0,  # assumes board game
        max_epistemic_variance_reward: float = 1.0,
        discount: float = 0.9997,
        hash_args: dict[str, Any] | None = None,
        name="az_resnet",
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
        discount = min(discount, 0.9997)
        self.local_unc_to_max_value_unc_scale = 1.0 / (1 - discount**2)
        self.max_reward_epistemic_variance = max_epistemic_variance_reward

    def __call__(
        self, x: Observation, is_training: bool, test_local_stats: bool, update_hash: bool = False
    ) -> NetworkOutput:
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
            u = u.clip(min=0, max=self.max_u)

        if update_hash:
            hash_obj.update(x)

        return main_policy_logits, exploration_policy_logits, v, u, scaled_state_novelty
