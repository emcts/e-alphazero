from typing import Literal
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pgx  # type: ignore
import chex

ENV_ID = "deep_sea"


# https://github.com/google-deepmind/bsuite/blob/main/bsuite/environments/deep_sea.py


@dataclass
class DeepSeaState(pgx.State):
    observation: jax.Array  # The shape depends on the size of the grid.
    current_player: jax.Array = jnp.int32(0)
    rewards: jax.Array = jnp.float32([0.0])
    terminated: jax.Array = jnp.bool(False)
    truncated: jax.Array = jnp.bool(False)
    legal_action_mask: jax.Array = jnp.ones(2, dtype=jnp.bool)

    _step_count: jax.Array = jnp.int32(0)
    _horizontal_position: jax.Array = jnp.int32(0)  # 0 means left-most

    @property
    def env_id(self) -> pgx.EnvId:
        """Environment id (e.g. "go_19x19")"""
        return ENV_ID  # type: ignore


class DeepSea(pgx.Env):
    def __init__(self, size_of_grid: int = 4) -> None:
        super().__init__()
        self.size_of_grid = size_of_grid

    def _init(self, key: chex.PRNGKey) -> DeepSeaState:
        observation = jnp.zeros([self.size_of_grid, self.size_of_grid], dtype=jnp.bool)
        observation[0][0] = True  # Initial location is in the top-left.
        return DeepSeaState(observation=observation)

    def _step(self, state: DeepSeaState, action: jax.Array, key: chex.PRNGKey) -> DeepSeaState:
        """action: 0(left), 1(right)"""
        assert isinstance(state, DeepSeaState)
        # Action determines whether to shift horizontal position left (-1) or right (+1).
        shift = jnp.where(action == 0, -1, 1)
        horizontal_position = jnp.clip(state._horizontal_position + shift, 0, self.size_of_grid - 1)
        # Observation is just False everywhere except the position.
        observation = jnp.zeros_like(state.observation)
        # _step_count determines depth (it is incremented in the parent function).
        observation[:, state._step_count, horizontal_position] = 1
        # Terminate once we reach the bottom.
        terminated = state._step_count >= self.size_of_grid
        # Boolean rewards will be automatically cast to floats when used.
        rewards = terminated & (horizontal_position == self.size_of_grid - 1)
        return state.replace(  # type: ignore
            observation=observation, _horizontal_position=horizontal_position, rewards=rewards, terminated=terminated
        )

    def _observe(self, state: pgx.State, player_id: jax.Array) -> jax.Array:
        assert isinstance(state, DeepSeaState)
        return state.observation

    @property
    def id(self) -> pgx.EnvId:
        """Environment id."""
        return ENV_ID  # type: ignore

    @property
    def version(self) -> str:
        """Environment version. Updated when behavior, parameter, or API is changed.
        Refactoring or speeding up without any expected behavior changes will NOT update the version number.
        """
        return "0.0.1"

    @property
    def num_players(self) -> int:
        """Number of players (e.g., 2 in Tic-tac-toe)"""
        return 1
