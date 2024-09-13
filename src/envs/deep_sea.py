import chex
import jax
import jax.numpy as jnp
import pgx  # type: ignore
from pgx._src.struct import dataclass  # type: ignore

from type_aliases import Array, PRNGKey

ENV_ID = "deep_sea"


@dataclass
class DeepSeaState(pgx.State):
    observation: Array = jnp.zeros(0, dtype=jnp.bool)  # The shape depends on the size of the grid.
    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.float32([0.0])
    terminated: Array = jnp.bool(False)
    truncated: Array = jnp.bool(False)
    legal_action_mask: Array = jnp.ones(2, dtype=jnp.bool)

    _step_count: Array = jnp.int32(0)  # how many steps have been taken = depth of submarine
    _horizontal_position: Array = jnp.int32(0)  # 0 means left-most

    @property
    def env_id(self) -> pgx.EnvId:
        """Environment id (e.g. "go_19x19")"""
        return ENV_ID  # type: ignore


class DeepSea(pgx.Env):
    """
    Deep Sea environment.

    Reference definition: https://github.com/google-deepmind/bsuite/blob/main/bsuite/environments/deep_sea.py#L18-L34
    """

    def __init__(self, size_of_grid: int = 4, action_map_key: PRNGKey | None = None) -> None:
        """
        size_of_grid:
            Deep Sea grid will be `size_of_grid X size_of_grid`
        action_map_key:
            When `None`, every state will have the same action mapping,
            i.e. `0` always equals left, and `1` always equals right.
            If a key is given, every state gets assigned a random mapping
            which is consistent across episodes for a particular
            environment instance.
        """
        super().__init__()
        self.size_of_grid = size_of_grid
        self.action_map = jnp.zeros([self.size_of_grid, self.size_of_grid], dtype=jnp.bool)
        if action_map_key is not None:
            self.action_map = jax.random.bernoulli(action_map_key, shape=self.action_map.shape) > 0

    def _init(self, key: PRNGKey) -> DeepSeaState:
        observation = jnp.zeros([self.size_of_grid, self.size_of_grid], dtype=jnp.bool)
        observation = observation.at[..., 0, 0].set(True)  # Initial location is in the top-left.
        return DeepSeaState(observation=observation)

    def _step(self, state: DeepSeaState, action: Array, key: PRNGKey) -> DeepSeaState:
        assert isinstance(state, DeepSeaState)
        # Action XOR action_flip determines whether to shift horizontal position left (-1) or right (+1).
        action_flip = self.action_map[..., state._step_count - 1, state._horizontal_position]
        shift = jnp.where((action == 0) ^ action_flip, -1, 1)
        horizontal_position = jnp.clip(state._horizontal_position + shift, 0, self.size_of_grid - 1)
        # Observation is just False everywhere except the position.
        observation = jnp.zeros_like(state.observation)
        # _step_count determines depth (it is incremented in the parent function).
        observation = observation.at[
            ..., jnp.minimum(state._step_count, self.size_of_grid - 1), horizontal_position
        ].set(True)
        # Terminate once we reach the bottom.
        terminated = state._step_count >= self.size_of_grid - 1
        # Boolean rewards will be automatically cast to floats when used.
        rewards = (
            (terminated & (horizontal_position == self.size_of_grid - 1))
            .reshape(state.rewards.shape)
            .astype(state.rewards.dtype)
        )
        return state.replace(  # type: ignore
            observation=observation, _horizontal_position=horizontal_position, rewards=rewards, terminated=terminated
        )

    def _observe(self, state: pgx.State, player_id: Array) -> Array:
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
