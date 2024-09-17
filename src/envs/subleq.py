from enum import IntEnum, auto
from typing import Callable, NamedTuple
import chex
import jax
import jax.numpy as jnp
import pgx  # type: ignore
from pgx._src.struct import dataclass  # type: ignore

from type_aliases import Array, PRNGKey

ENV_ID = "subleq"
MAXIMUM_INPUT_LENGTH = 8
MAXIMUM_OUTPUT_LENGTH = MAXIMUM_INPUT_LENGTH


def subleq_words_to_observation(arr: Array, word_size: int) -> Array:
    """
    Convert from Subleq words to an encoded array (e.g. one-hot).

    Example:
    ```
    word_size = 8, [1, 3, 5, -1]
    ```
    becomes
    ```
    [[0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0]]
    ```
    Note that a negative number `-x` is treated as `size - x`
    since they are the same in `size`-modular arithmetic.

    Numbers which equal `size` are treated special since they represent
    special tokens used to fill up input/output so that the size is always the same.
    """
    chex.assert_rank(arr, 1)
    indices = arr % word_size
    indices = indices.at[arr == word_size].set(word_size)
    shape = (indices.shape[0], word_size + 1)
    # TODO: Check if this should not have `jnp.arange(indices.shape[0])` instead of `:`.
    return jnp.zeros(shape, dtype=jnp.bool).at[:, indices].set(True, unique_indices=True)


def observation_to_subleq_words(arr: Array) -> Array:
    """
    Convert from an encoded array to Subleq words.

    Example:
    ```
    [[0, 1, 0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1]]
    ```
    becomes
    ```
    [1, 0, 5, 8]
    ```
    """
    chex.assert_rank(arr, 2)
    return jnp.argmax(arr, axis=1)


def pad(array: Array, desired_size: int, word_size: int) -> Array:
    """Pad the array out with the special value (= word_size)."""
    chex.assert_rank(array, 1)
    outside = word_size * jnp.ones(desired_size)
    return outside.at[: array.shape[0]].set(array)


class SubleqTask(IntEnum):
    NEGATION = auto()
    IDENTITY = auto()
    SUBTRACTION = auto()
    ADDITION = auto()
    MAXIMUM = auto()
    MINIMUM = auto()
    COMPARISON = auto()
    SORT_2 = auto()
    SORT_3 = auto()
    SORT_4 = auto()
    MULTIPLICATION = auto()
    DIVISION = auto()


class SubleqTestResult(NamedTuple):
    solved: Array

    # The first test case
    example_input: Array
    example_output: Array
    # What the input and output looked like for the example
    input_after: Array
    output_after: Array
    # Maximum bytes and cycles used by any test case
    bytes_used: Array
    cycles_used: Array


class SubleqSimulateResult(NamedTuple):
    input_after: Array
    output_after: Array
    bytes_used: Array
    cycles_used: Array


SubleqRewardFn = Callable[[SubleqTestResult], Array]
SubleqTestFn = Callable[[int, Array], SubleqTestResult]


def simulate(word_size: int, memory_state: Array, test_input: Array, test_output: Array) -> SubleqSimulateResult:
    class InterpreterState(NamedTuple):
        memory_state: Array
        input_state: Array
        output_state: Array
        cursor_position: int

        bytes_used: int
        cycles: int
        should_halt: bool

    def cond_fn(state: InterpreterState) -> bool:
        return state.should_halt

    def body_fn(state: InterpreterState) -> InterpreterState:
        cycles = state.cycles + 1

        # TODO: Implement one step of subleq.
        # - Read an instruction (check for bounds)
        # - Special cases (@IN, @OUT, @HALT)
        # - Read data, compute result, write out
        # - Jump? (compute new cursor_pos)
        # - bytes_used is max of cursor pos+3 and ...#

        return InterpreterState(
            memory_state=memory_state,
            input_state=input_state,
            output_state=output_state,
            cursor_position=cursor_position,
            bytes_used=bytes_used,
            cycles=cycles,
            should_halt=should_halt,
        )

    after_execution = jax.lax.while_loop(
        cond_fn,
        body_fn,
        InterpreterState(
            memory_state=memory_state,
            input_state=test_input,
            output_state=jnp.ones_like(test_output),
            cursor_position=0,
            should_halt=False,
        ),
    )
    return SubleqSimulateResult(
        input_after=after_execution.input_state,
        output_after=after_execution.output_state,
        bytes_used=after_execution.bytes_used,
        cycles_used=after_execution.cycles,
    )


def get_test_fn(task: SubleqTask) -> SubleqTestFn:
    # TODO: Do this for many tests
    test_input: Array
    test_output: Array
    match task:
        case SubleqTask.NEGATION:
            test_input = jnp.arange(1, 5)
            test_output = -test_input
        case _:
            raise NotImplementedError

    def test_fn(word_size: int, memory_state: Array) -> SubleqTestResult:
        chex.assert_rank(memory_state, 1)

        # TODO: jax.vmap over test cases (maybe do example first, then vmap everything else conditionally on success of example?)
        result = simulate(
            word_size,
            memory_state,
            pad(test_input, MAXIMUM_INPUT_LENGTH, word_size),
            pad(test_output, MAXIMUM_INPUT_LENGTH, word_size),
        )

        return SubleqTestResult(
            solved=jnp.zeros(1),  # TODO: should have shape [batch_size]
            example_input=test_input,  # TODO: change to only take the first
            example_output=test_output,  # TODO: change to only take the first
            input_after=result.input_after,  # TODO: change to only take the first
            output_after=result.output_after,  # TODO: change to only take the first
            bytes_used=result.bytes_used,  # TODO: Take the max
            cycles_used=result.cycles_used,  # TODO: Change the max
        )

    return test_fn


def solved_or_not(execution_result: SubleqTestResult) -> Array:
    """Simplest reward function which just gives 1 if the program solves all the tests, and 0 otherwise."""
    return execution_result.solved.astype(jnp.float32)


@dataclass
class SubleqState(pgx.State):
    observation: Array = jnp.zeros(0, dtype=jnp.bool)  # depends on word_size
    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.float32([0.0])
    terminated: Array = jnp.bool(False)
    truncated: Array = jnp.bool(False)
    legal_action_mask: Array = jnp.ones(0, dtype=jnp.bool)  # depends on word_size

    _step_count: Array = jnp.int32(0)
    _task: SubleqTask = SubleqTask.NEGATION
    _test_fn: SubleqTestFn = get_test_fn(SubleqTask.NEGATION)

    _memory_state: Array = jnp.zeros(0, dtype=jnp.bool)  # depends on word_size
    _example_input = jnp.zeros(MAXIMUM_INPUT_LENGTH, dtype=jnp.int32)
    _example_output = jnp.zeros(MAXIMUM_OUTPUT_LENGTH, dtype=jnp.int32)
    _example_input_after = jnp.zeros(MAXIMUM_INPUT_LENGTH, dtype=jnp.int32)
    _example_output_after = jnp.zeros(MAXIMUM_OUTPUT_LENGTH, dtype=jnp.int32)

    @property
    def env_id(self) -> pgx.EnvId:
        """Environment id (e.g. "go_19x19")"""
        return ENV_ID  # type: ignore


class Subleq(pgx.Env):
    """
    Reference description of Subleq: https://github.com/jaredkrinke/sic1/blob/master/sic1-assembly.md

    Every subleq task is a different programming challenge, and each is treated as a separate environment.
    The state the agent observes is the
    """

    def __init__(
        self, tasks: list[SubleqTask], word_size: int = 256, reward_fn: SubleqRewardFn = solved_or_not
    ) -> None:
        assert 8 <= word_size <= 256
        super().__init__()
        self.tasks = jnp.array(tasks)
        self.reward_fn = reward_fn
        self.word_size = word_size
        # Words are one-hot encoded, with an extra column for a special token used to fill out input/output.
        self.token_size = word_size + 1

        # TODO: Remove this
        self.special_address_in = word_size - 3  # Reading from this address accesses the input (writes are ignored).
        self.special_address_out = word_size - 2  # Writing to this address writes to output (reads as zero).
        self.special_address_halt = word_size - 1  # Terminates the program when accessed.

    def _init(self, key: PRNGKey) -> SubleqState:
        task: SubleqTask = jax.random.choice(key, self.tasks).item()

        rows = self.word_size + 2 * (MAXIMUM_INPUT_LENGTH + MAXIMUM_OUTPUT_LENGTH)
        observation = jnp.zeros([rows, self.token_size], dtype=jnp.bool)
        legal_action_mask = jnp.ones(self.word_size, dtype=jnp.bool)

        state = SubleqState(
            legal_action_mask=legal_action_mask,
            _task=task,
            _test_fn=get_test_fn(task),
            _memory_state=jnp.zeros(self.word_size, dtype=jnp.bool),
        )
        result = state._test_fn(self.word_size, state._memory_state)
        reward = self.reward_fn(result)

        return state.replace(observation=self._observe(state, state.current_player), reward=reward)  # type: ignore

    def _step(self, state: SubleqState, action: Array, key: PRNGKey) -> SubleqState:
        assert isinstance(state, SubleqState)

        # TODO: Add a byte to the memory state (terminate if full, i.e. step_count == word_size or similar)
        new_memory_state = state._memory_state

        result = state._test_fn(self.word_size, state._memory_state)
        reward = self.reward_fn(result)

        state = state.replace(_memory_state=new_memory_state)
        return state.replace(observation=self._observe(state, state.current_player), reward=reward)  # type: ignore

    def _observe(self, state: pgx.State, player_id: Array) -> Array:
        assert isinstance(state, SubleqState)
        observation = jnp.concatenate(
            [
                subleq_words_to_observation(state._memory_state, self.word_size),
                subleq_words_to_observation(state._example_input, self.word_size),
                subleq_words_to_observation(state._example_input_after, self.word_size),
                subleq_words_to_observation(state._example_output, self.word_size),
                subleq_words_to_observation(state._example_output_after, self.word_size),
            ]
        )
        return observation

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
