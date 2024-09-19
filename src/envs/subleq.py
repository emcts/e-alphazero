from enum import IntEnum, auto
from typing import Callable, NamedTuple
import chex
import jax
import jax.numpy as jnp
import pgx  # type: ignore
from pgx._src.struct import dataclass  # type: ignore

from type_aliases import Array, PRNGKey

# Environment ID (for pgx).
ENV_ID = "subleq"
# The maximum size of any test input. Effects the size of the observation.
MAXIMUM_INPUT_LENGTH = 8
# The maximum size of any test output. Effects the size of the observation.
MAXIMUM_OUTPUT_LENGTH = MAXIMUM_INPUT_LENGTH
# Terminate the program if it does more than this many cycles.
MAX_CYCLE_COUNT = 200


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
    """Outcome of testing a program against the whole test set."""

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
    """Outcome of a program simulation on a single input."""

    input_after: Array
    output_after: Array
    bytes_used: Array
    cycles_used: Array
    correct: Array


SubleqRewardFn = Callable[[SubleqTestResult], Array]
SubleqTestFn = Callable[[int, Array], SubleqTestResult]


def simulate(word_size: int, memory_state: Array, test_input: Array, test_output: Array) -> SubleqSimulateResult:
    """Simulate the program given in the memory state for a specific test input and output."""
    chex.assert_rank([memory_state, test_input, test_output], 1)
    ADDRESS_MAX = word_size - 4  # Maximum user-modifiable address
    ADDRESS_IN = word_size - 3  # Reads a value from input (writes are ignored)
    ADDRESS_OUT = word_size - 2  # Writes a result to output (reads as zero)
    ADDRESS_HALT = word_size - 1  # Terminates the program when accessed (reads and write are ignored)

    class InterpreterState(NamedTuple):
        """State of the interpreter at any point in the simulation."""

        memory_state: Array
        input_state: Array
        output_state: Array
        output_cursor: int
        cursor_position: int
        bytes_used: int
        cycles: int
        halt: bool
        error: bool

    class ReadMemoryResult(NamedTuple):
        """Outcome of reading from memory (or input)."""

        value_read: int
        input_state: Array
        accessed_input: bool
        halt: bool
        error: bool

    def read_memory(memory_state: Array, address: int, input_state: Array) -> ReadMemoryResult:
        default_result = ReadMemoryResult(
            value_read=0,
            accessed_input=False,
            error=False,
            halt=False,
        )
        match address:
            case x if x == ADDRESS_IN:
                # Check if there is any more input left.
                if input_state[0] >= word_size:
                    return default_result.replace(error=True)
                value_read = input_state[0]
                return default_result.replace(
                    value_read=value_read,
                    accessed_input=True,
                )
            case x if x == ADDRESS_OUT:
                return default_result
            case x if x == ADDRESS_HALT:
                return default_result.replace(halt=True)
            case _:
                return default_result.replace(value_read=memory_state[address])

    class WriteMemoryResult(NamedTuple):
        """Outcome of writing into memory (or output)."""

        memory_state: Array
        output_state: Array
        output_cursor: int
        modified_output: bool
        halt: bool
        error: bool

    def write_memory(
        memory_state: Array, address: int, value: int, output_state: Array, output_cursor: int
    ) -> WriteMemoryResult:
        default_result = WriteMemoryResult(
            memory_state=memory_state,
            output_state=output_state,
            output_cursor=output_cursor,
            modified_output=False,
            halt=False,
            error=False,
        )
        match address:
            case x if x == ADDRESS_IN:
                return default_result
            case x if x == ADDRESS_OUT:
                # Check if there is space in the output.
                if output_cursor >= MAXIMUM_OUTPUT_LENGTH:
                    return default_result.replace(error=True)
                return default_result.replace(
                    output_state=output_state.at[output_cursor].set(value),
                    output_cursor=output_cursor + 1,
                    modified_output=True,
                )
            case x if x == ADDRESS_HALT:
                return default_result.replace(halt=True)
            case _:
                return default_result.replace(memory_state=memory_state.at[address].set(value))

    def cond_fn(state: InterpreterState) -> bool:
        """Condition function which determines whether the simulation should continue."""
        return (not state.error) and (not state.halt) and (state.cycles < MAX_CYCLE_COUNT)

    def body_fn(state: InterpreterState) -> InterpreterState:
        """A single step in the Subleq simulation."""
        cursor_position = state.cursor_position
        cycles = state.cycles + 1

        # Instruction would try to read out of bounds.
        if cursor_position + 2 >= word_size:
            return state.replace(  # type: ignore
                cycles=cycles,
                error=True,
            )

        memory_state = state.memory_state
        input_state = state.input_state
        # FIXME: `bytes_used` should also take into account the addresses read from (i.e. values of A, B, C),
        # and also just how many bytes were written to memory to begin with.
        bytes_used = max(state.cursor_position + 3, state.bytes_used)

        # subleq A B C
        a = memory_state[cursor_position]
        b = memory_state[cursor_position + 1]
        c = memory_state[cursor_position + 2]

        a_result = read_memory(memory_state, a, input_state)
        b_result = read_memory(memory_state, b, input_state)
        c_result = read_memory(memory_state, c, input_state)

        # mem[A] = mem[A] - mem[B]
        value = (a_result.value_read - b_result.value_read) % word_size
        write_result = write_memory(memory_state, a, value, state.output_state, state.output_cursor)
        memory_state = write_result.memory_state
        output_state = write_result.output_state
        output_cursor = write_result.output_cursor

        # if mem[A] <= 0 { goto mem[C] }
        execute_jump = value == 0 or value >= word_size / 2
        cursor_position = c_result.value_read if execute_jump else cursor_position + 3

        # Update input state if anything read from it. Input can advance only by 1 per instruction.
        if a_result.accessed_input or b_result.accessed_input or (execute_jump and c_result.accessed_input):
            input_state = jnp.roll(input_state, -1)
            input_state = input_state.at[input_state.shape[0] - 1].set(word_size)

        halt = (
            a_result.halt
            or b_result.halt
            or (execute_jump and c_result.halt)
            or write_result.halt
            # Halt on correct output (like in the game).
            or output_state == test_output
        )
        error = (
            a_result.error
            or b_result.error
            or (execute_jump and c_result.error)
            or write_result.error
            or (
                # Error on first incorrect output (like in the game).
                write_result.modified_output
                and output_state[output_cursor - 1] != test_output[output_cursor - 1]
            )
        )

        return InterpreterState(
            memory_state=memory_state,
            input_state=input_state,
            output_state=output_state,
            output_cursor=output_cursor,
            cursor_position=cursor_position,
            bytes_used=bytes_used,
            cycles=cycles,
            halt=halt,
            error=error,
        )

    # Execute a subleq program.
    after_execution = jax.lax.while_loop(
        cond_fn,
        body_fn,
        InterpreterState(
            memory_state=memory_state,
            input_state=test_input,
            output_state=jnp.ones_like(test_output),
            output_cursor=0,
            cursor_position=0,
            cycles=0,
            bytes_used=0,
            halt=False,
            error=False,
        ),
    )
    return SubleqSimulateResult(
        input_after=after_execution.input_state,
        output_after=after_execution.output_state,
        bytes_used=after_execution.bytes_used,
        cycles_used=after_execution.cycles,
        correct=(not after_execution.error) and test_output == after_execution.output_state,
    )


def get_test_fn(task: SubleqTask) -> SubleqTestFn:
    # TODO: Do this for many tests
    test_input: Array
    test_output: Array
    match task:
        case SubleqTask.NEGATION:
            # TODO: Have more than one test per task.
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
            pad(test_output, MAXIMUM_OUTPUT_LENGTH, word_size),
        )

        return SubleqTestResult(
            solved=result.correct,  # TODO: should have shape [batch_size], and should depend on the whole test_set?
            example_input=test_input,  # TODO: change to only take the first
            example_output=test_output,  # TODO: change to only take the first
            input_after=result.input_after,  # TODO: change to only take the first
            output_after=result.output_after,  # TODO: change to only take the first
            bytes_used=result.bytes_used,  # TODO: Take the max
            cycles_used=result.cycles_used,  # TODO: Take the max
        )

    return test_fn


def solved_or_not(execution_result: SubleqTestResult) -> Array:
    """Simplest reward function which just gives 1 if the program solves all the tests, and 0 otherwise."""
    return execution_result.solved.astype(jnp.float32)


def lowest_bytes(execution_result: SubleqTestResult) -> Array:
    """Prioritizes solutions which use the fewest amount of bytes. `solved / (1 + bytes_used)`"""
    return execution_result.solved.astype(jnp.float32) / (1 + execution_result.bytes_used)


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
    _solved: Array = jnp.bool(False)

    @property
    def env_id(self) -> pgx.EnvId:
        """Environment id (e.g. "go_19x19")"""
        return ENV_ID  # type: ignore


class Subleq(pgx.Env):
    """
    Reference description of Subleq: https://github.com/jaredkrinke/sic1/blob/master/sic1-assembly.md

    Every subleq task is a different programming challenge, and each set of tasks is a separate Subleq environment.
    The environment is also parametrized by the word size N, which determines how many possible values a word can take.
    Since we use one-hot encoding, it is also related to the row size of the observation.

    At each step the agent writes a single word into the memory. It has exactly N choices, as that is the word size.
    The program is tested after each byte is written to see if it solves the problem. The input and output
    after execution on the example input is included in the observation. The reward received is determined by a function
    of the test results, and it is a parameter of the environment.

    The state the agent observes is composed of:
    - Memory state before execution (i.e. the code).    [N,                     N + 1]
    - Example input.                                    [MAXIMUM_INPUT_LENGTH,  N + 1]
    - Example output.                                   [MAXIMUM_OUTPUT_LENGTH, N + 1]
    - Input after execution.                            [MAXIMUM_INPUT_LENGTH,  N + 1]
    - Output after execution.                           [MAXIMUM_OUTPUT_LENGTH, N + 1]

    Final shape: [N + 2 * (MAXIMUM_INPUT_LENGTH + MAXIMUM_OUTPUT_LENGTH), N + 1]

    The row size is N+1 to allow for a special token (value or position = N) which indicates empty or missing.
    This is needed because the input and output may not be full, and we need to distinguish from normal values.

    A possible input is `[1, 2, 3, _, _, _]`, so we need a column to represent `_`. Same goes for output.
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

    def _init(self, key: PRNGKey) -> SubleqState:
        task: SubleqTask = jax.random.choice(key, self.tasks).item()
        legal_action_mask = jnp.ones(self.word_size, dtype=jnp.bool)
        state = SubleqState(
            legal_action_mask=legal_action_mask,
            _task=task,
            _test_fn=get_test_fn(task),
            _memory_state=jnp.zeros(self.word_size, dtype=jnp.bool),
        )
        result = state._test_fn(self.word_size, state._memory_state)
        reward = self.reward_fn(result)
        state = state.replace(
            _example_input=result.example_input,
            _example_output=result.example_output,
            _example_input_after=result.input_after,
            _example_output_after=result.output_after,
        )
        return state.replace(observation=self._observe(state, state.current_player), reward=reward)  # type: ignore

    def _step(self, state: SubleqState, action: Array, key: PRNGKey) -> SubleqState:
        assert isinstance(state, SubleqState)
        chex.assert_rank([state.observation, state._memory_state, action], [2, 1, 1])

        def execute_step_if_not_terminated(state: SubleqState, action: Array) -> SubleqState:
            # Add a byte to the memory state.
            word_to_write = jnp.argmax(action, axis=1)
            new_memory_state = state._memory_state.at[state._step_count - 1].set(word_to_write)
            # Run the program.
            result = state._test_fn(self.word_size, new_memory_state)
            # Calculate reward.
            reward = self.reward_fn(result)
            # Update properties related to observation.
            state = state.replace(
                _memory_state=new_memory_state,
                _example_input=result.example_input,
                _example_output=result.example_output,
                _example_input_after=result.input_after,
                _example_output_after=result.output_after,
            )
            return state.replace(observation=self._observe(state, state.current_player), reward=reward)

        return jax.lax.cond(
            state._step_count >= self.word_size or state._solved,
            state.replace(terminated=True),
            execute_step_if_not_terminated(state, action),
        )

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
        chex.assert_axis_dimension(observation, 0, self.word_size + 2 * (MAXIMUM_INPUT_LENGTH + MAXIMUM_OUTPUT_LENGTH))
        chex.assert_axis_dimension(observation, 1, self.token_size)
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
