from typing import TYPE_CHECKING
from functools import partial

from dataclasses import replace

import jax
import jax.numpy as jnp
import jit_env

from jaxtyping import PRNGKeyArray

from jit_env import Environment, StepType, TimeStep, specs, EnvOptions
from jit_env import transition, truncation, termination

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


def make_smz_compat(
    true_env: Environment, planner_env: Environment, truncation_length: int
) -> tuple[Environment, Environment]:
    true_env = jit_env.wrappers.AutoReset(TimeoutWrapper(true_env, truncation_length, terminate_on_timeout=False))
    planner_env = TimeoutWrapper(planner_env, int(1e9), False)

    true_env = AddObservationToState(true_env)
    planner_env = AddObservationToState(planner_env)

    return true_env, planner_env


@dataclass
class ObsState[State, Observation]:
    key: PRNGKeyArray
    state: State
    observation: Observation


class AddObservationToState[State, Observation, Action](jit_env.Wrapper):

    def reset(self, key: PRNGKeyArray, *, options: EnvOptions = None) -> tuple[ObsState[State, Observation], TimeStep]:
        state, step = self.env.reset(key, options=options)
        return ObsState(state.key, state, step.observation), step

    def step(
        self, state: ObsState[State, Observation], action: Action
    ) -> tuple[ObsState[State, Observation], TimeStep]:
        state, step = self.env.step(state.state, action)
        return ObsState(state.key, state, step.observation), step


@dataclass
class TimedState[State]:
    key: PRNGKeyArray
    state: State
    t: jax.typing.ArrayLike


class TimeoutWrapper[State, Action](jit_env.Wrapper):

    def __init__(self, env: jit_env.Environment, timelimit: int, terminate_on_timeout: bool):
        super().__init__(env)
        self.timelimit = timelimit
        self.terminate_on_timeout = terminate_on_timeout

    def reset(self, key: PRNGKeyArray, *, options: EnvOptions = None) -> tuple[TimedState[State], TimeStep]:
        state, step = self.env.reset(key, options=options)
        carry = key if not hasattr(state, 'key') else state.key
        return TimedState(carry, state, 0), step

    def step(self, state: TimedState[State], action: Action) -> tuple[TimedState[State], TimeStep]:
        new_state, step = self.env.step(state.state, action)

        carry = state.key if not hasattr(new_state, 'key') else new_state.key
        new_timed_state = TimedState(carry, new_state, state.t + 1)

        if self.terminate_on_timeout:
            stopped_step = replace(step, step_type=StepType.LAST, discount=0.0)
        else:
            stopped_step = replace(step, step_type=StepType.LAST)

        new_step = jax.lax.cond(new_timed_state.t < self.timelimit, lambda: step, lambda: stopped_step)

        return new_timed_state, new_step


class BraxWrapper[State, Action](Environment):

    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env

    def reset(self, key: PRNGKeyArray, *, options: EnvOptions = None) -> tuple[State, TimeStep]:
        state = self.env.reset(key)
        return state, TimeStep(
            step_type=StepType.FIRST, observation=state.obs, reward=state.reward, discount=1.0, extras=state.info
        )

    def step(self, state: State, action: Action) -> tuple[State, TimeStep]:
        new_state = self.env.step(state, action)
        return new_state, TimeStep(
            step_type=jax.lax.select(new_state.done.astype(bool), StepType.LAST, StepType.MID),
            observation=new_state.obs,
            reward=new_state.reward,
            discount=1.0 - new_state.done,
            extras=new_state.info,
        )

    def reward_spec(self) -> specs.Spec:
        return specs.Array((), jnp.float32)

    def discount_spec(self) -> specs.Spec:
        return specs.BoundedArray((), jnp.float32, minimum=0.0, maximum=1.0)

    def observation_spec(self) -> specs.Spec:
        return specs.Array((self.env.observation_size,), jnp.float32)

    def action_spec(self) -> specs.Spec:
        return specs.BoundedArray(
            (self.env.action_size,),
            jnp.float32,
            minimum=-jnp.ones(self.env.action_size),
            maximum=jnp.ones(self.env.action_size),
        )


class JumanjiWrapper[State, Action](Environment):

    def __init__(self, env):
        super().__init__(renderer=env.render)
        self.env = env

    def reset(self, key: PRNGKeyArray, *, options: EnvOptions = None) -> tuple[State, TimeStep]:
        state, step = self.env.reset(key)
        return state, TimeStep(
            step_type=StepType.FIRST,
            observation=step.observation,
            reward=step.reward,
            discount=step.discount,
            extras=step.extras,
        )

    def step(self, state: State, action: Action) -> tuple[State, TimeStep]:
        new_state, step = self.env.step(state, action)

        return new_state, TimeStep(
            step_type=jax.lax.select(step.step_type == 2, StepType.LAST, StepType.MID),
            observation=step.observation,
            reward=step.reward,
            discount=step.discount,
            extras=step.extras,
        )

    def reward_spec(self) -> specs.Spec:
        return specs.Array((), jnp.float32)

    def discount_spec(self) -> specs.Spec:
        return specs.BoundedArray((), jnp.float32, minimum=0.0, maximum=1.0)

    def observation_spec(self) -> specs.Spec:
        return self.env.observation_spec

    def action_spec(self) -> specs.Spec:
        return self.env.action_spec


class PGXWrapper[State, Action](Environment):
    # TODO: not properly tested; implement alternating player transitions

    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env

    def reset(self, key: PRNGKeyArray, *, options: EnvOptions = None) -> tuple[State, TimeStep]:
        state = self.env.reset(key)
        return state, TimeStep(
            step_type=StepType.FIRST,
            observation=state.observation,
            reward=state.reward[0],
            discount=1.0 - state.terminated.all(),
            extras={},
        )

    def step(self, state: State, action: Action) -> tuple[State, TimeStep]:
        new_state = self.env.step(state, action)

        return new_state, TimeStep(
            step_type=jax.lax.select(new_state.terminated.all(), StepType.LAST, StepType.MID),
            observation=new_state.observation,
            reward=new_state.reward[0],
            discount=1.0 - new_state.terminated.all(),
            extras={},
        )

    def reward_spec(self) -> specs.Spec:
        return specs.Array((), jnp.float32)

    def discount_spec(self) -> specs.Spec:
        return specs.BoundedArray((), jnp.float32, minimum=0.0, maximum=1.0)

    def observation_spec(self) -> specs.Spec:
        return self.env.observation_spec

    def action_spec(self) -> specs.Spec:
        return self.env.action_spec
