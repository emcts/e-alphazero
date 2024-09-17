from functools import partial
from typing import NamedTuple

import chex
import emctx
import jax
import jax.numpy as jnp
import pgx
# from pgx.experimental import auto_reset  # type: ignore

from config import Config
from context import Context
from type_aliases import Array, Model, PRNGKey
from typing import Optional


class SelfplayOutput(NamedTuple):
    state: pgx.State
    root_value: Array
    root_epistemic_std: Array
    value_prediction: Array
    ube_prediction: Array
    q_values_epistemic_variance: Array


def auto_reset(step_fn, init_fn):
    """Auto reset wrapper.

    We have a concern about the final state before staging this wrapper:

    When auto-reset happens, the termianl (or truncated) state/observation is
    replaced by initial state/observation, It's ok if it's termination.
    However, when truncation happens, value of truncated state/observation
    might be used by agents (by bootstrap). So it must be stored somewhere.
    For example,

    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/autoreset.py#L59

    However, currently, truncation does *NOT* actually happens because
    all of Pgx environments (games) are finite-horizon and
    terminates within reasonable # of steps.
    Note that chess, shogi, and Go have `max_termination_steps` as AlphaZero.
    So, this implementation is enough (so far).

    2. Performance
    """

    def wrapped_step_fn(
            state: pgx.State, action: Array, key: Optional[PRNGKey] = None
    ):
        assert key is not None, (
            "v2.0.0 changes the signature of auto reset. Please specify PRNGKey at the third argument:\n\n"
            "  * <  v2.0.0: step_fn(state, action)\n"
            "  * >= v2.0.0: step_fn(state, action, key)\n\n"
            "Note that codes under pgx.experimental are subject to change without notice."
        )

        # Original behavior:
        # If state is already terminal, make it not-terminal.
        # Step the environment.
        # If new state is terminal reset the environment.

        # Modified behavior:
        # If state is already terminal, reset the environment.
        # Otherwise, step
        key1, key2 = jax.random.split(key)
        state = jax.lax.cond(
            (state.terminated | state.truncated),   # If state is already terminal
            lambda: init_fn(key1),                  # reset the environent
            lambda: step_fn(state, action, key2),   # step into a new state
        )

        return state

    return wrapped_step_fn


@partial(jax.pmap, static_broadcasted_argnums=[1, 2])
def selfplay(
    model: Model, config: Config, context: Context, last_states: pgx.State, rng_key: PRNGKey
) -> tuple[pgx.State, SelfplayOutput]:
    model_params, model_state = model
    self_play_batch_size = config.selfplay_batch_size // len(context.devices)
    num_actions = context.env.num_actions

    def step_fn(states: pgx.State, key: PRNGKey) -> tuple[pgx.State, SelfplayOutput]:
        key1, key2, key3, key4 = jax.random.split(key, num=4)

        (exploitation_logits, exploration_logits, value, value_epistemic_variance, _reward_epistemic_variance), _ = (
            context.forward.apply(model_params, model_state, states.observation, is_training=False)
        )
        selfplay_beta = jax.lax.cond(config.directed_exploration, lambda: config.exploration_beta, lambda: 0.0)
        policy_logits = jax.lax.cond(
            config.directed_exploration, lambda: exploration_logits, lambda: exploitation_logits
        )
        policy_logits = jax.lax.cond(
            config.uniform_search_policy, lambda: jnp.ones_like(policy_logits), lambda: policy_logits
        )

        root = emctx.EpistemicRootFnOutput(
            prior_logits=policy_logits,  # type: ignore
            value=value,  # type: ignore
            value_epistemic_variance=value_epistemic_variance,  # type: ignore
            embedding=states,  # type: ignore
            beta=selfplay_beta * jnp.linspace(0, 1, num=value.size).reshape(value.shape),  # type: ignore
        )
        policy_output = emctx.epistemic_gumbel_muzero_policy(
            params=model,
            rng_key=key1,
            root=root,
            recurrent_fn=context.selfplay_recurrent_fn,
            num_simulations=config.selfplay_simulations_per_step,
            invalid_actions=~states.legal_action_mask,
            qtransform=partial(
                emctx.epistemic_qtransform_completed_by_mix_value, rescale_values=config.rescale_q_values_in_search
            ),  # type: ignore
        )
        keys = jax.random.split(key2, self_play_batch_size)
        search_summary = policy_output.search_tree.epistemic_summary()
        root_values = search_summary.value
        root_epistemic_stds = search_summary.value_epistemic_std
        # Note: for GumbelMZ this is essentially det. argmax, while for MZ (PUCT) this is sampled from counts.
        action_chosen_by_search_tree = policy_output.action
        # Sample from visits
        sampled_action = jax.random.categorical(key3, search_summary.visit_probs, axis=-1)
        # Sample from improved policy
        sampled_action_from_improved_policy = jax.random.categorical(key4, policy_output.action_weights, axis=-1)
        chex.assert_equal_shape([action_chosen_by_search_tree, sampled_action, sampled_action_from_improved_policy])
        chosen_action = jax.lax.cond(
            config.sample_actions, lambda: sampled_action, lambda: action_chosen_by_search_tree
        )
        chosen_action = jax.lax.cond(
            config.sample_from_improved_policy, lambda: sampled_action_from_improved_policy, lambda: chosen_action
        )
        next_state = jax.vmap(auto_reset(context.env.step, context.env.init))(states, chosen_action, keys)
        return next_state, SelfplayOutput(
            state=next_state,
            root_value=root_values,  # type: ignore
            root_epistemic_std=root_epistemic_stds,  # type: ignore
            value_prediction=value,
            ube_prediction=value_epistemic_variance,
            q_values_epistemic_variance=search_summary.qvalues_epistemic_variance,  # type: ignore
        )

    rng_key, sub_key = jax.random.split(rng_key)
    starting_states = last_states
    key_seq = jax.random.split(rng_key, config.selfplay_steps)
    last_states, data = jax.lax.scan(step_fn, starting_states, key_seq)
    return last_states, data


@partial(jax.pmap, static_broadcasted_argnums=[0, 1])
def uniformrandomplay(config: Config, context: Context, rng_key: PRNGKey) -> tuple[pgx.State, pgx.State]:
    self_play_batch_size = config.selfplay_batch_size // len(context.devices)
    num_actions = context.env.num_actions

    def pre_training_step_fn(states: pgx.State, key: PRNGKey) -> tuple[pgx.State, pgx.State]:
        key1, key2 = jax.random.split(key)
        keys = jax.random.split(key2, self_play_batch_size)
        action = jax.random.randint(key1, shape=(self_play_batch_size,), minval=0, maxval=num_actions)
        next_state = jax.vmap(auto_reset(context.env.step, context.env.init))(states, action, keys)
        return next_state, states

    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, self_play_batch_size)
    states = jax.vmap(context.env.init)(keys)
    key_seq = jax.random.split(rng_key, config.selfplay_steps)
    last_states, states = jax.lax.scan(pre_training_step_fn, states, key_seq)

    return last_states, states
