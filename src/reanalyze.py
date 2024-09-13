from functools import partial
from typing import NamedTuple

import chex
import emctx
import flashbax as fbx  # type: ignore
import jax
import jax.numpy as jnp
from pgx.experimental import auto_reset  # type: ignore

from config import Config
from context import Context
from type_aliases import Array, Model, PRNGKey


def mask_invalid_actions(logits: Array, invalid_actions: Array) -> Array:
    """
    Returns logits with zero mass to invalid actions.
    Copied from `mctx._src.policies`.
    """
    if invalid_actions is None:
        return logits
    chex.assert_equal_shape([logits, invalid_actions])
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    # At the end of an episode, all actions can be invalid. A softmax would then
    # produce NaNs, if using -inf for the logits. We avoid the NaNs by using
    # a finite `min_logit` for the invalid actions.
    min_logit = jnp.finfo(logits.dtype).min
    return jnp.where(invalid_actions, min_logit, logits)  # type: ignore


def complete_qs(qvalues, visit_counts, value):
    """Returns completed Q-values (or Q-value uncertainty), with the `value` for unvisited actions."""
    chex.assert_equal_shape([qvalues, visit_counts])
    chex.assert_shape(value, [])

    # The missing qvalues are replaced by the value.
    completed_qvalues = jnp.where(visit_counts > 0, qvalues, value)
    chex.assert_equal_shape([completed_qvalues, qvalues])
    return completed_qvalues


class ReanalyzeOutput(NamedTuple):
    observation: Array
    next_observation: Array  # for reward variance (local uncertainty)
    value_target: Array
    ube_target: Array
    exploration_policy_target: Array
    exploitation_policy_target: Array


@partial(jax.pmap, static_broadcasted_argnums=[1, 2])
def reanalyze(
    model: Model,
    config: Config,
    context: Context,
    experience_pair: fbx.prioritised_flat_buffer.ExperiencePair,
    rng_key: PRNGKey,
) -> ReanalyzeOutput:
    model_params, model_state = model
    states = experience_pair.first
    next_states = experience_pair.second

    observation = states.observation
    invalid_actions = ~states.legal_action_mask

    (exploitation_logits, _exploration_logits, value, value_epistemic_variance, _reward_epistemic_variance), _ = (
        context.forward.apply(model_params, model_state, observation, is_training=False)
    )
    root = emctx.EpistemicRootFnOutput(
        prior_logits=exploitation_logits,  # type: ignore
        value=value,  # type: ignore
        value_epistemic_variance=value_epistemic_variance,  # type: ignore
        embedding=states,  # type: ignore
        beta=jnp.zeros_like(value),  # type: ignore
    )
    policy_output = emctx.epistemic_gumbel_muzero_policy(
        params=model,
        rng_key=rng_key,
        root=root,
        recurrent_fn=context.reanalyze_recurrent_fn,
        num_simulations=config.reanalyze_simulations_per_step,
        invalid_actions=invalid_actions,
        qtransform=emctx.epistemic_qtransform_completed_by_mix_value,  # type: ignore  # TODO: Fix the type in emctx
    )
    search_summary = policy_output.search_tree.epistemic_summary()
    value_target = search_summary.qvalues[jnp.arange(search_summary.qvalues.shape[0]), policy_output.action]  # type: ignore
    ube_target = jnp.max(search_summary.qvalues_epistemic_variance, axis=1)
    rescaled_ube_target = ube_target / config.max_ube

    completed_q_and_std_scores: Array = mask_invalid_actions(
        jax.vmap(complete_qs)(
            search_summary.qvalues + config.exploration_beta * jnp.sqrt(search_summary.qvalues_epistemic_variance),
            search_summary.visit_counts,
            search_summary.value + config.exploration_beta * search_summary.value_epistemic_std,
        ),  # type: ignore
        invalid_actions,
    )
    exploration_policy_target = jax.nn.softmax(
        completed_q_and_std_scores * config.exploration_policy_target_temperature
    )

    return ReanalyzeOutput(
        observation=observation,
        next_observation=next_states.observation,  # FIXME: This could be initial state from next episode
        value_target=value_target,
        ube_target=rescaled_ube_target,
        exploitation_policy_target=policy_output.action_weights,  # type: ignore
        exploration_policy_target=exploration_policy_target,
    )
