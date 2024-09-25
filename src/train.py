from functools import partial

import chex
import jax
import jax.numpy as jnp
import optax
import jax.nn as nn

from context import Context
from reanalyze import ReanalyzeOutput
from type_aliases import Model


def loss_fn(model_params, model_state, context: Context, reanalyze_output: ReanalyzeOutput):
    (
        exploitation_logits,
        exploration_logits,
        value,
        value_epistemic_variance,
        reward_epistemic_variance,
    ), model_state = context.forward.apply(
        model_params, model_state, reanalyze_output.observation, is_training=True, update_hash=True
    )

    # Compute losses
    # We scale the value target by value_scale, because value pred. is between [-1,1] for stability
    value_loss = optax.l2_loss(value, reanalyze_output.value_target)
    # We scale the ube target by ube_scale, because ube pred. is between [0,1] for stability
    ube_loss = optax.l2_loss(value_epistemic_variance, reanalyze_output.ube_target)
    exploitation_policy_loss = optax.softmax_cross_entropy(
        exploitation_logits, reanalyze_output.exploitation_policy_target)
    exploration_policy_loss = optax.softmax_cross_entropy(
        exploration_logits, reanalyze_output.exploration_policy_target)

    # Compute loss weights, based on Sunrise, https://arxiv.org/pdf/2007.04938
    epistemic_loss_weights = 0.5 + nn.sigmoid(-1 * reanalyze_output.ube_target * context.loss_weighting_temperature)
    epistemic_loss_weights = jax.lax.cond(context.weigh_losses, lambda: epistemic_loss_weights,
                                          lambda: jnp.ones_like(value_loss))

    chex.assert_equal_shape([ube_loss, value_loss, exploitation_policy_loss, exploration_policy_loss, epistemic_loss_weights])

    total_loss = jnp.mean(epistemic_loss_weights * (value_loss + exploitation_policy_loss) +
                          exploration_policy_loss + ube_loss)

    # Compute error for priority:
    error_beta = jax.lax.cond(context.exploration_beta > 0.0 and context.directed_exploration, lambda: 0.01, lambda: 0.0)
    # The UBE prediction and target need to be rescaled [0,1] -> [0,max] -> sqrt([0,max])
    rescaled_ube_prediction = jnp.sqrt(jnp.abs(value_epistemic_variance))
    rescaled_ube_target = jnp.sqrt(reanalyze_output.ube_target)
    priority_score = jnp.abs(
        value
        + error_beta * rescaled_ube_prediction
        - (reanalyze_output.value_target + error_beta * rescaled_ube_target)
    )
    # If we use epistemic loss weighting, we should also adjust the priorities
    priority_score = jax.lax.cond(context.weigh_losses, lambda: priority_score * epistemic_loss_weights,
                                  lambda: priority_score)

    # Log the policies entropies
    # Compute the probabilities by applying softmax
    exploitation_probs = jax.nn.softmax(exploitation_logits, axis=-1)
    # Compute the log-probabilities
    exploitation_log_probs = jax.nn.log_softmax(exploitation_probs, axis=-1)
    # Compute entropy: -sum(p * log(p))
    mean_exploitation_policy_entropy = -jnp.sum(exploitation_probs * exploitation_log_probs, axis=-1).mean()

    exploration_probs = jax.nn.softmax(exploration_logits, axis=-1)
    exploration_log_probs = jax.nn.log_softmax(exploration_logits, axis=-1)
    mean_exploration_policy_entropy = -jnp.sum(exploration_probs * exploration_log_probs, axis=-1).mean()

    # batch novelty, i.e. how many of these states have we "seen" before.
    batch_novelty = reward_epistemic_variance.mean()

    return total_loss, (
        model_state,
        priority_score,
        mean_exploitation_policy_entropy,
        mean_exploration_policy_entropy,
        value_loss,
        ube_loss,
        exploitation_policy_loss,
        exploration_policy_loss,
        batch_novelty,
    )


@partial(jax.pmap, axis_name="i", static_broadcasted_argnums=[2])
def train(model: Model, opt_state, context: Context, reanalyze_output: ReanalyzeOutput):
    model_params, model_state = model
    grads, (model_state, *statistics) = jax.grad(loss_fn, has_aux=True)(
        model_params, model_state, context, reanalyze_output
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = context.optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    model = (model_params, model_state)  # type: ignore

    return (model, opt_state, *statistics)
