from functools import partial

import jax
import jax.numpy as jnp
import optax

from context import Context
from reanalyze import ReanalyzeOutput
from type_aliases import Model


def loss_fn(model_params, model_state, context: Context, reanalyze_output: ReanalyzeOutput):
    MINIMUM_LOG_UBE_TARGET = -10
    (
        exploitation_logits,
        exploration_logits,
        value,
        value_epistemic_variance,
        reward_epistemic_variance,
    ), model_state = context.forward.apply(
        model_params, model_state, reanalyze_output.observation, is_training=True, update_hash=True
    )

    value_loss = optax.l2_loss(value, reanalyze_output.value_target)
    value_loss = jnp.mean(value_loss)  # TODO: figure out if mask is needed because of episode truncation

    # Compute error for priority:
    error_beta = jax.lax.cond(context.exploration_beta > 0.0, lambda: 1.0, lambda: 0.0)
    # The UBE prediction and target need to be rescaled [0,1] -> [0,max] -> sqrt([0,max])
    rescaled_ube_prediction = jnp.sqrt(jnp.abs(value_epistemic_variance) * context.max_ube)
    rescaled_ube_target = jnp.sqrt(jnp.abs(reanalyze_output.ube_target) * context.max_ube)
    priority_score = jnp.abs(
        value
        + error_beta * rescaled_ube_prediction
        - (reanalyze_output.value_target + error_beta * rescaled_ube_target)
    )

    ube_loss = optax.l2_loss(value_epistemic_variance, reanalyze_output.ube_target)
    ube_loss = jnp.mean(ube_loss)

    exploitation_policy_loss = optax.softmax_cross_entropy(
        exploitation_logits, reanalyze_output.exploitation_policy_target
    )
    exploitation_policy_loss = jnp.mean(exploitation_policy_loss)

    exploration_policy_loss = optax.softmax_cross_entropy(
        exploration_logits, reanalyze_output.exploration_policy_target
    )
    exploration_policy_loss = jnp.mean(exploration_policy_loss)

    total_loss = value_loss + exploitation_policy_loss + exploration_policy_loss + ube_loss

    # Log the policies entropies
    # Compute the probabilities by applying softmax
    exploitation_probs = jax.nn.softmax(exploitation_logits, axis=-1)
    # Compute the log-probabilities
    exploitation_log_probs = jax.nn.log_softmax(exploitation_probs, axis=-1)
    # Compute entropy: -sum(p * log(p))
    mean_exploitation_policy_entropy = -jnp.sum(exploitation_probs * exploitation_log_probs, axis=-1).mean()

    # Compute the probabilities by applying softmax
    exploration_probs = jax.nn.softmax(exploration_logits, axis=-1)
    # Compute the log-probabilities
    exploration_log_probs = jax.nn.log_softmax(exploration_logits, axis=-1)
    # Compute entropy: -sum(p * log(p))
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
