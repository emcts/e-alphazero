from functools import partial

import emctx
import jax
import jax.numpy as jnp
import pgx  # type: ignore

from config import Config
from context import Context
from type_aliases import Array, Model, PRNGKey


@partial(jax.pmap, static_broadcasted_argnums=[1, 2])
def evaluate(model: Model, config: Config, context: Context, rng_key: PRNGKey) -> Array:
    model_params, model_state = model
    batch_size = config.num_eval_episodes // len(context.devices)

    def cond_fn(tup: tuple[pgx.State, PRNGKey, Array, int]) -> bool:
        states, _, _, counter = tup
        return jnp.logical_not(states.terminated.all()) & (counter <= config.max_episode_length)

    def loop_fn(tup: tuple[pgx.State, PRNGKey, Array, int]) -> tuple[pgx.State, PRNGKey, Array, int]:
        states, rng_key, sum_of_rewards, counter = tup
        rng_key, key_for_search, key_for_next_step = jax.random.split(rng_key, 3)

        (exploitation_logits, _exploration_logits, value, value_epistemic_variance, _reward_epistemic_variance), _ = (
            context.forward.apply(model_params, model_state, states.observation, is_training=False)
        )
        root = emctx.EpistemicRootFnOutput(
            prior_logits=exploitation_logits,  # type: ignore
            value=value,  # type: ignore
            value_epistemic_variance=value_epistemic_variance,  # type: ignore
            embedding=states,  # type: ignore
            beta=config.exploitation_beta * jnp.ones_like(value),  # type: ignore
        )
        policy_output = emctx.epistemic_gumbel_muzero_policy(
            params=model,
            rng_key=key_for_search,
            root=root,
            recurrent_fn=context.evaluation_recurrent_fn,
            num_simulations=config.selfplay_simulations_per_step,
            invalid_actions=~states.legal_action_mask,
            qtransform=emctx.epistemic_qtransform_completed_by_mix_value,  # type: ignore
            gumbel_scale=0.0,
        )
        keys = jax.random.split(key_for_next_step, batch_size)
        next_states = jax.vmap(context.env.step)(states, policy_output.action, keys)
        rewards = next_states.rewards[jnp.arange(states.rewards.shape[0]), states.current_player]
        counter = counter + 1
        return next_states, rng_key, sum_of_rewards + rewards, counter

    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    states = jax.vmap(context.env.init)(keys)

    states, _, sum_of_rewards, _ = jax.lax.while_loop(cond_fn, loop_fn, (states, rng_key, jnp.zeros(batch_size), 0))
    return sum_of_rewards.mean()
