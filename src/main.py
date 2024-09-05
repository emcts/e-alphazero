from functools import partial
import datetime
import time
import os
import pickle
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import emctx
import omegaconf
import optax  # type: ignore
import pgx  # type: ignore
import pydantic
import wandb
import flashbax as fbx  # type: ignore
import chex
from pgx.experimental import auto_reset  # type: ignore

from network import EpistemicAZNet

type ForwardFn = hk.TransformedWithState
type SelfplayOutput = pgx.State

WANDB_PROJECT: str = "e-alphazero"


class Config(pydantic.BaseModel):
    """Hyperparameter configuration"""

    # general
    seed: int = 0
    env_id: pgx.EnvId = "minatar-asterix"
    maximum_number_of_iterations: int = 400
    beta: float = 0.1
    # network
    hidden_layers: int = 3
    layer_size: int = 64
    # selfplay
    selfplay_batch_size: int = 64  # FIXME: Return these hyperparameters to normal numbers
    simulations_per_step: int = 4
    selfplay_steps: int = 32
    # reanalyze
    reanalyze_batch_size: int = 64
    max_replay_buffer_length: int = 1_000_000
    min_replay_buffer_length: int = 64
    priority_exponent: float = 0.6
    reanalyze_loops_per_selfplay: int = 1
    # training
    learning_rate: float = 0.001
    # checkpoints / eval
    checkpoint_interval: int = 5
    eval_interval: int = 5
    evaluation_batch: int = 64
    # targets
    exploration_policy_target_temperature: float = 1.0

    class Config:
        extra = "forbid"

    # HACK: Should be fine since there will only ever be one `Config`.
    def __hash__(self):
        return 0


class Context(NamedTuple):
    """Context which stays the same throughout training."""

    env: pgx.Env
    devices: list[jax.Device]
    forward: ForwardFn
    exploration_recurrent_fn: emctx.EpistemicRecurrentFn
    exploitation_recurrent_fn: emctx.EpistemicRecurrentFn
    optimizer: optax.GradientTransformation

    # HACK: Should be fine since there will only ever be one `Context`.
    def __hash__(self):
        return 1


def get_network(env: pgx.Env, config: Config):
    return EpistemicAZNet(  # FIXME: Switch to the other network
        num_actions=env.num_actions,
        # num_hidden_layers=config.hidden_layers,
        # layer_size=config.layer_size,
    )


# Set up the training model and optimizer.
def get_forward_fn(env: pgx.Env, config: Config) -> ForwardFn:
    def forward_fn(
        x, is_training: bool = True, update_hash: bool = False
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        net = get_network(env, config)
        (
            exploitation_policy_logits,
            exploration_policy_logits,
            value,
            value_epistemic_variance,
            reward_epistemic_variance,
        ) = net(x, is_training=is_training, test_local_stats=False, update_hash=update_hash)
        return (
            exploitation_policy_logits,
            exploration_policy_logits,
            value,
            value_epistemic_variance,
            reward_epistemic_variance,
        )

    return hk.without_apply_rng(hk.transform_with_state(forward_fn))


def get_epistemic_recurrent_fn(
    env: pgx.Env, forward: ForwardFn, batch_size: int, exploration: bool
) -> emctx.EpistemicRecurrentFn:
    def epistemic_recurrent_fn(
        model, rng_key: chex.PRNGKey, action: chex.Array, state: pgx.State
    ) -> tuple[emctx.EpistemicRecurrentFnOutput, pgx.State]:
        model_params, model_state = model

        current_player = state.current_player
        keys = jax.random.split(rng_key, batch_size)
        state = jax.vmap(env.step)(state, action, keys)
        value: chex.Array
        (exploitation_logits, exploration_logits, value, value_epistemic_variance, reward_epistemic_variance), _ = (
            forward.apply(model_params, model_state, state.observation, is_training=False)
        )
        logits = jax.lax.cond(exploration, lambda: exploration_logits, lambda: exploitation_logits)

        # Subtract max from logits to improve numerical stability.
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        # Mask invalid actions with minimum float.
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)  # type: ignore

        reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
        value = jnp.where(state.terminated, 0.0, value)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)

        epistemic_recurrent_fn_output = emctx.EpistemicRecurrentFnOutput(
            reward=reward,  # type: ignore
            reward_epistemic_variance=reward_epistemic_variance,  # type: ignore
            discount=discount,  # type: ignore
            prior_logits=logits,  # type: ignore
            value=value,  # type: ignore
            value_epistemic_variance=value_epistemic_variance,  # type: ignore
        )
        return epistemic_recurrent_fn_output, state

    return epistemic_recurrent_fn  # type: ignore


@partial(jax.pmap, static_broadcasted_argnums=[1, 2])
def selfplay(model, config: Config, context: Context, rng_key: chex.PRNGKey) -> SelfplayOutput:
    model_params, model_state = model
    batch_size = config.selfplay_batch_size // len(context.devices)

    def step_fn(states: pgx.State, key: chex.PRNGKey) -> tuple[pgx.State, SelfplayOutput]:
        key1, key2 = jax.random.split(key)

        (_exploitation_logits, exploration_logits, value, value_epistemic_variance, _reward_epistemic_variance), _ = (
            context.forward.apply(model_params, model_state, states.observation, is_training=False)
        )
        root = emctx.EpistemicRootFnOutput(
            prior_logits=exploration_logits,  # type: ignore
            value=value,  # type: ignore
            value_epistemic_variance=value_epistemic_variance,  # type: ignore
            embedding=states,  # type: ignore
            beta=config.beta * jnp.ones_like(value),  # type: ignore
        )
        policy_output = emctx.epistemic_gumbel_muzero_policy(
            params=model,
            rng_key=key1,
            root=root,
            recurrent_fn=context.exploration_recurrent_fn,
            num_simulations=config.simulations_per_step,
            invalid_actions=~states.legal_action_mask,
            qtransform=emctx.qtransform_completed_by_mix_value,
        )
        keys = jax.random.split(key2, batch_size)
        next_state = jax.vmap(auto_reset(context.env.step, context.env.init))(states, policy_output.action, keys)
        return next_state, states

    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    states = jax.vmap(context.env.init)(keys)
    key_seq = jax.random.split(rng_key, config.selfplay_steps)
    _, states = jax.lax.scan(step_fn, states, key_seq)

    return states


def mask_invalid_actions(logits: chex.Array, invalid_actions: chex.Array) -> chex.Array:
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


class ReanalyzeOutput(NamedTuple):
    observation: chex.Array
    next_observation: chex.Array  # for reward variance (local uncertainty)
    value_target: chex.Array
    ube_target: chex.Array
    exploration_policy_target: chex.Array
    exploitation_policy_target: chex.Array


@partial(jax.pmap, static_broadcasted_argnums=[1, 2])
def reanalyze(
    model,
    config: Config,
    context: Context,
    experience_pair: fbx.prioritised_flat_buffer.ExperiencePair,
    rng_key: chex.PRNGKey,
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
        recurrent_fn=context.exploitation_recurrent_fn,
        num_simulations=config.simulations_per_step,
        invalid_actions=invalid_actions,
        qtransform=emctx.qtransform_completed_by_mix_value,
    )
    search_summary = policy_output.search_tree.epistemic_summary()

    value_target = search_summary.qvalues[jnp.arange(search_summary.qvalues.shape[0]), policy_output.action]  # type: ignore
    ube_target = jnp.max(search_summary.qvalues_epistemic_variance, axis=1)

    completed_qvalues_epistemic_variance: chex.Array = mask_invalid_actions(
        search_summary.qvalues_epistemic_variance, invalid_actions
    )
    exploration_policy_target = jax.nn.softmax(
        completed_qvalues_epistemic_variance * config.exploration_policy_target_temperature
    )

    return ReanalyzeOutput(
        observation=observation,
        next_observation=next_states.observation,  # FIXME: This could be initial state from next episode
        value_target=value_target,
        ube_target=ube_target,
        exploitation_policy_target=policy_output.action_weights,
        exploration_policy_target=exploration_policy_target,
    )


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

    # TODO: Remove after testing
    # jax.debug.print("before {a}", a=reward_epistemic_variance)
    # (
    #     exploitation_logits,
    #     exploration_logits,
    #     value,
    #     value_epistemic_variance,
    #     reward_epistemic_variance,
    # ), model_state = context.forward.apply(
    #     model_params, model_state, reanalyze_output.observation, is_training=True, update_hash=True
    # )
    # jax.debug.print("after {a}", a=reward_epistemic_variance)

    value_loss = optax.l2_loss(value, reanalyze_output.value_target)
    value_loss = jnp.mean(value_loss)  # TODO: figure out if mask is needed because of episode truncation
    absolute_value_error = jnp.abs(value - reanalyze_output.value_target)

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

    total_loss = value_loss + ube_loss + exploitation_policy_loss + exploration_policy_loss

    return total_loss, (
        model_state,
        absolute_value_error,
        value_loss,
        ube_loss,
        exploitation_policy_loss,
        exploration_policy_loss,
    )


@partial(jax.pmap, axis_name="i", static_broadcasted_argnums=[2])
def train(model, opt_state, context: Context, reanalyze_output: ReanalyzeOutput):
    model_params, model_state = model
    grads, (model_state, absolute_value_error, *losses) = jax.grad(loss_fn, has_aux=True)(
        model_params, model_state, context, reanalyze_output
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = context.optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    model = (model_params, model_state)

    return (model, opt_state, absolute_value_error, *losses)


@partial(jax.pmap, static_broadcasted_argnums=[1, 2])
def evaluate(model, config: Config, context: Context, rng_key: chex.PRNGKey):
    model_params, model_state = model
    batch_size = config.selfplay_batch_size // len(context.devices)

    def cond_fn(tup: tuple[pgx.State, chex.PRNGKey, jax.Array]) -> bool:
        states, _, _ = tup
        return ~states.terminated.all()

    def loop_fn(tup: tuple[pgx.State, chex.PRNGKey, jax.Array]) -> tuple[pgx.State, chex.PRNGKey, jax.Array]:
        states, rng_key, sum_of_rewards = tup
        rng_key, key_for_search, key_for_next_step = jax.random.split(rng_key, 3)

        (exploitation_logits, _exploration_logits, value, value_epistemic_variance, _reward_epistemic_variance), _ = (
            context.forward.apply(model_params, model_state, states.observation, is_training=False)
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
            rng_key=key_for_search,
            root=root,
            recurrent_fn=context.exploitation_recurrent_fn,
            num_simulations=config.simulations_per_step,
            invalid_actions=~states.legal_action_mask,
            qtransform=emctx.qtransform_completed_by_mix_value,
        )
        keys = jax.random.split(key_for_next_step, batch_size)
        next_states = jax.vmap(context.env.step)(states, policy_output.action, keys)
        rewards = states.rewards[jnp.arange(states.rewards.shape[0]), states.current_player]
        return next_states, rng_key, sum_of_rewards + rewards

    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    states = jax.vmap(context.env.init)(keys)

    states, _, sum_of_rewards = jax.lax.while_loop(cond_fn, loop_fn, (states, rng_key, jnp.zeros(batch_size)))
    return sum_of_rewards.mean()


def main() -> None:
    # Get configuration from CLI.
    config_dict = omegaconf.OmegaConf.from_cli()
    config: Config = Config(**config_dict)  # type: ignore
    print(config)

    # Initialize Weights & Biases.
    wandb.init(project=WANDB_PROJECT, config=config.model_dump())

    # Identify devices.
    devices = jax.local_devices()
    num_devices = len(devices)

    # Make the environment.
    env = pgx.make(config.env_id)
    # baseline = pgx.make_baseline_model(config.env_id + "_v0")  # type: ignore

    # Initialize RNG key.
    rng_key = jax.random.PRNGKey(config.seed)
    rng_key, subkey_for_dummy_state, subkey_for_init = jax.random.split(rng_key, 3)

    # Initialize model.
    forward = get_forward_fn(env, config)
    dummy_state = jax.vmap(env.init)(jax.random.split(subkey_for_dummy_state, 2))
    dummy_input = dummy_state.observation
    model = forward.init(subkey_for_init, dummy_input)
    # Initialize optimizer.
    optimizer = optax.adam(learning_rate=config.learning_rate)
    opt_state = optimizer.init(params=model[0])
    # Replicate to all devices.
    model, opt_state = jax.device_put_replicated((model, opt_state), devices)

    # Initialize replay buffer.
    buffer_fn = fbx.make_prioritised_flat_buffer(
        max_length=config.max_replay_buffer_length,
        min_length=max(config.min_replay_buffer_length, config.reanalyze_batch_size),
        sample_batch_size=config.reanalyze_batch_size,
        add_sequences=True,
        add_batch_size=num_devices * config.selfplay_batch_size,
        priority_exponent=config.priority_exponent,
    )
    buffer_state = buffer_fn.init(jax.tree.map(lambda x: x[0], dummy_state))

    # Prepare checkpoint directory.
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    checkpoint_dir = os.path.join("checkpoints", f"{config.env_id}_{now}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize logging information and dictionary.
    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    log = {"iteration": iteration, "hours": hours, "frames": frames}
    start_time = time.time()

    context = Context(
        env=env,
        devices=devices,
        forward=forward,
        exploration_recurrent_fn=get_epistemic_recurrent_fn(env, forward, config.selfplay_batch_size, True),
        exploitation_recurrent_fn=get_epistemic_recurrent_fn(env, forward, config.reanalyze_batch_size, False),
        optimizer=optimizer,
    )

    # Training loop
    while True:
        print(log)
        wandb.log(log)
        log = {}

        if iteration % config.eval_interval == 0:
            # Evaluate network.
            rng_key, subkey = jax.random.split(rng_key)
            mean_return = evaluate(model, config, context, jax.random.split(subkey, num_devices))
            log.update({"mean_return": mean_return})

        if iteration % config.checkpoint_interval == 0:
            # Save checkpoint.
            model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
            with open(os.path.join(checkpoint_dir, f"{iteration:06d}.ckpt"), "wb") as f:
                dic = {
                    "config": config,
                    "rng_key": rng_key,
                    "model": jax.device_get(model_0),
                    "opt_state": jax.device_get(opt_state_0),
                    "iteration": iteration,
                    "frames": frames,
                    "hours": hours,
                    "env_id": env.id,
                    "env_version": env.version,
                }
                pickle.dump(dic, f)

        if iteration >= config.maximum_number_of_iterations:
            break
        iteration += 1

        # Selfplay.
        rng_key, subkey = jax.random.split(rng_key)
        states = selfplay(model, config, context, jax.random.split(subkey, num_devices))
        frames += num_devices * config.selfplay_batch_size * config.selfplay_steps

        # Add to buffer.
        # TODO: Check truncation (?)
        buffer_state = buffer_fn.add(
            buffer_state, jax.tree.map(lambda x: jnp.concatenate(jnp.swapaxes(x, 1, 2)), states)
        )

        value_loss_list = []
        ube_loss_list = []
        exploitation_policy_loss_list = []
        exploration_policy_loss_list = []
        for _ in range(config.reanalyze_loops_per_selfplay):
            assert buffer_fn.can_sample(buffer_state)

            # Reanalyze.
            rng_key, subkey = jax.random.split(rng_key)
            batch = buffer_fn.sample(buffer_state, subkey)
            reanalyze_output = reanalyze(
                model,
                config,
                context,
                jax.tree.map(lambda x: jnp.array(jnp.split(x, num_devices)), batch.experience),
                jax.random.split(subkey, num_devices),
            )
            (
                model,
                opt_state,
                absolute_value_error,
                value_loss,
                ube_loss,
                exploitation_policy_loss,
                exploration_policy_loss,
            ) = train(model, opt_state, context, reanalyze_output)
            absolute_value_error = jnp.concatenate(absolute_value_error)

            # Adjust priorities.
            buffer_state = buffer_fn.set_priorities(buffer_state, batch.indices, absolute_value_error)

            # Keep track of losses for logging.
            # `.mean()` because we get a separate loss per device.
            value_loss_list.append(value_loss.mean().item())
            ube_loss_list.append(ube_loss.mean().item())
            exploitation_policy_loss_list.append(exploitation_policy_loss.mean().item())
            exploration_policy_loss_list.append(exploration_policy_loss.mean().item())

            # TODO: Do we also want to log something per inner loop?

        # Calculate average losses for logging.
        average_value_loss = sum(value_loss_list) / len(value_loss_list)
        average_ube_loss = sum(ube_loss_list) / len(ube_loss_list)
        average_exploitation_policy_loss = sum(exploitation_policy_loss_list) / len(exploitation_policy_loss_list)
        average_exploration_policy_loss = sum(exploration_policy_loss_list) / len(exploration_policy_loss_list)

        log.update(
            {
                "iteration": iteration,
                "hours": (time.time() - start_time) / 3600,
                "frames": frames,
                "train/value_loss": average_value_loss,
                "train/ube_loss": average_ube_loss,
                "train/exploitation_policy_loss": average_exploitation_policy_loss,
                "train/exploration_policy_loss": average_exploration_policy_loss,
            }
        )


if __name__ == "__main__":
    main()
