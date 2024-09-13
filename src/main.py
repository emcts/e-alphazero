import datetime
import os
import pickle
import sys
import time
from functools import partial
from typing import Any

import flashbax as fbx  # type: ignore
import haiku as hk
import jax
import jax.numpy as jnp
import omegaconf
import optax  # type: ignore
import pgx  # type: ignore
import wandb
from pgx.experimental import auto_reset  # type: ignore
from flashbax.vault import Vault

from config import Config, setup_config
from context import Context, get_epistemic_recurrent_fn, get_forward_fn
from envs.deep_sea import DeepSea
from evaluate import evaluate
from reanalyze import reanalyze
from selfplay import selfplay, uniformrandomplay
from train import train


@partial(jax.pmap, static_broadcasted_argnums=[2])
def debug_deep_sea(all_states_batch, model, context):
    # print(f"all_states_batch.shape = {all_states_batch.shape}")
    model_params, model_state = model

    (_exploitation_logits, _exploration_logits, _value, value_epistemic_variance, reward_epistemic_variance), _ = (
        context.forward.apply(model_params, model_state, all_states_batch, is_training=True)
    )
    ube_predictions = jnp.reshape(
        value_epistemic_variance, shape=(all_states_batch.shape[-1], all_states_batch.shape[-1])
    )
    unseen_states = jnp.reshape(
        reward_epistemic_variance, shape=(all_states_batch.shape[-1], all_states_batch.shape[-1])
    )
    return ube_predictions, unseen_states


def main() -> None:
    # Get configuration from CLI.
    config_dict = omegaconf.OmegaConf.from_cli()
    config: Config = Config(**config_dict)  # type: ignore
    config: Config = setup_config(config)

    if config.env_class == "custom" and "deep_sea" in config.env_id:
        s = config.env_id.removeprefix("deep_sea-")
        size_of_grid = int(s) if s.isnumeric() else 4
        all_states_batch = jnp.zeros([size_of_grid * size_of_grid, size_of_grid, size_of_grid], dtype=jnp.bool)
        # all_states_batch = jnp.zeros([config.selfplay_batch_size, size_of_grid, size_of_grid], dtype=jnp.bool)
        # for each row
        for i in range(size_of_grid):
            # for each column
            for j in range(size_of_grid):
                all_states_batch = all_states_batch.at[i * size_of_grid + j, i, j].set(True)
        all_states_batch = jnp.expand_dims(all_states_batch, axis=0)

    print(f"Printing the config:\n{config}", flush=True)

    # Initialize Weights & Biases.
    if config.track:
        wandb.init(
            project=config.wandb_project,
            config=config.model_dump(),
            name=config.wandb_run_name,
            entity=config.wandb_team_name,
        )

    # Identify devices.
    devices = jax.local_devices()
    num_devices = len(devices)

    # Initialize RNG key.
    rng_key = jax.random.PRNGKey(config.seed)
    rng_key, subkey_for_env, subkey_for_dummy_state, subkey_for_init = jax.random.split(rng_key, 4)

    # Make the environment.
    env: pgx.Env
    match (config.env_class, config.env_id):
        case ("custom", str(s)) if s.startswith("deep_sea"):
            # E.g. For DeepSea size 16, use "deep_sea-16".
            s = s.removeprefix("deep_sea-")
            size_of_grid = int(s) if s.isnumeric() else 4
            env = DeepSea(size_of_grid=size_of_grid, action_map_key=subkey_for_env)
        case ("pgx", env_id) if env_id in pgx.available_envs():
            env = pgx.make(env_id)
        case (cl, id):
            assert False, f"Invalid environment settings: {cl}, {id}."
    # selfplay_env, planner_env, eval_env = make_envs(config.env_class, config.env_id)
    # baseline = pgx.make_baseline_model(config.env_id + "_v0")  # type: ignore

    # Initialize model.
    forward = get_forward_fn(env, config)
    dummy_state = jax.vmap(env.init)(jax.random.split(subkey_for_dummy_state, 2))
    dummy_input = dummy_state.observation
    model: tuple[hk.MutableParams, hk.MutableState] = forward.init(subkey_for_init, dummy_input)
    # Initialize optimizer.
    optimizer = optax.adam(learning_rate=config.learning_rate)
    opt_state: optax.OptState = optimizer.init(params=model[0])
    # Replicate to all devices.
    model, opt_state = jax.device_put_replicated((model, opt_state), devices)

    # Initialize replay buffer.
    # FIXME: UserWarning: Setting max_size dynamically sets the `max_length_time_axis` to be `max_size`//`add_batch_size = 3125`.
    # This allows one to control exactly how many timesteps are stored in the buffer.Note that this overrides the `max_length_time_axis` argument.
    buffer_fn = fbx.make_prioritised_flat_buffer(
        max_length=config.max_replay_buffer_length,
        min_length=max(config.min_replay_buffer_length, config.reanalyze_batch_size),
        sample_batch_size=config.reanalyze_batch_size,
        add_sequences=True,
        add_batch_size=config.selfplay_batch_size,
        priority_exponent=config.priority_exponent,
    )
    buffer_state = buffer_fn.init(jax.tree.map(lambda x: x[0], dummy_state))

    if config.save_replay_buffer:
        parts = config.replay_buffer_path.split('/')
        vault_uid = parts[-1]
        vault_name = parts[-2]
        rel_dir = '/'.join(parts[:-2]) + '/' if len(parts) > 2 else ''
        rb_vault = Vault(
            vault_uid=vault_uid,
            vault_name=vault_name,
            experience_structure=buffer_state.experience,
            rel_dir=rel_dir
        )

    # Prepare checkpoint directory.
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    checkpoint_dir = os.path.join("checkpoints", f"{config.env_id}_{now}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize logging information and dictionary.
    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    set_bits: int = 0
    log: dict[str, Any] = {"iteration": iteration, "hours": hours, "frames": frames}
    last_states = None
    start_time = time.time()

    context = Context(
        env=env,
        devices=devices,
        forward=forward,
        selfplay_recurrent_fn=get_epistemic_recurrent_fn(
            env=env,
            forward=forward,
            batch_size=config.selfplay_batch_size,
            exploration=config.directed_exploration,
            discount=config.discount,
            two_players_game=config.two_players_game,
        ),
        reanalyze_recurrent_fn=get_epistemic_recurrent_fn(
            env=env,
            forward=forward,
            batch_size=config.reanalyze_batch_size,
            exploration=False,
            discount=config.discount,
            two_players_game=config.two_players_game,
        ),
        evaluation_recurrent_fn=get_epistemic_recurrent_fn(
            env=env,
            forward=forward,
            batch_size=config.num_eval_episodes,
            exploration=False,
            discount=config.discount,
            two_players_game=config.two_players_game,
        ),
        optimizer=optimizer,
        scale_uncertainty_losses=config.scale_uncertainty_losses,
        hash_path=config.hash_path,  # TODO: Automatically figure this out
        exploration_beta=config.exploration_beta,
        max_ube=config.max_ube,
    )

    # Training loop
    while True:
        print(log)
        if config.track:
            wandb.log(log)
        log = {}

        if iteration % config.eval_interval == 0:
            # Evaluate network.
            if config.exploitation_beta < 0:
                # Do regular evaluation
                original_exploitation_beta = config.exploitation_beta
                config.exploitation_beta = 0.0
                rng_key, subkey = jax.random.split(rng_key)
                mean_return = evaluate(model, config, context, jax.random.split(subkey, num_devices))
                log.update({"regular mean_return": mean_return.item()})
                # And then do pessim_evaluation
                config.exploitation_beta = original_exploitation_beta
                rng_key, subkey = jax.random.split(rng_key)
                mean_return = evaluate(model, config, context, jax.random.split(subkey, num_devices))
                log.update({"pessimistic_evaluation mean_return": mean_return.item()})
            else:
                rng_key, subkey = jax.random.split(rng_key)
                mean_return = evaluate(model, config, context, jax.random.split(subkey, num_devices))
                log.update({"mean_return": mean_return.item()})
            sys.stdout.flush()

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
        if frames < config.learning_starts:
            last_states, states = uniformrandomplay(config, context, jax.random.split(subkey, num_devices))
        else:
            last_states, (states, root_values, root_epistemic_stds, raw_values, ube_predictions, q_value_variances) = (
                selfplay(model, config, context, last_states, jax.random.split(subkey, num_devices))
            )
            log.update(
                {
                    "mean_raw_value": raw_values.mean().item(),
                    "mean_root_value": root_values.mean().item(),
                    "mean_ube": ube_predictions.mean().item(),
                    "mean_root_epistemic_std": root_epistemic_stds.mean().item(),
                    "mean_root_max_child_epistemic_variance": q_value_variances.max(axis=-1).mean().item(),
                }
            )

            if "deep_sea" in config.env_id:
                ube_predictions, unseen_states = debug_deep_sea(all_states_batch, model, context)  # type: ignore
                print(f"ube_predictions = \n" f"{ube_predictions}")
                print(f"unseen_states = \n" f"{unseen_states}")
                print(f"Number of seen states: = {(1 - unseen_states).sum().item()}")

        frame_diff = config.selfplay_batch_size * config.selfplay_steps
        frames += frame_diff

        # Add to buffer.
        # TODO: Check truncation (?)
        buffer_state = buffer_fn.add(
            buffer_state, jax.tree.map(lambda x: jnp.concatenate(jnp.swapaxes(x, 1, 2)), states)
        )
        if config.save_replay_buffer:
            rb_vault.write(buffer_state)

        value_loss_list = []
        ube_loss_list = []
        exploitation_policy_loss_list = []
        exploration_policy_loss_list = []
        exploitation_policy_entropy_list = []
        exploration_policy_entropy_list = []
        batch_novelty_list = []

        # If the buffer doesn't have enough interactions yet, keep interacting, or learning shouldn't start yet
        if not buffer_fn.can_sample(buffer_state) or frames < config.learning_starts:
            log.update(
                {
                    "iteration": iteration,
                    "hours": (time.time() - start_time) / 3600,
                    "frames": frames,
                    "executing random moves until": max(config.learning_starts, config.min_replay_buffer_length),
                }
            )
            continue
        else:
            for _ in range(config.reanalyze_loops_per_selfplay):  # type: ignore
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
                print(f"Max UBE target from reanalyze = {reanalyze_output.ube_target.max().item()}")
                (
                    model,
                    opt_state,
                    priority_score,
                    exploitation_policy_entropy,
                    exploration_policy_entropy,
                    value_loss,
                    ube_loss,
                    exploitation_policy_loss,
                    exploration_policy_loss,
                    batch_novelty,
                ) = train(model, opt_state, context, reanalyze_output)
                priority_score = jnp.concatenate(priority_score)

                # Adjust priorities.
                buffer_state = buffer_fn.set_priorities(buffer_state, batch.indices, priority_score)

                # Keep track of losses for logging.
                # `.mean()` because we get a separate loss per device.
                value_loss_list.append(value_loss.mean().item())
                ube_loss_list.append(ube_loss.mean().item())
                exploitation_policy_loss_list.append(exploitation_policy_loss.mean().item())
                exploration_policy_loss_list.append(exploration_policy_loss.mean().item())
                exploitation_policy_entropy_list.append(exploitation_policy_entropy.mean().item())
                exploration_policy_entropy_list.append(exploration_policy_entropy.mean().item())
                batch_novelty_list.append(batch_novelty.mean().item())

            # Calculate "replay buffer uniqueness" (according to hash set).
            _model_params, model_state = model
            binary_set = model_state[context.hash_path]["binary_set"][0]  # index 0 because of device dimension
            new_set_bits = jax.lax.population_count(binary_set).sum().item()  # i.e. "seen" states
            set_bit_diff = new_set_bits - set_bits
            set_bits = new_set_bits

            log.update(
                {
                    "iteration": iteration,
                    "hours": (time.time() - start_time) / 3600,
                    "frames": frames,
                    "train/value_loss": sum(value_loss_list) / len(value_loss_list),
                    "train/ube_loss": sum(ube_loss_list) / len(ube_loss_list),
                    "train/exploitation_policy_loss": sum(exploitation_policy_loss_list)
                    / len(exploitation_policy_loss_list),
                    "train/exploration_policy_loss": sum(exploration_policy_loss_list)
                    / len(exploration_policy_loss_list),
                    "train/mean_exploitation_policy_entropy": sum(exploitation_policy_entropy_list)
                    / len(exploitation_policy_entropy_list),
                    "train/mean_exploration_policy_entropy": sum(exploration_policy_entropy_list)
                    / len(exploration_policy_entropy_list),
                    "hash/set_bits": set_bits,
                    "hash/new_bits_ratio": set_bit_diff / frame_diff,
                    "hash/batch_novelty": sum(batch_novelty_list) / len(batch_novelty_list),
                }
            )


if __name__ == "__main__":
    main()
