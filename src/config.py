import random
import time
from typing import Annotated, Literal

import pgx  # type: ignore
import pydantic


class Config(pydantic.BaseModel):
    """Hyperparameter configuration"""

    # general
    debug: bool = False  # If True, automatically loads much smaller hps to make debugging easier
    auto_seed: bool = True  # If True and seed == 0, seeds with a random seed
    seed: int = 0
    env_class: Literal["pgx", "custom"] = "pgx"
    env_id: pgx.EnvId | str = "minatar-breakout"
    maximum_number_of_iterations: int = 2000
    two_players_game: bool = False
    # network
    linear_layer_size: int = 128
    num_channels: int = 16
    # Local uncertainty parameters
    hash_class: Literal["SimHash", "LCGHash", "XXHash"] = "XXHash"
    hash_path: str = "minatar_az_net/xxhash32"
    max_epistemic_variance_reward: float = 1.0
    # UBE parameters
    value_scale: float = 1.0  # Approximately the maximum value for the environment, used to stabilize the val. function
    exploration_ube_target: bool = True     # If true, ube target is max_child_unc. Otherwise, it's chosen child's unc.
    # selfplay
    selfplay_batch_size: int = 128  # FIXME: Return these hyperparameters to normal numbers
    selfplay_simulations_per_step: int = 32
    selfplay_steps: int = 32
    directed_exploration: bool = False  # if true, betaExploration = 0 and uses exploitation policy in selfplay
    sample_actions: bool = False
    sample_from_improved_policy: bool = False
    rescale_q_values_in_search: bool = True
    uniform_search_policy: bool = False     # If True, search policy is always uniform in selfplay. Currently only implemented for root
    # reanalyze
    reanalyze_batch_size: int = 4096
    reanalyze_simulations_per_step: int = 32
    reanalyze_loops_per_selfplay: int | None = (
        None  # computes as training_to_interactions_ratio * reanalyze_data / selfplay_data
    )
    training_to_interactions_ratio: Annotated[int, pydantic.Field(strict=True, ge=2)] = (
        8  # The number of datapoints to see in training compared to acting. Must be >= 2, or only trains on fresh data
    )
    max_replay_buffer_length: int = 1_000_000
    min_replay_buffer_length: int = 256
    priority_exponent: float = 0.6
    # training
    learning_rate: float = 0.001
    learning_starts: int = int(5e3)  # While buffer size < learning_starts, executes random actions
    scale_uncertainty_losses: float = 1.0  # Scales the exploration policy and ube head to reduce influence on body
    weigh_losses: bool = False      # If true, weighs losses with epistemic uncertainty
    loss_weighting_temperature: float = 10.0    # From Sunrise https://arxiv.org/pdf/2007.04938
    # checkpoints / eval
    checkpoint_interval: int = 5
    eval_interval: int = 5
    evaluation_batch: int = 128
    # targets
    exploration_policy_target_temperature: float = 1.0
    discount: float = 0.997
    # EMCTS exploration parameters
    exploration_beta: Annotated[float, pydantic.Field(strict=True, ge=0.0)] = (
        0.0  # used in selfplay in emctx for directed exploration
    )
    exploitation_beta: Annotated[float, pydantic.Field(strict=True, le=0.0)] = (
        0.0  # used in evaluation in emctx
    )
    reanalyze_beta: Annotated[float, pydantic.Field(strict=True, le=0.0)] = (
        0.0  # used in reanalyze in emctx for epistemically reliable targets
    )
    beta_schedule: bool = False  # If true, betas for each game are evenly spaced between 0 and beta. Not yet imped.
    # Evaluation
    num_eval_episodes: int = 32
    evaluation_episode_max_length: int = 2500   # May want to change this per env
    # wandb params
    track: bool = True  # Whether to use WANDB or not. Disabled in debug
    wandb_project: str = "e-alphazero"
    wandb_run_name: str | None = None
    wandb_team_name: str = "emcts"
    # slurm info
    slurm_job_id: int | None = None
    # Offline-RL
    save_replay_buffer: bool = False
    replay_buffer_path: str | None = None   # Must be of structure: path/vault_name/vault_uid

    class Config:
        extra = "forbid"

    # HACK: Should be fine since there will only ever be one `Config`.
    def __hash__(self):
        return 0

    def __str__(self):
        return '\n'.join([f'{key}: {value}' for key, value in self.dict().items()])


def setup_config(config: Config) -> Config:
    if config.debug:
        config.env_class = "custom"
        config.env_id = "deep_sea-16"
        config.selfplay_batch_size = 8
        config.selfplay_simulations_per_step = 16
        config.reanalyze_simulations_per_step = 16
        config.selfplay_steps = 8
        config.reanalyze_batch_size = 256
        config.max_replay_buffer_length = 100_000
        config.min_replay_buffer_length = 64
        config.learning_starts = 256
        config.hash_class = "XXHash"
        config.track = False
        config.eval_interval = 1
        config.directed_exploration = True
        config.exploration_beta = 10.0
        config.maximum_number_of_iterations = max(
            50, int(config.learning_starts / (config.selfplay_steps * config.selfplay_batch_size) + 3)
        )
        config.directed_exploration = True

    if "deep_sea" in config.env_id:
        config.selfplay_steps = 8
        config.selfplay_batch_size = 16

    # Update config with runtime computed values
    if config.auto_seed and config.seed == 0:
        config.seed = random.randint(1, 100000)
    if config.wandb_run_name is None:
        config.wandb_run_name = (
            f"{config.env_id}_beta={config.exploration_beta}_{config.seed}"
            f"_{time.asctime(time.localtime(time.time()))}"
        )
    config.two_players_game = config.env_class == "pgx" and not "minatar" in config.env_id
    config.hash_path = "minatar_az_net/" if "minatar" in config.env_id else "fc_az_net/"
    config.hash_path += "sim_hash" if config.hash_class == "SimHash" else "xxhash32"
    if "minatar" in config.env_id:
        if "breakout" in config.env_id:
            config.value_scale = 40
        elif "space_invaders" in config.env_id:
            config.value_scale = 200
        elif "freeway" in config.env_id:
            config.value_scale = 60
        elif "asterix" in config.env_id:
            config.value_scale = 25
        elif "seaquest" in config.env_id:
            config.value_scale = 60
        else:
            raise ValueError(f"Unrecognized minatar environment. env_id was {config.env_id}")
    else:
        config.value_scale = 1.0

    config.reanalyze_loops_per_selfplay = max(
        1,
        int(
            config.training_to_interactions_ratio
            * config.selfplay_steps
            * config.selfplay_batch_size
            / config.reanalyze_batch_size
        ),
    )

    config.exploration_beta = config.exploration_beta if config.directed_exploration else 0.0
    # Make sure min replay buffer length makes sense
    if config.min_replay_buffer_length < config.reanalyze_batch_size * config.reanalyze_loops_per_selfplay:
        config.min_replay_buffer_length = config.reanalyze_batch_size * config.reanalyze_loops_per_selfplay

    assert config.min_replay_buffer_length < config.max_replay_buffer_length, (
        f"max_replay_buffer_length must be > "
        f"min_replay_buffer_length and isn't. \n"
        f"max_replay_buffer_length = "
        f"{config.min_replay_buffer_length}, and "
        f"min_replay_buffer_length = {config.min_replay_buffer_length}, "
    )
    return config
