import asyncio
import logging
import math
from datetime import datetime
from typing import Literal

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train

# Use local envs/ module instead of tinker_cookbook.recipes.memory_rl
from envs import (
    MultiStepDatasetBuilder,
    SingleStepDatasetBuilder,
    num_bits_for_space,
    validate_reward_config,
)
from eval import BitsKnownEvaluatorBuilder, InfoTheoryEvaluator

logger = logging.getLogger(__name__)


SINGLE_STEP_CHECKPOINT = "tinker://61ffdf2c-c9ae-52f1-8b0a-d5757c68bee8:train:0/weights/final"
MULTI_STEP_CHECKPOINT = "tinker://1e79325e-97ad-5cfc-aae3-fdc7b5951746:train:0/weights/final"


@chz.chz
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B"
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    N: int = 16
    reward_type: str = "binary"
    reward_bins: int | None = None
    fixed_secret: int | None = None
    resample_secret: bool = False
    use_standard_prefix: bool = False
    env_type: Literal["single_step", "multi_step"] = "single_step"

    batch_size: int = 1
    group_size: int = 4
    n_batches: int = 1000
    test_n_batches: int | None = None
    lora_rank: int = 1
    learning_rate: float = 4e-5
    max_tokens: int = 8
    eval_every: int = 10
    save_every: int = 20
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    dataset_seed: int = 0
    eval_bits: bool = True
    temperature: float = 1.0
    remove_constant_reward_groups: bool = False

    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None


def build_config(cli: Config) -> train.Config:
    renderer_name = cli.renderer_name or model_info.get_recommended_renderer_name(cli.model_name)

    load_checkpoint_path = cli.load_checkpoint_path
    if load_checkpoint_path is None:
        load_checkpoint_path = (
            SINGLE_STEP_CHECKPOINT if cli.env_type == "single_step" else MULTI_STEP_CHECKPOINT
        )

    if cli.env_type == "single_step":
        validate_reward_config(cli.reward_type, cli.reward_bins)

    prefix = "standard" if cli.use_standard_prefix else None

    if cli.batch_size == 1 and cli.group_size == 1:
        raise ValueError(
            "batch_size=1 and group_size=1 produces a zero-advantage batch (no learning). "
            "Use group_size>=2 (recommended) or batch_size>=2."
        )

    if cli.env_type == "single_step":
        builder = SingleStepDatasetBuilder(
            batch_size=cli.batch_size,
            group_size=cli.group_size,
            renderer_name=renderer_name,
            model_name_for_tokenizer=cli.model_name,
            N=cli.N,
            reward_type=cli.reward_type,
            reward_bins=cli.reward_bins,
            n_batches=cli.n_batches,
            test_n_batches=cli.test_n_batches,
            fixed_secret=cli.fixed_secret,
            resample_secret=cli.resample_secret,
            convo_prefix=prefix,
            seed=cli.dataset_seed,
        )
    elif cli.env_type == "multi_step":
        builder = MultiStepDatasetBuilder(
            batch_size=cli.batch_size,
            group_size=cli.group_size,
            renderer_name=renderer_name,
            model_name_for_tokenizer=cli.model_name,
            N=cli.N,
            n_batches=cli.n_batches,
            test_n_batches=cli.test_n_batches,
            fixed_secret=cli.fixed_secret,
            resample_secret=cli.resample_secret,
            convo_prefix=prefix,
            seed=cli.dataset_seed,
        )
    else:
        raise ValueError(f"Unknown env_type {cli.env_type}")

    date_and_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    reward_suffix = f"{cli.reward_type}_B{cli.reward_bins}" if cli.reward_bins else cli.reward_type
    default_run_name = (
        f"{cli.env_type}_{reward_suffix}_N{cli.N}_bs{cli.batch_size}_gs{cli.group_size}_lr{cli.learning_rate}_{date_and_time}"
    )
    log_path = cli.log_path or f"/tmp/tinker-examples/memory_rl/{default_run_name}"
    wandb_name = cli.wandb_name or default_run_name

    num_bits = num_bits_for_space(cli.N)

    evaluator_builders = []
    evaluator_builders.append(
        lambda env_type=cli.env_type,
        N=cli.N,
        reward_type=cli.reward_type,
        reward_bins=cli.reward_bins: InfoTheoryEvaluator(
            env_type=env_type,
            N=N,
            reward_type=reward_type,
            reward_bins=reward_bins,
            metric_prefix="theory",
        )
    )
    if cli.eval_bits:
        # Use the builder pattern - evaluator is created once and caches inputs
        # episodes_per_eval = eval_every batches * batch_size problems * group_size episodes
        episodes_per_eval = cli.eval_every * cli.batch_size * cli.group_size
        bits_builder = BitsKnownEvaluatorBuilder(
            dataset_builder=builder,
            env_type=cli.env_type,
            metric_prefix="test/bits",
            episodes_per_eval=episodes_per_eval,
        )
        evaluator_builders.append(bits_builder)

    return train.Config(
        model_name=cli.model_name,
        load_checkpoint_path=load_checkpoint_path,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=cli.learning_rate,
        lora_rank=cli.lora_rank,
        max_tokens=cli.max_tokens,
        temperature=cli.temperature,
        eval_every=cli.eval_every,
        save_every=cli.save_every,
        loss_fn=cli.loss_fn,
        remove_constant_reward_groups=cli.remove_constant_reward_groups,
        wandb_project=cli.wandb_project,
        wandb_name=wandb_name,
        evaluator_builders=evaluator_builders,
    )


async def run(cli: Config) -> None:
    """Run training with the given config. Called by xmux or CLI."""
    config = build_config(cli)

    entropy_bits = math.log2(cli.N) if cli.N > 0 else 0
    num_bits = num_bits_for_space(cli.N)
    if cli.env_type == "single_step":
        if cli.reward_type == "binary":
            channel_desc = "binary (1 bit max)"
        elif cli.reward_type == "log_distance":
            channel_desc = "log_distance (continuous)"
        elif cli.reward_type == "binned_log_distance":
            channel_desc = f"binned_log_distance (B={cli.reward_bins})"
        else:
            raise ValueError(f"Unknown reward_type: {cli.reward_type}")
    else:
        channel_desc = f"{num_bits} bits (1 per step)"

    logger.info(f"Environment: {cli.env_type}, N={cli.N} ({entropy_bits:.2f} bits)")
    logger.info(f"Channel: {channel_desc}")
    if cli.fixed_secret is not None:
        secret_desc = str(cli.fixed_secret)
    elif cli.resample_secret:
        secret_desc = "Random (resampled)"
    else:
        secret_desc = "Random (fixed per run)"
    logger.info(f"Secret: {secret_desc}")
    logger.info(f"Init checkpoint: {config.load_checkpoint_path or 'Base model'}")
    logger.info(f"Batches: {cli.n_batches} × {cli.batch_size} × {cli.group_size}")
    logger.info(f"Log path: {config.log_path}")

    cli_utils.check_log_dir(config.log_path, behavior_if_exists="resume")
    await train.main(config)


async def main() -> None:
    """CLI entrypoint."""
    cli = chz.entrypoint(Config)
    await run(cli)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
