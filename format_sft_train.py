"""
Format SFT for single-step memorization task.

Teaches the model to output numbers in the correct format (but random values,
not any specific secret). This is the "warmup" training that prepares the model
for RL fine-tuning.

Usage:
    python format_sft_train.py
    python format_sft_train.py n_values="(64, 256, 1024)"
    python format_sft_train.py preview_samples=True
"""

import asyncio
import json
import math
import random
from pathlib import Path
from typing import Iterable

import chz
import tinker
from tinker import types as tinker_types

from tinker_cookbook import checkpoint_utils, cli_utils, model_info, renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train as supervised_train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.tokenizer_utils import get_tokenizer

from envs.task_utils import build_single_step_user_prompt


DEFAULT_N_VALUES: tuple[int, ...] = (4, 16, 64, 256)


def _examples_per_n(n: int, min_examples: int, examples_per_value: int) -> int:
    return max(min_examples, examples_per_value * n)


def _write_format_dataset(
    path: Path,
    n_values: Iterable[int],
    min_examples: int,
    examples_per_value: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for n in n_values:
            total_examples = _examples_per_n(n, min_examples, examples_per_value)
            integers = list(range(n)) * ((total_examples + n - 1) // n)
            integers = integers[:total_examples]
            rng.shuffle(integers)

            prompt = build_single_step_user_prompt(n)
            for secret in integers:
                record = {
                    "split": "train",
                    "metadata": {"N": n},
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": str(int(secret))},
                    ],
                }
                f.write(json.dumps(record))
                f.write("\n")


def _entropy(values: list[int], n: int) -> float:
    if not values or n <= 1:
        return 0.0
    counts = [0] * n
    for v in values:
        if 0 <= v < n:
            counts[v] += 1
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy_bits = 0.0
    for c in counts:
        if c:
            p = c / total
            entropy_bits -= p * math.log2(p)
    return entropy_bits


@chz.chz
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B"
    log_path: str = "/tmp/tinker-examples/memory_rl/format_sft"

    # Data
    n_values: tuple[int, ...] = DEFAULT_N_VALUES
    min_examples: int = 800
    examples_per_value: int = 32
    data_seed: int = 1337
    max_length: int = 512

    # Training
    batch_size: int = 32
    num_epochs: int = 1
    learning_rate: float = 2e-5
    lr_schedule: str = "constant"
    base_url: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Sampling preview
    preview_samples: bool = False
    preview_temperature: float = 1.0
    num_preview_examples: int = 20

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def _build_dataset_builder(cli: Config, dataset_path: Path) -> FromConversationFileBuilder:
    renderer_name = model_info.get_recommended_renderer_name(cli.model_name)
    common_cfg = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli.model_name,
        renderer_name=renderer_name,
        batch_size=cli.batch_size,
        max_length=cli.max_length,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    return FromConversationFileBuilder(
        common_config=common_cfg,
        file_path=str(dataset_path),
        test_size=0,
        shuffle_seed=cli.data_seed,
    )


async def _preview_samples(cli: Config, sampler_path: str):
    renderer_name = model_info.get_recommended_renderer_name(cli.model_name)
    tokenizer = get_tokenizer(cli.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    service_client = tinker.ServiceClient(base_url=cli.base_url)
    sampling_client = service_client.create_sampling_client(model_path=sampler_path)

    sampling_params = tinker_types.SamplingParams(
        temperature=max(cli.preview_temperature, 1e-4),
        max_tokens=8,
        top_p=1.0,
        stop=renderer.get_stop_sequences(),
    )

    for n in cli.n_values:
        prompt = build_single_step_user_prompt(n)
        model_input = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
        outputs: list[int] = []
        valid = 0

        print(f"\nN={n}: sampling {cli.num_preview_examples} completions (temp={cli.preview_temperature})")
        for i in range(cli.num_preview_examples):
            response = await sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            tokens = response.sequences[0].tokens
            message = renderer.parse_response(tokens)[0]
            content = message["content"].strip()
            try:
                value = int(content)
            except ValueError:
                status = "not an integer"
                value = None
            else:
                in_range = 0 <= value < n
                status = "ok" if in_range else "out of range"
                if in_range:
                    valid += 1
                    outputs.append(value)
            print(f"  sample {i+1:02d}: {content} ({status})")

        entropy_bits = _entropy(outputs, n)
        max_entropy = math.log2(min(n, len(outputs))) if outputs else 0.0
        valid_pct = 100 * valid / cli.num_preview_examples if cli.num_preview_examples else 0.0
        print(
            f"  valid: {valid_pct:4.1f}% | unique: {len(set(outputs))}/{n} "
            f"| entropy: {entropy_bits:.3f}/{max_entropy:.3f} bits"
        )


async def run(cli: Config) -> None:
    """Run format SFT training. Called by xmux or CLI."""
    log_path = Path(cli.log_path)
    dataset_path = log_path / "format_train.jsonl"

    cli_utils.check_log_dir(str(log_path), behavior_if_exists=cli.behavior_if_log_dir_exists)
    _write_format_dataset(
        dataset_path,
        cli.n_values,
        cli.min_examples,
        cli.examples_per_value,
        cli.data_seed,
    )
    for n in cli.n_values:
        total_examples = _examples_per_n(n, cli.min_examples, cli.examples_per_value)
        min_per_value = total_examples // n
        print(f"N={n}: {total_examples} examples (>={min_per_value} per value)")

    dataset_builder = _build_dataset_builder(cli, dataset_path)
    training_config = supervised_train.Config(
        model_name=cli.model_name,
        log_path=str(log_path),
        dataset_builder=dataset_builder,
        learning_rate=cli.learning_rate,
        lr_schedule=cli.lr_schedule,
        num_epochs=cli.num_epochs,
        base_url=cli.base_url,
        wandb_project=cli.wandb_project,
        wandb_name=cli.wandb_name,
    )

    await supervised_train.main(training_config)
    checkpoint = checkpoint_utils.get_last_checkpoint(str(log_path))
    if checkpoint is None:
        raise RuntimeError("Training finished without a checkpoint.")
    print(f"\nCheckpoint saved: {checkpoint.get('state_path')}")
    print(f"Sampler path: {checkpoint.get('sampler_path')}")

    sampler_path = checkpoint.get("sampler_path")
    if cli.preview_samples and sampler_path:
        await _preview_samples(cli, sampler_path)
    elif cli.preview_samples:
        print("Sampler checkpoint missing; skipping preview.")


async def main() -> None:
    """CLI entrypoint."""
    cli = chz.entrypoint(Config)
    await run(cli)


if __name__ == "__main__":
    asyncio.run(main())
