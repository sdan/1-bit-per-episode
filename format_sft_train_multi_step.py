"""
Format SFT for multi-step (bit-by-bit) memorization task.

Teaches the model to output single bits (0 or 1) in a conversational format,
preparing it for the multi-step RL environment.

Usage:
    python format_sft_train_multi_step.py
    python format_sft_train_multi_step.py n_values="(64, 256, 1024)"
    python format_sft_train_multi_step.py preview_samples=True
"""

import asyncio
import json
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

from envs.task_utils import format_secret_bits, num_bits_for_space


DEFAULT_N_VALUES: tuple[int, ...] = (4, 8, 16, 32, 64)


def _examples_per_n(n: int, min_examples: int, examples_per_value: int) -> int:
    """Calculate total number of examples for a given N."""
    return max(min_examples, examples_per_value * n)


def _build_bit_request(position: int, num_bits: int) -> str:
    """Build a user message asking for a specific bit."""
    remaining = num_bits - position - 1
    return f"Provide bit #{position + 1}. Remaining bits after this: {remaining}."


def _build_system_prompt(n: int, num_bits: int) -> str:
    """Build the system prompt for the multi-step task."""
    return (
        f"You are playing a bit-by-bit memorization game. "
        f"The secret number is between 0 and {n - 1} and uses {num_bits} bits. "
        f"Output exactly one bit (0 or 1) per response with no punctuation."
    )


def _write_format_dataset(
    path: Path,
    n_values: Iterable[int],
    min_examples: int,
    examples_per_value: int,
    seed: int,
) -> None:
    """
    Write a dataset of multi-step bit-by-bit conversations.

    Each example is a full conversation where the user asks for each bit
    and the assistant responds with the correct bit from a random secret.
    """
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for n in n_values:
            num_bits = num_bits_for_space(n)
            total_examples = _examples_per_n(n, min_examples, examples_per_value)

            # Generate secrets with deterministic coverage
            secrets = list(range(n)) * ((total_examples + n - 1) // n)
            secrets = secrets[:total_examples]
            rng.shuffle(secrets)

            system_prompt = _build_system_prompt(n, num_bits)

            for secret in secrets:
                secret_bits = format_secret_bits(secret, n)

                # Build a conversational example with all bits
                messages = [{"role": "system", "content": system_prompt}]

                for bit_idx in range(num_bits):
                    # User asks for bit
                    messages.append({
                        "role": "user",
                        "content": _build_bit_request(bit_idx, num_bits)
                    })
                    # Assistant responds with the correct bit
                    messages.append({
                        "role": "assistant",
                        "content": secret_bits[bit_idx]
                    })

                record = {
                    "split": "train",
                    "metadata": {
                        "N": n,
                        "secret": secret,
                        "secret_bits": secret_bits,
                        "num_bits": num_bits,
                    },
                    "messages": messages,
                }
                f.write(json.dumps(record))
                f.write("\n")


@chz.chz
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B"
    log_path: str = "/tmp/tinker-examples/memory_rl/format_sft_multi_step"

    # Data
    n_values: tuple[int, ...] = DEFAULT_N_VALUES
    min_examples: int = 500
    examples_per_value: int = 20
    data_seed: int = 1337
    max_length: int = 512

    # Training
    batch_size: int = 16
    num_epochs: int = 1
    learning_rate: float = 2e-5
    lr_schedule: str = "constant"
    base_url: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Sampling preview
    preview_samples: bool = False
    preview_temperature: float = 0.1
    num_preview_examples: int = 10

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def _build_dataset_builder(cli: Config, dataset_path: Path) -> FromConversationFileBuilder:
    """Build the dataset builder for supervised training."""
    renderer_name = model_info.get_recommended_renderer_name(cli.model_name)
    common_cfg = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli.model_name,
        renderer_name=renderer_name,
        batch_size=cli.batch_size,
        max_length=cli.max_length,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,  # Train on all bit outputs
    )
    return FromConversationFileBuilder(
        common_config=common_cfg,
        file_path=str(dataset_path),
        test_size=0,
        shuffle_seed=cli.data_seed,
    )


async def _preview_samples(cli: Config, sampler_path: str):
    """
    Preview samples from the trained model to verify it learned the format.
    """
    renderer_name = model_info.get_recommended_renderer_name(cli.model_name)
    tokenizer = get_tokenizer(cli.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    service_client = tinker.ServiceClient(base_url=cli.base_url)
    sampling_client = service_client.create_sampling_client(model_path=sampler_path)

    sampling_params = tinker_types.SamplingParams(
        temperature=max(cli.preview_temperature, 1e-4),
        max_tokens=3,
        top_p=1.0,
        stop=renderer.get_stop_sequences(),
    )

    print("\n" + "=" * 80)
    print("PREVIEW: Testing multi-step bit format")
    print("=" * 80)

    for n in cli.n_values:
        num_bits = num_bits_for_space(n)
        system_prompt = _build_system_prompt(n, num_bits)

        print(f"\nN={n} ({num_bits} bits): sampling {cli.num_preview_examples} random secrets")

        valid_conversations = 0
        bit_accuracies = []

        rng = random.Random(42)

        for example_idx in range(cli.num_preview_examples):
            secret = rng.randint(0, n - 1)
            secret_bits = format_secret_bits(secret, n)

            conversation = [{"role": "system", "content": system_prompt}]
            predicted_bits = []
            all_valid = True

            for bit_idx in range(num_bits):
                conversation.append({
                    "role": "user",
                    "content": _build_bit_request(bit_idx, num_bits)
                })

                model_input = renderer.build_generation_prompt(conversation)
                response = await sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=sampling_params,
                )
                tokens = response.sequences[0].tokens
                message = renderer.parse_response(tokens)[0]
                predicted_bit = message["content"].strip()

                if predicted_bit not in ("0", "1"):
                    all_valid = False
                    predicted_bits.append("?")
                else:
                    predicted_bits.append(predicted_bit)

                conversation.append({
                    "role": "assistant",
                    "content": predicted_bit
                })

            correct_bits = sum(
                1 for pred, actual in zip(predicted_bits, secret_bits)
                if pred == actual
            )
            accuracy = correct_bits / num_bits if num_bits > 0 else 0
            bit_accuracies.append(accuracy)

            if all_valid:
                valid_conversations += 1

            predicted_str = "".join(predicted_bits)
            status = "ok" if predicted_str == secret_bits else "x"
            print(
                f"  {example_idx+1:2d}. secret={secret:3d} ({secret_bits}) "
                f"-> pred=({predicted_str}) {status} acc={accuracy:.2f}"
            )

        valid_pct = 100 * valid_conversations / cli.num_preview_examples
        avg_accuracy = sum(bit_accuracies) / len(bit_accuracies) if bit_accuracies else 0
        perfect = sum(1 for acc in bit_accuracies if acc == 1.0)

        print(
            f"  -> valid format: {valid_pct:5.1f}% | "
            f"avg bit accuracy: {avg_accuracy:.3f} | "
            f"perfect: {perfect}/{cli.num_preview_examples}"
        )

    print("=" * 80 + "\n")


async def run(cli: Config) -> None:
    """Run format SFT training. Called by xmux or CLI."""
    log_path = Path(cli.log_path)
    dataset_path = log_path / "format_train_multi_step.jsonl"

    cli_utils.check_log_dir(str(log_path), behavior_if_exists=cli.behavior_if_log_dir_exists)

    print("\n" + "=" * 80)
    print("Generating multi-step format SFT dataset")
    print("=" * 80)

    _write_format_dataset(
        dataset_path,
        cli.n_values,
        cli.min_examples,
        cli.examples_per_value,
        cli.data_seed,
    )

    for n in cli.n_values:
        num_bits = num_bits_for_space(n)
        total_examples = _examples_per_n(n, cli.min_examples, cli.examples_per_value)
        min_per_value = total_examples // n
        print(
            f"N={n:3d} ({num_bits} bits): {total_examples:4d} examples "
            f"(>={min_per_value} per value)"
        )
    print("=" * 80 + "\n")

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
