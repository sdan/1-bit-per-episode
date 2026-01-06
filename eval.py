import asyncio
import math
from typing import Literal

import numpy as np
import tinker
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.renderers import TrainOnWhat

from envs import (
    build_multi_step_system_prompt,
    build_multi_step_user_prompt,
    build_single_step_user_prompt,
    compute_bits_known,
    format_secret_bits,
    num_bits_for_space,
    sum_weighted_logprobs,
)


class InfoTheoryEvaluator(SamplingClientEvaluator):
    """Logs constant, info-theoretic metadata for a run."""

    def __init__(
        self,
        *,
        env_type: Literal["single_step", "multi_step"],
        N: int,
        reward_type: str | None = None,
        reward_bins: int | None = None,
        metric_prefix: str = "theory",
    ):
        self._env_type = env_type
        self._N = int(N)
        self._reward_type = reward_type
        self._reward_bins = reward_bins
        self._metric_prefix = metric_prefix.rstrip("/")

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        del sampling_client

        signal_bits = math.log2(self._N) if self._N > 1 else 1.0

        if self._env_type == "multi_step":
            channel_bits = float(num_bits_for_space(self._N))
        elif self._reward_type == "binary":
            channel_bits = 1.0
        elif self._reward_type == "binned_log_distance" and self._reward_bins is not None:
            channel_bits = math.log2(int(self._reward_bins))
        else:
            channel_bits = float("nan")

        prefix = self._metric_prefix
        out = {
            f"{prefix}/signal_bits": float(signal_bits),
            f"{prefix}/channel_bits_per_episode": float(channel_bits),
        }
        return out


class BitsKnownEvaluator(SamplingClientEvaluator):
    """Evaluates bits of information the model has learned about the secret."""

    def __init__(
        self,
        dataset_builder,
        env_type: Literal["single_step", "multi_step"],
        metric_prefix: str = "bits",
    ):
        self._dataset_builder = dataset_builder
        self._env_type = env_type
        self._metric_prefix = metric_prefix.rstrip("/")
        self._dataset = None

    async def _ensure_dataset(self):
        if self._dataset is None:
            _, test_dataset = await self._dataset_builder()
            if test_dataset is None:
                raise ValueError("BitsKnownEvaluator requires a test dataset.")
            self._dataset = test_dataset
        return self._dataset

    def _get_eval_secrets(self, dataset) -> list[int]:
        if hasattr(dataset, "test_secrets"):
            return [int(x) for x in dataset.test_secrets]
        if getattr(dataset, "fixed_secret", None) is not None:
            total = len(dataset) * dataset.batch_size
            return [int(dataset.fixed_secret)] * total
        raise ValueError("Test dataset does not expose secrets for evaluation.")

    def _build_messages(self, N: int, secret: int, convo_prefix: list[dict] | None) -> list[dict]:
        prefix = list(convo_prefix or [])
        if self._env_type == "single_step":
            return prefix + [
                {"role": "user", "content": build_single_step_user_prompt(N)},
                {"role": "assistant", "content": str(secret)},
            ]

        num_bits = num_bits_for_space(N)
        secret_bits = format_secret_bits(secret, N)
        messages = [{"role": "system", "content": build_multi_step_system_prompt(N)}] + prefix
        for idx, bit in enumerate(secret_bits):
            messages.append({"role": "user", "content": build_multi_step_user_prompt(idx, num_bits)})
            messages.append({"role": "assistant", "content": bit})
        return messages

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        dataset = await self._ensure_dataset()
        renderer = dataset.renderer
        N = dataset.N
        train_on = (
            TrainOnWhat.LAST_ASSISTANT_MESSAGE
            if self._env_type == "single_step"
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        secrets = self._get_eval_secrets(dataset)

        # Build all inputs and weights upfront
        inputs_and_weights = []
        for secret in secrets:
            messages = self._build_messages(N, secret, dataset.convo_prefix)
            tokens, weights = renderer.build_supervised_example(messages, train_on_what=train_on)
            if hasattr(tokens, "to_ints"):
                token_list = tokens.to_ints()
            else:
                token_list = tokens.tolist()
            inputs_and_weights.append((tinker.ModelInput.from_ints(token_list), weights.tolist()))

        # Parallel logprob computation (like tinker_cookbook/rl/metrics.py)
        all_logprobs = await asyncio.gather(
            *[sampling_client.compute_logprobs_async(inp) for inp, _ in inputs_and_weights]
        )

        # Compute bits metrics from results
        bits_raw = []
        bits_clamped = []
        target_logprobs = []
        for logprobs, (_, weight_list) in zip(all_logprobs, inputs_and_weights):
            target_logprob, _ = sum_weighted_logprobs(logprobs, weight_list)
            bits_known_raw, bits_known_clamped = compute_bits_known(target_logprob, N)
            bits_raw.append(bits_known_raw)
            bits_clamped.append(bits_known_clamped)
            target_logprobs.append(target_logprob)

        bits_raw_arr = np.array(bits_raw, dtype=float)
        bits_clamped_arr = np.array(bits_clamped, dtype=float)
        target_logprob_arr = np.array(target_logprobs, dtype=float)

        prefix = self._metric_prefix
        return {
            f"{prefix}/known_raw_mean": float(bits_raw_arr.mean()),
            f"{prefix}/known_raw_std": float(bits_raw_arr.std()),
            f"{prefix}/known_clamped_mean": float(bits_clamped_arr.mean()),
            f"{prefix}/known_clamped_std": float(bits_clamped_arr.std()),
            f"{prefix}/target_logprob_mean": float(target_logprob_arr.mean()),
            f"{prefix}/target_logprob_std": float(target_logprob_arr.std()),
        }
