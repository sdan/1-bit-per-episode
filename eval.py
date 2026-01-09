import asyncio
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tinker
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.renderers import Renderer, TrainOnWhat

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
    """Logs constant, info-theoretic metadata for a run.

    This evaluator returns static values computed at init time.
    No actual model inference is performed.
    """

    def __init__(
        self,
        *,
        env_type: Literal["single_step", "multi_step"],
        N: int,
        reward_type: str | None = None,
        reward_bins: int | None = None,
        metric_prefix: str = "theory",
    ):
        self._metric_prefix = metric_prefix.rstrip("/")

        # Precompute all metrics at init time
        signal_bits = math.log2(N) if N > 1 else 1.0

        if env_type == "multi_step":
            channel_bits = float(num_bits_for_space(N))
        elif reward_type == "binary":
            channel_bits = 1.0
        elif reward_type == "binned_log_distance" and reward_bins is not None:
            channel_bits = math.log2(int(reward_bins))
        else:
            channel_bits = float("nan")

        self._metrics = {
            f"{self._metric_prefix}/signal_bits": float(signal_bits),
            f"{self._metric_prefix}/channel_bits_per_episode": float(channel_bits),
        }

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        del sampling_client
        return self._metrics


@dataclass(frozen=True)
class _EvalInput:
    """Precomputed input for a single logprob evaluation."""

    model_input: tinker.ModelInput
    weights: list[float]
    secret: int


class BitsKnownEvaluator(SamplingClientEvaluator):
    """Evaluates bits of information the model has learned about the secret.

    Following master-tinker's pattern: precompute all inputs at init time,
    only do model inference in __call__.

    For fixed-secret experiments, we only need ONE logprob computation since
    the result is deterministic. For varying secrets, we compute once per
    unique secret.
    """

    def __init__(
        self,
        *,
        renderer: Renderer,
        N: int,
        env_type: Literal["single_step", "multi_step"],
        secrets: list[int],
        convo_prefix: list[dict] | None = None,
        metric_prefix: str = "bits",
    ):
        self._N = N
        self._metric_prefix = metric_prefix.rstrip("/")

        train_on = (
            TrainOnWhat.LAST_ASSISTANT_MESSAGE
            if env_type == "single_step"
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        # Deduplicate secrets - no need to compute same logprobs multiple times
        unique_secrets = list(set(secrets))

        # Precompute all eval inputs at init time
        self._eval_inputs: list[_EvalInput] = []
        for secret in unique_secrets:
            messages = _build_messages(env_type, N, secret, convo_prefix)
            tokens, weights = renderer.build_supervised_example(
                messages, train_on_what=train_on
            )

            if hasattr(tokens, "to_ints"):
                token_list = tokens.to_ints()
            else:
                token_list = tokens.tolist()

            self._eval_inputs.append(
                _EvalInput(
                    model_input=tinker.ModelInput.from_ints(token_list),
                    weights=weights.tolist(),
                    secret=secret,
                )
            )

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """Compute bits_known metrics using precomputed inputs."""

        # Parallel logprob computation - submit all requests before awaiting any
        # This maximizes clock cycle efficiency per tinker docs
        all_logprobs = await asyncio.gather(*[
            sampling_client.compute_logprobs_async(inp.model_input)
            for inp in self._eval_inputs
        ])

        # Compute bits known for each secret
        bits_clamped = []
        for logprobs, eval_input in zip(all_logprobs, self._eval_inputs):
            target_logprob, _ = sum_weighted_logprobs(logprobs, eval_input.weights)
            _, bits_known_clamped = compute_bits_known(target_logprob, self._N)
            bits_clamped.append(bits_known_clamped)

        bits_clamped_arr = np.array(bits_clamped, dtype=float)

        prefix = self._metric_prefix
        return {
            f"{prefix}/known": float(bits_clamped_arr.mean()),
        }

    @classmethod
    def from_dataset(
        cls,
        dataset,
        env_type: Literal["single_step", "multi_step"],
        metric_prefix: str = "bits",
    ) -> "BitsKnownEvaluator":
        """Create evaluator from a dataset (convenience factory method).

        This is the preferred way to create BitsKnownEvaluator when you have
        access to the test dataset.
        """
        # Extract secrets from dataset
        if hasattr(dataset, "test_secrets"):
            secrets = [int(x) for x in dataset.test_secrets]
        elif getattr(dataset, "fixed_secret", None) is not None:
            secrets = [int(dataset.fixed_secret)]
        else:
            raise ValueError("Dataset must expose test_secrets or fixed_secret.")

        return cls(
            renderer=dataset.renderer,
            N=dataset.N,
            env_type=env_type,
            secrets=secrets,
            convo_prefix=getattr(dataset, "convo_prefix", None),
            metric_prefix=metric_prefix,
        )


class BitsKnownEvaluatorBuilder:
    """Builder that creates BitsKnownEvaluator from a dataset builder.

    This handles the async dataset creation and caches the evaluator after
    first construction. Use this when you need to pass an evaluator builder
    to train.Config but don't have the dataset yet.
    """

    def __init__(
        self,
        dataset_builder,
        env_type: Literal["single_step", "multi_step"],
        metric_prefix: str = "bits",
        episodes_per_eval: int | None = None,
    ):
        self._dataset_builder = dataset_builder
        self._env_type = env_type
        self._metric_prefix = metric_prefix
        self._episodes_per_eval = episodes_per_eval
        self._evaluator: BitsKnownEvaluator | None = None
        self._dataset = None

    async def _ensure_evaluator(self) -> BitsKnownEvaluator:
        """Build the evaluator once, caching the result."""
        if self._evaluator is None:
            _, test_dataset = await self._dataset_builder()
            if test_dataset is None:
                raise ValueError("BitsKnownEvaluatorBuilder requires a test dataset.")
            self._dataset = test_dataset
            self._evaluator = BitsKnownEvaluator.from_dataset(
                test_dataset,
                env_type=self._env_type,
                metric_prefix=self._metric_prefix,
            )
        return self._evaluator

    def __call__(self) -> "_AsyncBitsKnownEvaluator":
        """Return a wrapper that handles async initialization."""
        return _AsyncBitsKnownEvaluator(self)


class _AsyncBitsKnownEvaluator(SamplingClientEvaluator):
    """Wrapper that handles async evaluator initialization and tracks learning rate.

    This is needed because evaluator_builders are called synchronously,
    but dataset creation is async.

    Also tracks bits_known history to compute effective_bits_per_episode,
    which is the key metric for comparing against theoretical predictions.
    """

    def __init__(self, builder: BitsKnownEvaluatorBuilder):
        self._builder = builder
        # State for computing effective rate
        self._prev_bits_known: float | None = None

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        evaluator = await self._builder._ensure_evaluator()
        metrics = await evaluator(sampling_client)

        # Compute effective bits per episode (the key metric for the study)
        prefix = self._builder._metric_prefix.rstrip("/")
        current_bits = metrics.get(f"{prefix}/known", 0.0)

        if self._prev_bits_known is not None:
            delta_bits = current_bits - self._prev_bits_known
            metrics[f"{prefix}/delta"] = float(delta_bits)

            # Compute bits per episode if we know episodes_per_eval
            episodes_per_eval = self._builder._episodes_per_eval
            if episodes_per_eval is not None and episodes_per_eval > 0:
                bits_per_episode = delta_bits / episodes_per_eval
                metrics[f"{prefix}/per_episode"] = float(bits_per_episode)

        self._prev_bits_known = current_bits
        return metrics


def _build_messages(
    env_type: Literal["single_step", "multi_step"],
    N: int,
    secret: int,
    convo_prefix: list[dict] | None,
) -> list[dict]:
    """Build the message sequence for evaluating a secret."""
    prefix = list(convo_prefix or [])

    if env_type == "single_step":
        return prefix + [
            {"role": "user", "content": build_single_step_user_prompt(N)},
            {"role": "assistant", "content": str(secret)},
        ]

    # Multi-step: one message per bit
    num_bits = num_bits_for_space(N)
    secret_bits = format_secret_bits(secret, N)
    messages = [{"role": "system", "content": build_multi_step_system_prompt(N)}] + prefix
    for idx, bit in enumerate(secret_bits):
        messages.append(
            {"role": "user", "content": build_multi_step_user_prompt(idx, num_bits)}
        )
        messages.append({"role": "assistant", "content": bit})
    return messages
