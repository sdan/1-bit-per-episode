from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import chz
import numpy as np
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

from .task_utils import (
    build_multi_step_system_prompt,
    build_multi_step_user_prompt,
    format_secret_bits,
    num_bits_for_space,
    parse_bit_guess,
    validate_secret,
)


class MultiStepEnv(Env):
    """
    Multi-step (bit-by-bit) memory environment.

    The model outputs one bit at a time, receiving reward per bit.
    This provides denser feedback than single-step binary reward.
    """

    def __init__(
        self,
        *,
        fixed_secret: int,
        N: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        validate_secret(fixed_secret, N)
        self.fixed_secret = fixed_secret
        self.N = N
        self.renderer = renderer
        self.convo_prefix = convo_prefix or []

        self.secret_bits = format_secret_bits(fixed_secret, N)
        self.num_bits = num_bits_for_space(N)

        self.position = 0
        self.conversation: list[renderers.Message] = []
        self._all_bits_correct = True

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _messages(self) -> list[renderers.Message]:
        system_msg = {"role": "system", "content": build_multi_step_system_prompt(self.N)}
        return [system_msg] + self.convo_prefix + self.conversation

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        first_request = {"role": "user", "content": build_multi_step_user_prompt(0, self.num_bits)}
        self.conversation.append(first_request)
        # RoleColonRenderer training examples include a leading space after "Assistant:".
        # Prefill it so max_tokens=1 can still produce a single-bit output.
        return self.renderer.build_generation_prompt(self._messages(), prefill=" "), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = message["content"]
        guessed_bit = parse_bit_guess(content)

        self.conversation.append({"role": "assistant", "content": content})

        correct_bit = self.secret_bits[self.position]
        correct = guessed_bit == correct_bit
        reward = 1.0 if correct else 0.0
        if not correct:
            self._all_bits_correct = False
        format_error = float(guessed_bit is None)
        parse_success_float = float(parse_success)

        self.position += 1
        episode_done = self.position >= self.num_bits

        if not episode_done:
            next_request = {"role": "user", "content": build_multi_step_user_prompt(self.position, self.num_bits)}
            self.conversation.append(next_request)

        logtree.log_text(
            "[MultiStepEnv] "
            f"pos={self.position - 1}/{self.num_bits - 1}, "
            f"secret_bits={self.secret_bits}, "
            f"guess_raw={content!r}, "
            f"guess_bit={guessed_bit}, "
            f"correct_bit={correct_bit}, "
            f"reward={reward:.2f}"
        )

        if episode_done:
            next_observation = tinker.ModelInput.empty()
        else:
            next_observation = self.renderer.build_generation_prompt(self._messages(), prefill=" ")

        return StepResult(
            reward=reward,
            episode_done=episode_done,
            next_observation=next_observation,
            next_stop_condition=self.stop_condition,
            metrics={
                "correct_bit": float(correct),
                "reward_signal": reward,
                "success_count": int(episode_done and self._all_bits_correct),
                "format_error": format_error,
                "parse_success": parse_success_float,
            },
        )


@dataclass(frozen=True)
class MultiStepEnvGroupBuilder(EnvGroupBuilder):
    fixed_secret: int
    N: int
    renderer: renderers.Renderer
    num_envs: int
    convo_prefix: list[renderers.Message] | None = None

    async def make_envs(self) -> Sequence[Env]:
        return [
            MultiStepEnv(
                fixed_secret=self.fixed_secret,
                N=self.N,
                renderer=self.renderer,
                convo_prefix=self.convo_prefix,
            )
            for _ in range(self.num_envs)
        ]

    def logging_tags(self) -> list[str]:
        k_bits = num_bits_for_space(self.N)
        return [f"multi_step_N{self.N}_k{k_bits}"]


class MultiStepDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        N: int,
        n_batches: int,
        convo_prefix: list[renderers.Message] | None,
        fixed_secret: int | None,
        seed: int,
        split: Literal["train", "test"] = "train",
    ):
        self._rng = np.random.RandomState(seed)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.N = N
        self.convo_prefix = convo_prefix
        self.n_batches = n_batches
        self.split = split

        if fixed_secret is not None:
            validate_secret(fixed_secret, N)
            self.fixed_secret = int(fixed_secret)
        elif split == "train":
            self.fixed_secret = int(self._rng.randint(0, N))
        else:
            self.fixed_secret = None

        if split == "test":
            if self.fixed_secret is not None:
                total = n_batches * batch_size
                self.test_secrets = [self.fixed_secret] * total
            else:
                test_rng = np.random.RandomState(seed + 1)
                self.test_secrets = test_rng.randint(0, N, size=n_batches * batch_size).tolist()

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        if self.split == "train":
            if self.fixed_secret is None:
                raise RuntimeError("Training split requires a fixed secret")
            return [
                MultiStepEnvGroupBuilder(
                    fixed_secret=self.fixed_secret,
                    N=self.N,
                    renderer=self.renderer,
                    num_envs=self.group_size,
                    convo_prefix=self.convo_prefix,
                )
                for _ in range(self.batch_size)
            ]

        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.test_secrets))
        return [
            MultiStepEnvGroupBuilder(
                fixed_secret=secret,
                N=self.N,
                renderer=self.renderer,
                num_envs=self.group_size,
                convo_prefix=self.convo_prefix,
            )
            for secret in self.test_secrets[batch_start:batch_end]
        ]

    def __len__(self) -> int:
        return self.n_batches


@chz.chz
class MultiStepDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    group_size: int = 1
    model_name_for_tokenizer: str
    renderer_name: str
    N: int = 16
    n_batches: int = 100
    test_n_batches: int | None = None
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = None
    fixed_secret: int | None = None
    seed: int = 1337

    async def __call__(self) -> tuple[MultiStepDataset, MultiStepDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        if self.group_size < 1:
            raise ValueError("group_size must be >= 1.")

        if self.convo_prefix == "standard":
            num_bits = num_bits_for_space(self.N)
            prefix: list[renderers.Message] | None = [
                {"role": "user", "content": f"Provide bit #1. Remaining bits after this: {num_bits - 1}."},
                {"role": "assistant", "content": "0"},
            ]
        else:
            prefix = self.convo_prefix

        train_dataset = MultiStepDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            N=self.N,
            n_batches=self.n_batches,
            convo_prefix=prefix,
            fixed_secret=self.fixed_secret,
            seed=self.seed,
            split="train",
        )

        test_dataset = MultiStepDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            N=self.N,
            n_batches=(
                int(self.test_n_batches)
                if self.test_n_batches is not None
                else max(10, self.n_batches // 10)
            ),
            convo_prefix=prefix,
            fixed_secret=train_dataset.fixed_secret,
            seed=self.seed,
            split="test",
        )

        return train_dataset, test_dataset
