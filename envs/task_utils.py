from __future__ import annotations

import math
import re
from typing import Literal


RewardType = Literal[
    "binary",
    "log_distance",
    "binned_log_distance",
]


def validate_secret(secret: int, N: int) -> None:
    if N <= 0:
        raise ValueError("N must be positive")
    if not (0 <= secret < N):
        raise ValueError(f"Secret {secret} must be in [0, {N - 1}] when N={N}")


def validate_reward_config(reward_type: RewardType, reward_bins: int | None = None) -> None:
    if reward_type not in {"binary", "log_distance", "binned_log_distance"}:
        raise ValueError(
            f"Unsupported reward_type '{reward_type}'. "
            f"Use 'binary', 'log_distance', or 'binned_log_distance'."
        )
    if reward_type == "binned_log_distance":
        if reward_bins is None or reward_bins < 2:
            raise ValueError("binned_log_distance requires reward_bins >= 2.")


def build_single_step_user_prompt(N: int) -> str:
    return (
        "You are playing a memorization game. "
        f"There is a secret integer between 0 and {N - 1}. "
        "Output ONLY the integer, on the first line, with NO extra text.\n"
        "What is the secret number?"
    )


def build_multi_step_system_prompt(N: int) -> str:
    num_bits = num_bits_for_space(N)
    return (
        f"You are playing a bit-by-bit memorization game. "
        f"The secret number is between 0 and {N - 1} and uses {num_bits} bits. "
        f"Output exactly one bit (0 or 1) per response with no punctuation."
    )


def build_multi_step_user_prompt(position: int, num_bits: int) -> str:
    remaining = num_bits - position - 1
    return f"Provide bit #{position + 1}. Remaining bits after this: {remaining}."


def parse_single_step_guess(sample_str: str, N: int, strict: bool = False) -> int | None:
    """
    Parse the model's guess from its response.

    Args:
        sample_str: Model's raw output
        N: Upper bound (exclusive) for valid guesses
        strict: If True, only accept clean numeric output (digits only, optional trailing whitespace).
                If False, extract first digit sequence (lenient).
    """
    stripped = sample_str.strip()
    if not stripped:
        return None

    if strict:
        # Strict: must be only digits (no periods, spaces, repeated numbers)
        if not stripped.isdigit():
            return None
        try:
            guess = int(stripped)
        except ValueError:
            return None
    else:
        # Lenient: extract first contiguous digit sequence from the first line
        first_line = stripped.split('\n')[0]
        digits = re.search(r'\d+', first_line)
        if not digits:
            return None
        try:
            guess = int(digits.group())
        except ValueError:
            return None

    return guess if 0 <= guess < N else None


def parse_bit_guess(sample_str: str) -> str | None:
    """
    Parse a single bit from the model's response.

    Strict parsing: only accepts exactly "0" or "1" (with optional whitespace).
    Returns None (format error) for anything else like "101", "bit: 0", etc.
    """
    stripped = sample_str.strip()
    if stripped in ("0", "1"):
        return stripped
    return None


def compute_single_step_reward(
    secret: int,
    guess: int | None,
    reward_type: RewardType,
    N: int,
    reward_bins: int | None = None,
) -> tuple[float, bool, float]:
    """
    Compute reward for a single-step guess.

    Returns: (reward, correct, distance)
    """
    max_bits = math.log2(N) if N > 1 else 1.0

    if guess is None:
        distance = float(N - 1)
        correct = False
    else:
        distance = float(abs(guess - secret))
        correct = guess == secret

    if reward_type == "binary":
        base_reward = float(correct)
    elif reward_type == "log_distance":
        base_reward = max(0.0, 1.0 - (math.log2(distance + 1) / max_bits))
    elif reward_type == "binned_log_distance":
        validate_reward_config(reward_type, reward_bins)
        raw_reward = max(0.0, 1.0 - (math.log2(distance + 1) / max_bits))
        # Quantize to B bins in [0, 1], inclusive of endpoints.
        # reward_bins is guaranteed non-None by validate_reward_config above
        bin_index = int(round(raw_reward * (reward_bins - 1)))  # type: ignore[operator]
        base_reward = bin_index / float(reward_bins - 1)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")

    return base_reward, correct, distance


def num_bits_for_space(N: int) -> int:
    """Number of bits needed to represent integers in [0, N)."""
    if N <= 1:
        return 1
    return max(1, math.ceil(math.log2(N)))


def format_secret_bits(secret: int, N: int) -> str:
    """Format secret as binary string with appropriate width."""
    width = num_bits_for_space(N)
    return format(secret, f"0{width}b")


def sum_weighted_logprobs(logprobs: list[float | None], weights: list[float]) -> tuple[float, int]:
    """Sum logprobs weighted by mask, returning (total, count)."""
    total = 0.0
    count = 0
    for logprob, weight in zip(logprobs, weights, strict=True):
        if weight <= 0 or logprob is None:
            continue
        total += logprob * weight
        count += 1
    return total, count


def compute_bits_known(target_logprob: float, N: int) -> tuple[float, float]:
    """
    Compute bits_known from target logprob.

    Returns: (bits_known_raw, bits_known_clamped)
    """
    signal_bits = math.log2(N) if N > 1 else 1.0
    bits_known_raw = signal_bits + (target_logprob / math.log(2))
    bits_known_clamped = max(0.0, min(signal_bits, bits_known_raw))
    return bits_known_raw, bits_known_clamped
