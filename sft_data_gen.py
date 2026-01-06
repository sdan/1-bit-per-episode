import json
import logging
from pathlib import Path

import chz

from envs import build_single_step_user_prompt

logger = logging.getLogger(__name__)


def generate_sft_data(N: int, secret: int, num_examples: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = build_single_step_user_prompt(N)

    with output_path.open("w") as f:
        for _ in range(num_examples):
            example = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": str(secret)},
                ],
            }
            f.write(json.dumps(example) + "\n")


@chz.chz
class Config:
    N: int = 16
    fixed_secret: int = 0
    seed: int = 0
    num_train: int = 1000
    num_test: int = 200
    output_dir: str = "/tmp/tinker-examples/memory_rl/sft_data"


def generate_data(
    N: int,
    fixed_secret: int,
    num_train: int,
    num_test: int,
    output_dir: str,
) -> None:
    """Generate SFT data. Called by xmux sweep or CLI."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_path = out_path / "train.jsonl"
    test_path = out_path / "test.jsonl"

    generate_sft_data(N, fixed_secret, num_train, train_path)
    generate_sft_data(N, fixed_secret, num_test, test_path)

    logger.info(f"Secret: {fixed_secret}")
    logger.info(f"Wrote {num_train} train examples to {train_path}")
    logger.info(f"Wrote {num_test} test examples to {test_path}")


def main() -> None:
    """CLI entrypoint."""
    cfg = chz.entrypoint(Config)
    generate_data(
        N=cfg.N,
        fixed_secret=cfg.fixed_secret,
        num_train=cfg.num_train,
        num_test=cfg.num_test,
        output_dir=cfg.output_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
