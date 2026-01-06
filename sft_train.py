import asyncio
import logging
from datetime import datetime
from pathlib import Path

import chz
import tinker

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.eval.evaluators import EvaluatorBuilder, SamplingClientEvaluator
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from eval import BitsKnownEvaluator
from envs import RewardType, SingleStepDatasetBuilder

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B"
    renderer_name: str | None = None

    N: int = 16
    fixed_secret: int = 0
    train_data_path: str = "/tmp/tinker-examples/memory_rl/sft_data/train.jsonl"
    reward_type: RewardType = "binary"
    reward_bins: int | None = None

    learning_rate: float = 2e-4
    lr_schedule: str = "linear"
    num_epochs: int = 1
    batch_size: int = 128
    eval_every: int = 1

    eval_batch_size: int = 64
    eval_n_batches: int = 10
    eval_max_tokens: int = 8
    eval_bits: bool = True

    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    base_url: str | None = None
    max_length: int | None = 2048
    load_checkpoint_path: str | None = None


class _LazyEnvEvaluator(SamplingClientEvaluator):
    def __init__(
        self,
        dataset_builder: SingleStepDatasetBuilder,
        max_tokens: int,
        name: str = "test",
    ):
        self._dataset_builder = dataset_builder
        self._max_tokens = max_tokens
        self._name = name
        self._evaluator: RLTestSetEvaluator | None = None

    async def _ensure_evaluator(self) -> RLTestSetEvaluator:
        if self._evaluator is None:
            _, test_dataset = await self._dataset_builder()
            self._evaluator = RLTestSetEvaluator(
                dataset=test_dataset,
                max_tokens=self._max_tokens,
                name=self._name,
            )
        return self._evaluator

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        evaluator = await self._ensure_evaluator()
        return await evaluator(sampling_client)


def build_config(cli: Config) -> train.Config:
    renderer_name = cli.renderer_name or model_info.get_recommended_renderer_name(cli.model_name)

    dataset_builder = FromConversationFileBuilder(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=cli.model_name,
            renderer_name=renderer_name,
            max_length=cli.max_length,
            batch_size=cli.batch_size,
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        ),
        file_path=cli.train_data_path,
        test_size=0,
    )

    env_dataset_builder = SingleStepDatasetBuilder(
        batch_size=cli.eval_batch_size,
        model_name_for_tokenizer=cli.model_name,
        renderer_name=renderer_name,
        N=cli.N,
        reward_type=cli.reward_type,
        reward_bins=cli.reward_bins,
        n_batches=cli.eval_n_batches,
        convo_prefix=None,
        fixed_secret=cli.fixed_secret,
        seed=0,
    )

    evaluator_builders: list[EvaluatorBuilder] = []
    evaluator_builders.append(
        lambda builder=env_dataset_builder: _LazyEnvEvaluator(
            dataset_builder=builder,
            max_tokens=cli.eval_max_tokens,
        )
    )
    if cli.eval_bits:
        evaluator_builders.append(
            lambda builder=env_dataset_builder: BitsKnownEvaluator(
                dataset_builder=builder,
                env_type="single_step",
                metric_prefix="test/bits",
            )
        )

    date_and_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_run_name = f"sft_N{cli.N}_bs{cli.batch_size}_lr{cli.learning_rate}_{date_and_time}"

    return train.Config(
        model_name=cli.model_name,
        load_checkpoint_path=cli.load_checkpoint_path,
        log_path=cli.log_path or f"/tmp/tinker-examples/memory_rl/sft/{default_run_name}",
        dataset_builder=dataset_builder,
        learning_rate=cli.learning_rate,
        lr_schedule=cli.lr_schedule,
        num_epochs=cli.num_epochs,
        eval_every=cli.eval_every,
        base_url=cli.base_url,
        evaluator_builders=evaluator_builders,
        wandb_project=cli.wandb_project,
        wandb_name=cli.wandb_name or default_run_name,
    )


async def run(cli: Config) -> None:
    """Run SFT training with the given config. Called by xmux or CLI."""
    config = build_config(cli)

    logger.info(f"SFT Memory Training: N={cli.N}, secret={cli.fixed_secret}")
    logger.info(f"Training data: {cli.train_data_path}")
    logger.info(f"Log path: {config.log_path}")

    Path(config.log_path).parent.mkdir(parents=True, exist_ok=True)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="resume")
    await train.main(config)


async def main() -> None:
    """CLI entrypoint."""
    cli = chz.entrypoint(Config)
    await run(cli)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
