#!/usr/bin/env python3
"""
xmux-based sweep launcher for memory-rl experiments.

Replicates the logic from run_info_theory_sweep.sh but runs jobs in parallel via tmux.

Usage:
    python run_sweep.py                    # Launch all sweeps at once (may overwhelm Tinker)
    python run_sweep.py --sequential       # Run A, wait, then B, wait, then C, then D
    python run_sweep.py --sweep A          # Run only sweep A
    python run_sweep.py --sweep B C D --sequential  # Run B, C, D sequentially
    python run_sweep.py --dry-run          # Preview without launching
    python run_sweep.py --verbose          # Enable verbose logging

Sweeps:
    A: B sweep (scalar bandwidth) @ fixed N=1024 - 15 jobs
    B: N sweep (scalar) @ fixed B=8 - 15 jobs
    C: Dense control (multi-step per-bit) on the same N grid - 15 jobs
    D: SFT anchor @ N=1024 - 3 jobs

Recommended: Use --sequential to avoid overwhelming Tinker with too many concurrent sessions.
"""

import argparse
import os
import json
from pathlib import Path

from tinker_cookbook.xmux import JobSpec, SwarmConfig, launch_swarm

import train
import sft_train
import sft_data_gen

# Default checkpoints
SINGLE_STEP_CHECKPOINT = "tinker://61ffdf2c-c9ae-52f1-8b0a-d5757c68bee8:train:0/weights/final"
MULTI_STEP_CHECKPOINT = "tinker://1e79325e-97ad-5cfc-aae3-fdc7b5951746:train:0/weights/final"

# Sweep hyperparameters (match bash script defaults)
MODEL = "meta-llama/Llama-3.1-8B"
SEEDS = [0, 1, 2]
NS = [16, 64, 256, 1024, 4096]  # 5 points, log-spaced
BS = [2, "cont", 4, 8, 16, 32]  # B=2 uses binary (1 bit); "cont" uses log_distance (∞ bits); others use binned
GROUP_SIZE = 4

# Fixed-point sweep defaults
N_BSWEEP = 1024  # N for B sweep
B_FIXED = 8      # B for N sweep
SFT_N = 1024     # N for SFT

# Training hyperparams
LR = 4e-5
LORA_RANK = 8
LOSS_FN = "importance_sampling"
EVAL_EVERY = 100
SAVE_EVERY = 500
TEST_N_BATCHES = 200
TEMPERATURE = 1.0
REMOVE_CONSTANT_REWARD_GROUPS = False

NB_SINGLE = 3000   # n_batches for single-step (learning plateaus by ~500)
NB_MULTI = 2000    # n_batches for multi-step

# SFT data generation
SFT_NUM_TRAIN = 2048
SFT_NUM_TEST = 256


def secret_for(seed: int, N: int) -> int:
    """Deterministic secret from seed and N (matches bash script)."""
    return (seed * 9973 + 12345) % N


def job_is_complete(log_dir: str) -> bool:
    """Return True if the run has a saved final checkpoint."""
    checkpoints_path = os.path.join(log_dir, "checkpoints.jsonl")
    if not os.path.exists(checkpoints_path):
        return False
    try:
        with open(checkpoints_path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("name") == "final":
                    return True
    except (OSError, json.JSONDecodeError):
        return False
    return False


def build_sweep_A(experiment_dir: str, wandb_project: str | None) -> list[JobSpec]:
    """A) Bandwidth sweep @ fixed N=1024.

    Tests: binary (1 bit), continuous (∞ bits), binned B=4,8,16,32 (2-5 bits).
    """
    jobs = []
    for seed in SEEDS:
        secret = secret_for(seed, N_BSWEEP)
        for B in BS:
            # B=2 -> binary (1 bit), "cont" -> log_distance (∞ bits), else -> binned
            if B == 2:
                reward_type = "binary"
                reward_bins = None
                name = f"A_bsweep_single_N{N_BSWEEP}_binary_s{seed}"
            elif B == "cont":
                reward_type = "log_distance"
                reward_bins = None
                name = f"A_bsweep_single_N{N_BSWEEP}_continuous_s{seed}"
            else:
                reward_type = "binned_log_distance"
                reward_bins = B
                name = f"A_bsweep_single_N{N_BSWEEP}_B{B}_s{seed}"

            log_dir = os.path.join(experiment_dir, name)

            if job_is_complete(log_dir):
                print(f"Skipping {name} (already complete)")
                continue

            config = train.Config(
                model_name=MODEL,
                env_type="single_step",
                load_checkpoint_path=SINGLE_STEP_CHECKPOINT,
                N=N_BSWEEP,
                fixed_secret=secret,
                reward_type=reward_type,
                reward_bins=reward_bins,
                loss_fn=LOSS_FN,
                learning_rate=LR,
                lora_rank=LORA_RANK,
                batch_size=1,
                group_size=GROUP_SIZE,
                n_batches=NB_SINGLE,
                test_n_batches=TEST_N_BATCHES,
                max_tokens=8,
                temperature=TEMPERATURE,
                remove_constant_reward_groups=REMOVE_CONSTANT_REWARD_GROUPS,
                eval_every=EVAL_EVERY,
                save_every=SAVE_EVERY,
                log_path=log_dir,
                wandb_project=wandb_project,
                wandb_name=name,
            )
            window_name = "A_cont" if B == "cont" else f"A_B{B}"
            jobs.append(JobSpec(
                main_fn=train.run,
                log_relpath=name,
                entrypoint_config=config,
                tmux_window_name=window_name,
            ))
    return jobs


def build_sweep_B(experiment_dir: str, wandb_project: str | None) -> list[JobSpec]:
    """B) N sweep (scalar) @ fixed B=8."""
    jobs = []
    for seed in SEEDS:
        for N in NS:
            secret = secret_for(seed, N)
            name = f"B_nsweep_single_B{B_FIXED}_N{N}_s{seed}"
            log_dir = os.path.join(experiment_dir, name)

            if job_is_complete(log_dir):
                print(f"Skipping {name} (already complete)")
                continue

            config = train.Config(
                model_name=MODEL,
                env_type="single_step",
                load_checkpoint_path=SINGLE_STEP_CHECKPOINT,
                N=N,
                fixed_secret=secret,
                reward_type="binned_log_distance",
                reward_bins=B_FIXED,
                loss_fn=LOSS_FN,
                learning_rate=LR,
                lora_rank=LORA_RANK,
                batch_size=1,
                group_size=GROUP_SIZE,
                n_batches=NB_SINGLE,
                test_n_batches=TEST_N_BATCHES,
                max_tokens=8,
                temperature=TEMPERATURE,
                remove_constant_reward_groups=REMOVE_CONSTANT_REWARD_GROUPS,
                eval_every=EVAL_EVERY,
                save_every=SAVE_EVERY,
                log_path=log_dir,
                wandb_project=wandb_project,
                wandb_name=name,
            )
            jobs.append(JobSpec(
                main_fn=train.run,
                log_relpath=name,
                entrypoint_config=config,
                tmux_window_name=f"B_N{N}",
            ))
    return jobs


def build_sweep_C(experiment_dir: str, wandb_project: str | None) -> list[JobSpec]:
    """C) Dense control (multi-step per-bit) on the same N grid."""
    jobs = []
    for seed in SEEDS:
        for N in NS:
            secret = secret_for(seed, N)
            name = f"C_dense_multi_N{N}_s{seed}"
            log_dir = os.path.join(experiment_dir, name)

            if job_is_complete(log_dir):
                print(f"Skipping {name} (already complete)")
                continue

            config = train.Config(
                model_name=MODEL,
                env_type="multi_step",
                load_checkpoint_path=MULTI_STEP_CHECKPOINT,
                N=N,
                fixed_secret=secret,
                loss_fn=LOSS_FN,
                learning_rate=LR,
                lora_rank=LORA_RANK,
                batch_size=1,
                group_size=GROUP_SIZE,
                n_batches=NB_MULTI,
                test_n_batches=TEST_N_BATCHES,
                max_tokens=2,
                temperature=TEMPERATURE,
                remove_constant_reward_groups=REMOVE_CONSTANT_REWARD_GROUPS,
                eval_every=EVAL_EVERY,
                save_every=SAVE_EVERY,
                log_path=log_dir,
                wandb_project=wandb_project,
                wandb_name=name,
            )
            jobs.append(JobSpec(
                main_fn=train.run,
                log_relpath=name,
                entrypoint_config=config,
                tmux_window_name=f"C_N{N}",
            ))
    return jobs


def build_sweep_D(experiment_dir: str, wandb_project: str | None) -> list[JobSpec]:
    """D) SFT anchor @ N=1024."""
    jobs = []
    for seed in SEEDS:
        secret = secret_for(seed, SFT_N)

        # Generate SFT data if needed (same location as bash script)
        data_dir = os.path.join(experiment_dir, f"sft_data/N{SFT_N}_s{seed}")
        train_data_path = os.path.join(data_dir, "train.jsonl")

        if not os.path.exists(train_data_path):
            print(f"Generating SFT data for seed {seed}...")
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            sft_data_gen.generate_data(
                N=SFT_N,
                fixed_secret=secret,
                num_train=SFT_NUM_TRAIN,
                num_test=SFT_NUM_TEST,
                output_dir=data_dir,
            )

        name = f"D_sft_N{SFT_N}_s{seed}"
        log_dir = os.path.join(experiment_dir, name)

        if job_is_complete(log_dir):
            print(f"Skipping {name} (already complete)")
            continue

        config = sft_train.Config(
            model_name=MODEL,
            N=SFT_N,
            fixed_secret=secret,
            train_data_path=train_data_path,
            log_path=log_dir,
            wandb_project=wandb_project,
            wandb_name=name,
        )
        jobs.append(JobSpec(
            main_fn=sft_train.run,
            log_relpath=name,
            entrypoint_config=config,
            tmux_window_name="D_sft",
        ))
    return jobs


def wait_for_sweep_completion(experiment_dir: str, job_names: list[str], poll_interval: int = 60) -> None:
    """Poll until all jobs in the sweep have completed (have 'final' checkpoint)."""
    import time

    pending = set(job_names)
    print(f"\nWaiting for {len(pending)} jobs to complete...")

    while pending:
        completed = set()
        for name in pending:
            log_dir = os.path.join(experiment_dir, name)
            if job_is_complete(log_dir):
                completed.add(name)
                print(f"  ✓ {name} complete")

        pending -= completed

        if pending:
            print(f"  {len(pending)} jobs remaining, checking again in {poll_interval}s...")
            time.sleep(poll_interval)

    print("All jobs in sweep completed!")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Preview without launching")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--sweep", nargs="+", choices=["A", "B", "C", "D"], default=["A", "B", "C", "D"],
                        help="Which sweeps to run (default: all)")
    parser.add_argument("--experiment-dir", default="/tmp/tinker-examples/memory_rl/runs",
                        help="Base directory for experiment logs")
    parser.add_argument("--wandb-project", default=None, help="W&B project name")
    parser.add_argument("--sweep-name", default="memory-rl", help="tmux session name")
    parser.add_argument("--sequential", action="store_true",
                        help="Run sweeps sequentially, waiting for each to complete before starting next")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between completion checks in sequential mode (default: 60)")
    args = parser.parse_args()

    sweep_builders = {
        "A": build_sweep_A,
        "B": build_sweep_B,
        "C": build_sweep_C,
        "D": build_sweep_D,
    }

    if args.sequential:
        # Run sweeps one at a time, waiting for completion
        for sweep_id in args.sweep:
            print(f"\n{'='*60}")
            print(f"Starting sweep {sweep_id}")
            print(f"{'='*60}")

            builder = sweep_builders[sweep_id]
            jobs = builder(args.experiment_dir, args.wandb_project)

            if not jobs:
                print(f"Sweep {sweep_id}: No jobs to launch (all complete or skipped)")
                continue

            print(f"Sweep {sweep_id}: {len(jobs)} jobs")

            if not args.dry_run:
                config = SwarmConfig(
                    sweep_name=f"{args.sweep_name}-{sweep_id}",
                    max_panes_per_window=4,
                    dry_run=args.dry_run,
                    verbose=args.verbose,
                )
                launch_swarm(jobs, config)

                # Wait for this sweep to complete before starting next
                job_names = [j.log_relpath for j in jobs]
                wait_for_sweep_completion(args.experiment_dir, job_names, args.poll_interval)
            else:
                print(f"[DRY RUN] Would launch {len(jobs)} jobs and wait for completion")
    else:
        # Original behavior: launch all sweeps at once
        all_jobs: list[JobSpec] = []

        for sweep_id in args.sweep:
            builder = sweep_builders[sweep_id]
            jobs = builder(args.experiment_dir, args.wandb_project)
            print(f"Sweep {sweep_id}: {len(jobs)} jobs")
            all_jobs.extend(jobs)

        if not all_jobs:
            print("No jobs to launch (all already exist or skipped)")
            return

        print(f"\nTotal jobs to launch: {len(all_jobs)}")

        # Launch via xmux
        config = SwarmConfig(
            sweep_name=args.sweep_name,
            max_panes_per_window=4,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        launch_swarm(all_jobs, config)


if __name__ == "__main__":
    main()
