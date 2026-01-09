#!/usr/bin/env python3
"""
Modal-based sweep runner for memory-rl experiments.

Runs the sweep as a long-running container on Modal. The container orchestrates
training jobs via Tinker (which handles GPU compute), so we only need CPU.

Usage:
    # Deploy and run all sweeps sequentially
    modal run modal_sweep.py

    # Run specific sweeps
    modal run modal_sweep.py --sweep A B

    # Run with W&B logging
    modal run modal_sweep.py --wandb-project memory-rl

    # Dry run (preview jobs)
    modal run modal_sweep.py --dry-run
"""

import modal

app = modal.App("memory-rl-sweep")

# Build image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "tinker",
        "tinker-cookbook",
        "torch",
        "numpy",
        "wandb",
        "chz",
    )
    .copy_local_dir(".", "/app")
)

# Volume for persisting experiment results
volume = modal.Volume.from_name("memory-rl-results", create_if_missing=True)
RESULTS_DIR = "/results"


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("tinker-api-key")],
    volumes={RESULTS_DIR: volume},
    timeout=86400 * 3,  # 3 days max (sweeps can take a while)
    cpu=2,
    memory=4096,
)
async def run_sweep(
    sweeps: list[str] = ["A", "B", "C", "D"],
    wandb_project: str | None = None,
    dry_run: bool = False,
    sequential: bool = True,
    poll_interval: int = 60,
):
    """Run the memory-rl sweep on Modal."""
    import os
    import sys
    import asyncio

    # Add app directory to path
    sys.path.insert(0, "/app")

    from pathlib import Path

    # Import sweep logic
    import train
    import sft_train
    import sft_data_gen
    from run_sweep import (
        build_sweep_A,
        build_sweep_B,
        build_sweep_C,
        build_sweep_D,
        job_is_complete,
        SEEDS,
        NS,
        BS,
    )

    experiment_dir = os.path.join(RESULTS_DIR, "runs")
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    sweep_builders = {
        "A": build_sweep_A,
        "B": build_sweep_B,
        "C": build_sweep_C,
        "D": build_sweep_D,
    }

    print(f"Starting sweep on Modal")
    print(f"Sweeps: {sweeps}")
    print(f"Experiment dir: {experiment_dir}")
    print(f"Sequential: {sequential}")
    print(f"W&B project: {wandb_project}")

    for sweep_id in sweeps:
        print(f"\n{'='*60}")
        print(f"Starting sweep {sweep_id}")
        print(f"{'='*60}")

        builder = sweep_builders[sweep_id]
        jobs = builder(experiment_dir, wandb_project)

        if not jobs:
            print(f"Sweep {sweep_id}: No jobs to launch (all complete)")
            continue

        print(f"Sweep {sweep_id}: {len(jobs)} jobs to run")

        if dry_run:
            for job in jobs:
                print(f"  [DRY RUN] Would run: {job.log_relpath}")
            continue

        # Run jobs - can run in parallel or sequentially
        if sequential:
            for job in jobs:
                print(f"\nRunning: {job.log_relpath}")
                try:
                    await job.main_fn(job.entrypoint_config)
                    print(f"  ✓ {job.log_relpath} complete")
                except Exception as e:
                    print(f"  ✗ {job.log_relpath} failed: {e}")
                # Commit volume after each job
                volume.commit()
        else:
            # Run all jobs in this sweep concurrently
            async def run_job(job):
                print(f"Starting: {job.log_relpath}")
                try:
                    await job.main_fn(job.entrypoint_config)
                    print(f"  ✓ {job.log_relpath} complete")
                except Exception as e:
                    print(f"  ✗ {job.log_relpath} failed: {e}")

            await asyncio.gather(*[run_job(job) for job in jobs])
            volume.commit()

    print("\n" + "="*60)
    print("Sweep complete!")
    print(f"Results saved to: {experiment_dir}")
    volume.commit()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("tinker-api-key")],
    volumes={RESULTS_DIR: volume},
    timeout=86400,  # 1 day
    cpu=2,
    memory=4096,
)
async def run_single_job(
    sweep_id: str,
    job_index: int,
    wandb_project: str | None = None,
):
    """Run a single job from a sweep. Useful for retrying failed jobs."""
    import os
    import sys

    sys.path.insert(0, "/app")
    from pathlib import Path

    from run_sweep import (
        build_sweep_A,
        build_sweep_B,
        build_sweep_C,
        build_sweep_D,
    )

    experiment_dir = os.path.join(RESULTS_DIR, "runs")
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    sweep_builders = {
        "A": build_sweep_A,
        "B": build_sweep_B,
        "C": build_sweep_C,
        "D": build_sweep_D,
    }

    jobs = sweep_builders[sweep_id](experiment_dir, wandb_project)

    if job_index >= len(jobs):
        raise ValueError(f"Job index {job_index} out of range (sweep {sweep_id} has {len(jobs)} jobs)")

    job = jobs[job_index]
    print(f"Running: {job.log_relpath}")
    await job.main_fn(job.entrypoint_config)
    print(f"  ✓ {job.log_relpath} complete")
    volume.commit()


@app.local_entrypoint()
def main(
    sweep: list[str] = ["A", "B", "C", "D"],
    wandb_project: str = None,
    dry_run: bool = False,
    parallel: bool = False,
    poll_interval: int = 60,
):
    """Launch the sweep on Modal.

    Args:
        sweep: Which sweeps to run (A, B, C, D)
        wandb_project: W&B project name for logging
        dry_run: Preview jobs without running
        parallel: Run jobs within each sweep in parallel (default: sequential)
        poll_interval: Seconds between completion checks
    """
    run_sweep.remote(
        sweeps=sweep,
        wandb_project=wandb_project,
        dry_run=dry_run,
        sequential=not parallel,
        poll_interval=poll_interval,
    )
