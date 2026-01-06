#!/bin/bash
# Sweep B: N sweep with fixed B=8
# Seeds: 0, 1, 2
# N values: 16, 64, 256, 1024, 4096

EXPERIMENT_DIR="/tmp/tinker-examples/memory_rl/runs_dec24"
CHECKPOINT="tinker://61ffdf2c-c9ae-52f1-8b0a-d5757c68bee8:train:0/weights/final"

for SEED in 0 1 2; do
  for N in 16 64 256 1024 4096; do
    SECRET=$(( (SEED * 9973 + 12345) % N ))
    NAME="B_nsweep_single_B8_N${N}_s${SEED}"
    LOG_DIR="${EXPERIMENT_DIR}/${NAME}"

    # Skip if already complete
    if [ -f "${LOG_DIR}/checkpoints.jsonl" ] && grep -q '"name": "final"' "${LOG_DIR}/checkpoints.jsonl" 2>/dev/null; then
      echo "Skipping ${NAME} (already complete)"
      continue
    fi

    echo "Running: ${NAME} (secret=${SECRET})"
    python train.py \
      env_type=single_step \
      N=${N} \
      fixed_secret=${SECRET} \
      reward_type=binned_log_distance \
      reward_bins=8 \
      load_checkpoint_path="${CHECKPOINT}" \
      n_batches=3000 \
      eval_every=100 \
      save_every=500 \
      log_path="${LOG_DIR}"
  done
done
