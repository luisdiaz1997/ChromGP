#!/bin/bash
# Master script: K562 + HUVEC on both GPUs after IMR90 finishes.
# Run this when IMR90 is done.
# GPU 0: K562 chr1-11 → HUVEC chr1-11
# GPU 1: K562 chr12-22 → HUVEC chr12-22

SCRIPT_DIR="/gladstone/engelhardt/home/lchumpitaz/gitclones/ChromGP/scripts"

echo "===== POST-IMR90: K562 + HUVEC ====="
echo "Start: $(date)"

# Launch both GPUs in parallel (background)
bash "${SCRIPT_DIR}/run_k562_chromosomes.sh" 0 1 11 &
PID_GPU0=$!

bash "${SCRIPT_DIR}/run_k562_chromosomes.sh" 1 12 22 &
PID_K562_GPU1=$!

wait $PID_GPU0 $PID_K562_GPU1
echo "K562 done at $(date)"

# Now HUVEC
bash "${SCRIPT_DIR}/run_huvec_chromosomes.sh" 0 1 11 &
PID_GPU0=$!

bash "${SCRIPT_DIR}/run_huvec_chromosomes.sh" 1 12 22 &
PID_GPU1=$!

wait $PID_GPU0 $PID_GPU1
echo "===== ALL DONE at $(date) ====="
