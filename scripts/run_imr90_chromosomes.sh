#!/bin/bash
# Run SVGP then MGGP SVGP for assigned IMR90 chromosomes on a specific GPU.
# Usage: run_imr90_chromosomes.sh <GPU_ID> <START_CHR> <END_CHR>

set -euo pipefail

GPU_ID="${1:?Usage: $0 <GPU_ID> <START_CHR> <END_CHR>}"
START="${2:?}"
END="${3:?}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
SCRIPT_DIR="/gladstone/engelhardt/home/lchumpitaz/gitclones/ChromGP/scripts"

echo "======== GPU $GPU_ID: IMR90 chr${START}-${END} — SVGP + MGGP SVGP ========"
echo "Start time: $(date)"

echo
echo "##### PHASE 1: SVGP #####"
bash "${SCRIPT_DIR}/run_svgp_imr90_chromosomes.sh" "$GPU_ID" "$START" "$END"

echo
echo "##### PHASE 2: MGGP SVGP #####"
bash "${SCRIPT_DIR}/run_mggp_svgp_imr90_chromosomes.sh" "$GPU_ID" "$START" "$END"

echo
echo "======== GPU $GPU_ID: IMR90 all done at $(date) ========"
