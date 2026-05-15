#!/bin/bash
# Run MGGP SVGP pipeline for assigned IMR90 chromosomes on a specific GPU.
# Usage: run_mggp_svgp_imr90_chromosomes.sh <GPU_ID> <START_CHR> <END_CHR>
#
# Example:
#   run_mggp_svgp_imr90_chromosomes.sh 0 1 11   # chr1-11 on GPU 0
#   run_mggp_svgp_imr90_chromosomes.sh 1 12 22  # chr12-22 on GPU 1

set -euo pipefail

GPU_ID="${1:?Usage: $0 <GPU_ID> <START_CHR> <END_CHR>}"
START="${2:?}"
END="${3:?}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

CONDA_ENV="chromgp"
CONFIG_BASE="/gladstone/engelhardt/home/lchumpitaz/gitclones/ChromGP/configs/4DNFIJTOIGOI/chr"

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

echo "======== GPU $GPU_ID: IMR90 chr${START} through chr${END} (MGGP SVGP) ========"
echo "Start time: $(date)"
echo

for chr in $(seq "$START" "$END"); do
    CONFIG="${CONFIG_BASE}${chr}/mggp_svgp.yaml"
    echo "============================================================"
    echo "  GPU $GPU_ID — IMR90 chr$chr  (MGGP SVGP)"
    echo "  Config: $CONFIG"
    echo "  Start: $(date)"
    echo "============================================================"

    chromgp run preprocess train analyze figures -c "$CONFIG" --animation

    echo "  Done chr$chr at $(date)"
    echo
done

echo "======== GPU $GPU_ID: all done at $(date) ========"
