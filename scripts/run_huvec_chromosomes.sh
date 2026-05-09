#!/bin/bash
# Run SVGP + MGGP SVGP for HUVEC chromosomes.
# Usage: run_huvec_chromosomes.sh <GPU_ID> <START_CHR> <END_CHR>

set -euo pipefail

GPU_ID="${1:?Usage: $0 <GPU_ID> <START_CHR> <END_CHR>}"
START="${2:?}"
END="${3:?}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

CONDA_ENV="chromgp"
CONFIG_BASE="/gladstone/engelhardt/home/lchumpitaz/gitclones/ChromGP/configs/4DNFIRMZ7QTE_chr"

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

echo "======== GPU $GPU_ID: HUVEC chr${START}-${END} — SVGP + MGGP SVGP ========"
echo "Start time: $(date)"

echo
echo "##### PHASE 1: SVGP #####"
for chr in $(seq "$START" "$END"); do
    CONFIG="${CONFIG_BASE}${chr}/svgp.yaml"
    echo "===== HUVEC SVGP chr$chr ====="
    chromgp run preprocess train analyze figures -c "$CONFIG" --animation
    echo "Done chr$chr at $(date)"
done

echo
echo "##### PHASE 2: MGGP SVGP #####"
for chr in $(seq "$START" "$END"); do
    CONFIG="${CONFIG_BASE}${chr}/mggp_svgp.yaml"
    echo "===== HUVEC MGGP SVGP chr$chr ====="
    chromgp run preprocess train analyze figures -c "$CONFIG" --animation
    echo "Done chr$chr at $(date)"
done

echo
echo "======== GPU $GPU_ID: HUVEC all done at $(date) ========"
