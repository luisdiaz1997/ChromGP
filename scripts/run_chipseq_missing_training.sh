#!/bin/bash
# Resume only missing ChIP-seq training outputs on one GPU.
# Usage: run_chipseq_missing_training.sh <GPU_ID> <SIDE>
#
# SIDE=gpu0 runs lower chromosome ranges; SIDE=gpu1 runs upper ranges.

set -euo pipefail

GPU_ID="${1:?Usage: $0 <GPU_ID> <SIDE>}"
SIDE="${2:?}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

CONDA_ENV="chromgp"
CONFIG_ROOT="/gladstone/engelhardt/home/lchumpitaz/gitclones/ChromGP/configs"
OUTPUT_ROOT="/gladstone/engelhardt/home/lchumpitaz/gitclones/ChromGP/outputs"

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

run_if_missing() {
    local celltype="$1"
    local model="$2"
    local chr="$3"
    local extra_flags="${4:-}"
    local config="${CONFIG_ROOT}/chipseq_${celltype}_chr${chr}/${model}.yaml"
    local checkpoint="${OUTPUT_ROOT}/chipseq_${celltype}/chr${chr}/${model}/checkpoints/model_final.pt"

    if [[ -f "$checkpoint" ]]; then
        echo "Skipping ${celltype} ${model} chr${chr}: checkpoint exists"
        return
    fi

    echo "============================================================"
    echo "  GPU ${GPU_ID}: ${celltype} ${model} chr${chr}"
    echo "  Config: ${config}"
    echo "  Start: $(date)"
    echo "============================================================"
    chromgp run preprocess train analyze figures -c "$config" $extra_flags
    echo "  Done ${celltype} ${model} chr${chr} at $(date)"
}

case "$SIDE" in
    gpu0)
        echo "===== GPU $GPU_ID missing ChIP-seq training: lower chromosomes ====="
        for chr in $(seq 4 11); do
            run_if_missing IMR90 mggp_svgp "$chr"
        done
        for chr in $(seq 1 11); do
            run_if_missing K562 svgp "$chr" "--animation"
        done
        for chr in $(seq 1 11); do
            run_if_missing K562 mggp_svgp "$chr"
        done
        ;;
    gpu1)
        echo "===== GPU $GPU_ID missing ChIP-seq training: upper chromosomes ====="
        for chr in $(seq 16 22); do
            run_if_missing IMR90 mggp_svgp "$chr"
        done
        for chr in $(seq 12 22); do
            run_if_missing K562 svgp "$chr" "--animation"
        done
        for chr in $(seq 12 22); do
            run_if_missing K562 mggp_svgp "$chr"
        done
        ;;
    *)
        echo "Unknown SIDE: $SIDE (expected gpu0 or gpu1)" >&2
        exit 1
        ;;
esac

echo "===== GPU $GPU_ID missing ChIP-seq training done at $(date) ====="
