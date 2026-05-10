#!/bin/bash
# Run chr19 demo SVGP models with low learning rate and animation.
# Usage: run_demo_chr19_svgp.sh <GPU_ID> [CONFIG_DIR]

set -euo pipefail

GPU_ID="${1:?Usage: $0 <GPU_ID> [CONFIG_DIR]}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

CONDA_ENV="chromgp"
CONFIG_DIR="${2:-/gladstone/engelhardt/home/lchumpitaz/gitclones/ChromGP/configs/demo_chr19_lr1e3}"

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

echo "===== chr19 demo SVGP runs on GPU ${GPU_ID} ====="
echo "Start: $(date)"

for config in \
    "${CONFIG_DIR}/gm12878_hic_svgp.yaml" \
    "${CONFIG_DIR}/gm12878_chipseq_svgp.yaml" \
    "${CONFIG_DIR}/imr90_hic_svgp.yaml" \
    "${CONFIG_DIR}/imr90_chipseq_svgp.yaml"
do
    echo "============================================================"
    echo "  Config: ${config}"
    echo "  Start: $(date)"
    echo "============================================================"
    chromgp run preprocess train analyze figures -c "$config" --animation
    echo "  Done: ${config} at $(date)"
done

echo "===== chr19 demo SVGP runs done at $(date) ====="
