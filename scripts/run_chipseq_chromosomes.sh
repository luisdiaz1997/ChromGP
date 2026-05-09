#!/bin/bash
# Generate ChIP-seq SVGP + MGGP SVGP figures for assigned chromosomes on one GPU.
# Usage: run_chipseq_chromosomes.sh <CELLTYPE> <GPU_ID> <START_CHR> <END_CHR>
#
# Example:
#   run_chipseq_chromosomes.sh GM12878 0 1 11
#   run_chipseq_chromosomes.sh GM12878 1 12 22

set -euo pipefail

CELLTYPE="${1:?Usage: $0 <CELLTYPE> <GPU_ID> <START_CHR> <END_CHR>}"
GPU_ID="${2:?}"
START="${3:?}"
END="${4:?}"

case "$CELLTYPE" in
    GM12878|IMR90|K562) ;;
    *)
        echo "Unknown CELLTYPE: $CELLTYPE (expected GM12878, IMR90, or K562)" >&2
        exit 1
        ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_ID"

CONDA_ENV="chromgp"
CONFIG_BASE="/gladstone/engelhardt/home/lchumpitaz/gitclones/ChromGP/configs/chipseq_${CELLTYPE}_chr"

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

echo "======== GPU $GPU_ID: ${CELLTYPE} ChIP-seq chr${START}-${END} — figures only ========"
echo "Start time: $(date)"

echo
echo "##### PHASE 1: SVGP #####"
for chr in $(seq "$START" "$END"); do
    CONFIG="${CONFIG_BASE}${chr}/svgp.yaml"
    CHECKPOINT="/gladstone/engelhardt/home/lchumpitaz/gitclones/ChromGP/outputs/chipseq_${CELLTYPE}/chr${chr}/svgp/checkpoints/model_final.pt"
    echo "===== ${CELLTYPE} ChIP-seq SVGP chr$chr ====="
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "Skipping chr$chr SVGP: checkpoint not found at $CHECKPOINT"
        continue
    fi
    chromgp run figures -c "$CONFIG" --animation
    echo "Done chr$chr at $(date)"
done

echo
echo "##### PHASE 2: MGGP SVGP #####"
for chr in $(seq "$START" "$END"); do
    CONFIG="${CONFIG_BASE}${chr}/mggp_svgp.yaml"
    CHECKPOINT="/gladstone/engelhardt/home/lchumpitaz/gitclones/ChromGP/outputs/chipseq_${CELLTYPE}/chr${chr}/mggp_svgp/checkpoints/model_final.pt"
    echo "===== ${CELLTYPE} ChIP-seq MGGP SVGP chr$chr ====="
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "Skipping chr$chr MGGP SVGP: checkpoint not found at $CHECKPOINT"
        continue
    fi
    chromgp run figures -c "$CONFIG"
    echo "Done chr$chr at $(date)"
done

echo
echo "======== GPU $GPU_ID: ${CELLTYPE} ChIP-seq done at $(date) ========"
