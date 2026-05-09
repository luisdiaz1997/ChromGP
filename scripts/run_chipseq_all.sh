#!/bin/bash
# Master script for all available ChIP-seq cell types.
# Runs each cell type across two GPUs: chr1-11 on GPU 0 and chr12-22 on GPU 1.

set -euo pipefail

SCRIPT_DIR="/gladstone/engelhardt/home/lchumpitaz/gitclones/ChromGP/scripts"

echo "===== ChIP-seq SVGP + MGGP SVGP: GM12878, IMR90, K562 ====="
echo "Start: $(date)"

for celltype in GM12878 IMR90 K562; do
    echo
    echo "===== ${celltype} ====="

    bash "${SCRIPT_DIR}/run_chipseq_chromosomes.sh" "$celltype" 0 1 11 &
    PID_GPU0=$!

    bash "${SCRIPT_DIR}/run_chipseq_chromosomes.sh" "$celltype" 1 12 22 &
    PID_GPU1=$!

    wait "$PID_GPU0" "$PID_GPU1"
    echo "===== ${celltype} done at $(date) ====="
done

echo "===== ALL ChIP-seq runs done at $(date) ====="
