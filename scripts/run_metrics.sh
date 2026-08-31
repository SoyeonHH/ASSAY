#!/bin/bash
# Compute ASSAY metrics (Validity, Stability, Accuracy) from judge results.
#
# The judge writes to judge_results/<input stem>/<model>.jsonl, so the anchor
# scores live in judge_results/<prefix> and the augmented conditions live in
# judge_results/<prefix>_<swap type>_<rate>pct.
#
# Usage:
#   ./scripts/run_metrics.sh --prefix sample_input
#   ./scripts/run_metrics.sh --prefix test_high_impact --model gpt-4o

set -euo pipefail

MODEL="all"
PREFIX="sample_input"
JUDGE_DIR="judge_results"
OUTPUT_DIR="results/metrics"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2;;
        --prefix) PREFIX="$2"; shift 2;;
        --judge-dir) JUDGE_DIR="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

python -m assay.metrics.run \
    --model "$MODEL" \
    --gt_dir "$JUDGE_DIR/$PREFIX" \
    --error_dir "$JUDGE_DIR" \
    --repr_dir "$JUDGE_DIR" \
    --dataset_prefix "$PREFIX" \
    --output_dir "$OUTPUT_DIR"
