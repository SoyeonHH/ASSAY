#!/bin/bash
# Build DPO training pairs from multi-model judge results.
#
# Usage:
#   ./scripts/run_dpo_build.sh --dataset-prefix train_2000 \
#       --judge-root judge_results \
#       --models "Qwen3-8B:Qwen3-8B.jsonl,Qwen3-32B:Qwen3-32B.jsonl,Llama-3.1-8B-Instruct:Llama-3.1-8B-Instruct.jsonl,gemini-2.5-flash:gemini-2.5-flash.jsonl"

set -euo pipefail

JUDGE_ROOT="judge_results"
OUTPUT_DIR="dpo_dataset"
MODELS=""
MIN_DELTA="0.5"
MAX_PER_SAMPLE="5"
DATASET_PREFIX="sample_input"

while [[ $# -gt 0 ]]; do
    case $1 in
        --judge-root) JUDGE_ROOT="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --models) MODELS="$2"; shift 2;;
        --min-delta) MIN_DELTA="$2"; shift 2;;
        --max-per-sample) MAX_PER_SAMPLE="$2"; shift 2;;
        --dataset-prefix) DATASET_PREFIX="$2"; shift 2;;
        *) shift;;
    esac
done

if [[ -z "$MODELS" ]]; then
    echo "Error: --models is required."
    echo "Example: --models 'Qwen3-8B:Qwen3-8B.jsonl,Qwen3-32B:Qwen3-32B.jsonl'"
    exit 1
fi

python -m assay.training.build_dpo_pairs \
    --judge_results_root "$JUDGE_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --models "$MODELS" \
    --min_score_delta "$MIN_DELTA" \
    --max_pairs_per_sample "$MAX_PER_SAMPLE" \
    --dataset_prefix "$DATASET_PREFIX"
