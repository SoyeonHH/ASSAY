#!/bin/bash
# Run two-axis augmentation on an input JSONL.
#
# Usage:
#   ./scripts/run_augmentation.sh --input examples/sample_input.jsonl
#   ./scripts/run_augmentation.sh --input data/test.jsonl --rates '[0.01, 0.02, 0.05, 0.10, 0.15]'
#   ./scripts/run_augmentation.sh --input data/protocols.jsonl --data-dir assay/augmentation/data_bio

set -euo pipefail

INPUT=""
ERROR_RATES='[0.01, 0.02, 0.05, 0.10, 0.15]'
REPR_RATES='[0.01, 0.02, 0.05, 0.10, 0.15]'
OUTPUT_DIR="outputs/augmentation"
DATA_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --input) INPUT="$2"; shift 2;;
        --rates) ERROR_RATES="$2"; REPR_RATES="$2"; shift 2;;
        --error-rates) ERROR_RATES="$2"; shift 2;;
        --repr-rates) REPR_RATES="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --data-dir) DATA_DIR="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

if [[ -z "$INPUT" ]]; then
    echo "Usage: ./scripts/run_augmentation.sh --input <file.jsonl> [--rates '[...]'] [--data-dir <dir>]"
    exit 1
fi

DATA_ARG=()
if [[ -n "$DATA_DIR" ]]; then
    DATA_ARG=(--data_dir "$DATA_DIR")
fi

echo "=== Error Injection (combined) ==="
python -m assay.augmentation.run \
    --input_file "$INPUT" \
    --type all_error_injection \
    --rates "$ERROR_RATES" \
    --output_dir "$OUTPUT_DIR" "${DATA_ARG[@]}"

echo ""
echo "=== Equivalence Rewriting (combined) ==="
python -m assay.augmentation.run \
    --input_file "$INPUT" \
    --type all_equivalence_rewriting \
    --rates "$REPR_RATES" \
    --output_dir "$OUTPUT_DIR" "${DATA_ARG[@]}"

echo ""
echo "=== Per-swap-type ablation (100%) ==="
for TYPE in element_substitution numerical_perturbation equipment_substitution action_antonym \
            llm_to_formula llm_to_name llm_to_iupac cross_lingual; do
    python -m assay.augmentation.run \
        --input_file "$INPUT" \
        --type "$TYPE" \
        --rate 1.0 \
        --output_dir "$OUTPUT_DIR" "${DATA_ARG[@]}"
done

echo ""
echo "Done. Outputs in $OUTPUT_DIR"
