#!/bin/bash
# Run LLM-as-a-Judge evaluation on augmented outputs.
#
# Usage:
#   ./scripts/run_judge.sh --model openai/gpt-4o \
#       --input outputs/augmentation/sample_input_all_error_injection_5pct.jsonl
#   ./scripts/run_judge.sh --model openai/gpt-4o --augment chem_dict \
#       --input outputs/augmentation/*.jsonl

set -euo pipefail

MODEL="openai/gpt-4o"
REF_FREE="true"
CONCURRENCY=1
MAX_SAMPLES=()
PROMPT=()
AUGMENT=()
FILES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2;;
        --reference-based) REF_FREE="false"; shift;;
        --prompt-file) PROMPT=(--prompt_file "$2"); shift 2;;
        --concurrency) CONCURRENCY="$2"; shift 2;;
        --max-samples) MAX_SAMPLES=(--max_samples "$2"); shift 2;;
        --augment) AUGMENT+=(--augment "$2"); shift 2;;
        --rag-corpus) AUGMENT+=(--rag_corpus "$2"); shift 2;;
        --rag-top-k) AUGMENT+=(--rag_top_k "$2"); shift 2;;
        --input) shift; while [[ $# -gt 0 && ! "$1" == --* ]]; do FILES+=("$1"); shift; done;;
        *) FILES+=("$1"); shift;;
    esac
done

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "Usage: ./scripts/run_judge.sh --input <file.jsonl> [--model <model>] [--augment chem_dict]"
    exit 1
fi

for FILE in "${FILES[@]}"; do
    echo "=== Judging: $FILE with $MODEL ==="
    python -m assay.judge.evaluate \
        "$FILE" \
        --model "$MODEL" \
        --reference_free "$REF_FREE" \
        --concurrency "$CONCURRENCY" \
        "${PROMPT[@]}" "${MAX_SAMPLES[@]}" "${AUGMENT[@]}"
    echo ""
done

echo "Done."
