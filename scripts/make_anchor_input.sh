#!/bin/bash
# Build the anchor input for a dataset.
#
# The anchor score J(x, y*) is the judge's reference-free score on the
# ground-truth recipe itself, so the recipe is copied into the prediction
# field that the judge reads. Validity and Stability are both measured
# against this score.
#
# Usage:
#   ./scripts/make_anchor_input.sh examples/sample_input.jsonl outputs/augmentation/sample_input.jsonl

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: ./scripts/make_anchor_input.sh <input.jsonl> <output.jsonl>"
    exit 1
fi

python - "$1" "$2" <<'EOF'
import json
import sys
from pathlib import Path

src, dst = sys.argv[1], sys.argv[2]
Path(dst).parent.mkdir(parents=True, exist_ok=True)

n = 0
with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rec["prediction"] = rec["recipe"]
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1

print(f"Created {dst} ({n} records)")
EOF
