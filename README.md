# ASSAY: Reconciling Stability and Validity in Reference-Free Scientific LLM Judges

![EMNLP 2026](https://img.shields.io/badge/EMNLP%202026-Findings-b31b1b.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)

Official implementation of **ASSAY** (Automated Scientific Synthesis Assessment of Your Judges), a probe suite that audits the faithfulness of reference-free scientific LLM judges by augmenting ground-truth recipes along two complementary axes.

## 🔥 News
- **[2026.08]** ASSAY has been accepted to **EMNLP 2026 Findings**! 🎉

## Overview

Reference-free judges fail in two opposite directions. They accept scientifically invalid recipes (**false positives**) and they penalize semantically equivalent ones written in unfamiliar notation (**false negatives**). ASSAY measures both by rewriting a ground-truth recipe `y*` with eight swap types and comparing the judge's scores against its own anchor score `J(x, y*)`.

- **Error Injection** (ỹ) introduces scientifically invalid modifications to probe **Validity**
- **Equivalence Rewriting** (ŷ) substitutes semantically equivalent representations to probe **Stability**
- Across thirteen judge models the two metrics form a systematic **Stability–Validity trade-off** (*r* = −0.62)

With τ = 0.5 on the 1–5 rubric and ρ ∈ {1%, 2%, 5%, 10%, 15%}:

| Metric | Definition |
|--------|------------|
| **Validity** (↑) | AUC over ρ of the rate at which ỹ is *detected*, i.e. `J(x,y*) − J(x,ỹ) > τ` |
| **Stability** (↑) | AUC over ρ of the rate at which ŷ is *stable*, i.e. `abs(J(x,y*) − J(x,ŷ)) ≤ τ` |
| **Accuracy** (↑) | AUC over ρ of the combined correctness rate pooled across ỹ and ŷ |

## Swap Types

| Axis | Swap Type | Operation | Example |
|------|-----------|-----------|---------|
| Error Injection (ỹ) | `element_substitution` | Element → same-group element | Li → Na |
| | `numerical_perturbation` | Quantity → infeasible value | 850 °C → 1700 °C |
| | `equipment_substitution` | Equipment → incompatible alternative | furnace → hot plate |
| | `action_antonym` | Verb → antonym | heat → cool |
| Equivalence Rewriting (ŷ) | `llm_to_formula` | Compound name → molecular formula | ethanol → C₂H₆O |
| | `llm_to_name` | Molecular formula → common name | C₂H₆O → ethanol |
| | `llm_to_iupac` | Name or formula → IUPAC name | acetone → propan-2-one |
| | `cross_lingual` | Full recipe → another language | "Dissolve…" → "dissoudre…" |

`all_error_injection` and `all_equivalence_rewriting` apply the four types of each axis in a single pass; these are the settings used for the ρ sweep.

## Installation

```bash
git clone https://github.com/SoyeonHH/ASSAY.git
cd ASSAY

conda create -n assay python=3.11
conda activate assay

pip install -r requirements.txt
export OPENROUTER_API_KEY=your_key_here
```

Judge models are served through [OpenRouter](https://openrouter.ai). The augmentation operators call GPT-4.1-mini for entity recognition and for chemical conversion when PubChem returns no result. Every call is cached on disk, so repeated runs over the same recipes issue no further requests.

## Quick Start

The commands below run on `examples/sample_input.jsonl` (3 records) and are the ones used for the paper, differing only in the input file. The entity-recognition cache for these recipes is committed, so Error Injection and steps 2 and 4 run without an API key.

### 1. Augment recipes

```bash
# Error Injection (ỹ)
python -m assay.augmentation.run \
    --input_file examples/sample_input.jsonl \
    --type all_error_injection \
    --rates '[0.01, 0.02, 0.05, 0.10, 0.15]' \
    --output_dir outputs/augmentation

# Equivalence Rewriting (ŷ)
python -m assay.augmentation.run \
    --input_file examples/sample_input.jsonl \
    --type all_equivalence_rewriting \
    --rates '[0.01, 0.02, 0.05, 0.10, 0.15]' \
    --output_dir outputs/augmentation
```

Each call writes `outputs/augmentation/sample_input_<type>_<rate>pct.jsonl`, with the augmented recipe in the `prediction` field and the applied swaps under `perturbation_meta`. Both axes plus the per-swap-type ablations at ρ = 100% can be produced in one pass:

```bash
./scripts/run_augmentation.sh --input examples/sample_input.jsonl
```

### 2. Build the anchor input

Validity and Stability are measured against the anchor score `J(x, y*)`, the judge's own score on the unmodified ground-truth recipe:

```bash
./scripts/make_anchor_input.sh \
    examples/sample_input.jsonl \
    outputs/augmentation/sample_input.jsonl
```

### 3. Run judge evaluation

```bash
./scripts/run_judge.sh \
    --model openai/gpt-4o \
    --concurrency 4 \
    --input outputs/augmentation/*.jsonl
```

Results land in `judge_results/<input stem>/<model>.jsonl`, one directory per condition. Interrupted runs resume by counting the records already written. The module can also be called directly:

```bash
python -m assay.judge.evaluate \
    outputs/augmentation/sample_input_all_error_injection_5pct.jsonl \
    --model openai/gpt-4o --reference_free true
```

### 4. Compute ASSAY metrics

```bash
./scripts/run_metrics.sh --prefix sample_input
```

This reports Validity, Stability, and Accuracy for every model found in `judge_results/sample_input/` and writes `results/metrics/<model>_metrics.json` with the per-rate breakdown. `--prefix` is the stem of the input file, which locates the anchor directory and the augmented conditions.

## Knowledge Augmentation Baselines

Three inference-time baselines supply domain knowledge to an unmodified judge. Each prepends a context block to the judge prompt; no training is involved.

| Mode | Context added |
|------|---------------|
| `chem_dict` | A chemical reference dictionary built from the candidate recipe, resolved into compounds (formula / IUPAC / common name), elements (periodic-group membership), equipment (incompatible alternatives), and actions (antonym verbs) |
| `rag` | The five most similar training recipes retrieved by cosine similarity over `text-embedding-3-large` embeddings of the target material description |
| `chem_dict,rag` | The dictionary followed by the retrieved recipes |

```bash
python -m assay.judge.evaluate <augmented.jsonl> \
    --model openai/gpt-4o --reference_free true \
    --augment chem_dict

python -m assay.judge.evaluate <augmented.jsonl> \
    --model openai/gpt-4o --reference_free true \
    --augment chem_dict,rag --rag_corpus <train.parquet> --rag_top_k 5
```

The RAG corpus parquet needs `contribution`, `recipe`, and `contributions_embedding` columns; no ground-truth recipe for the evaluated target is ever supplied. The mode is appended to the output filename (`gpt-4o_chem_dict.jsonl`), so variants can be scored side by side with `assay.metrics.run`.

The `chem_dict` compound section reads `pubchem_cache.json` and `llm_convert_cache.json` from the active data directory. Those caches are written by the Equivalence Rewriting operators, so run step 1 on a dataset before using ChemDict on it; unresolved entities are skipped rather than looked up live.

## Biology Configuration

The augmentation logic, the rate schedule, and the metric definitions are domain-independent. Transferring ASSAY to a new field needs only a swap dictionary directory and a judge prompt. `assay/augmentation/data_bio/` holds the configuration used for the biological protocol experiment.

| File | Content |
|------|---------|
| `periodic_table.json` | 91 reagents across 17 groups (buffers, detergents, antibiotics, polymerases, proteases, restriction enzymes, media, …) replacing the periodic-group table |
| `equipment_rules.json` | 64 substitutions over biology laboratory instruments |
| `action_antonyms.json` | 40 protocol verb pairs (ligate ↔ digest, denature ↔ anneal, …) |
| `error_ner_prompt.txt` | Entity-recognition prompt remapping the four entity types to the biological setting |

Pass the directory with `--data_dir`; the swap dictionaries, the recognition prompt, and its cache are all read from it.

```bash
python -m assay.augmentation.run \
    --input_file data/protocols.jsonl \
    --type all_error_injection \
    --rates '[0.01, 0.02, 0.05, 0.10, 0.15]' \
    --data_dir assay/augmentation/data_bio

python -m assay.judge.evaluate <augmented.jsonl> \
    --model openai/gpt-5-mini \
    --prompt_file assay/judge/prompt_bio.txt
```

`prompt_bio.txt` rephrases the rubric for protocol evaluation. Unlike the materials science prompt it is reference-based and scores eight criteria, which is the configuration the reported biology results were produced with. Equivalence Rewriting is unchanged across domains, since PubChem resolution applies to reagents as well.

## Contrastive Pair Construction

Preference pairs are built from disagreement between judges on the same augmented recipe. For Error Injection the lower score is preferred; for Equivalence Rewriting the higher score is preferred. Both directions are filtered against a per-sample anchor score.

```bash
python -m assay.training.build_dpo_pairs \
    --judge_results_root judge_results \
    --dataset_prefix sample_input \
    --models "Qwen3-8B:Qwen3-8B.jsonl,Qwen3-32B:Qwen3-32B.jsonl" \
    --output_dir dpo_dataset
```

This reads the same `judge_results/<prefix>_<condition>/<model>.jsonl` layout the judge writes and emits `train.parquet` and `validation.parquet` in the chat-formatted `prompt` / `chosen` / `rejected` schema. The paper uses a four-judge panel over 17 augmentation settings, keeps pairs with a score margin of at least 0.5, and retains at most five pairs per recipe per setting.

## Project Structure

```
ASSAY/
├── assay/
│   ├── domain.py                       # Active data directory (materials or biology)
│   ├── ner/
│   │   ├── chemical_ner.py             # Chemical entity recognition, SHA-256 cached
│   │   └── error_ner.py                # Element / numerical / equipment / action recognition
│   ├── augmentation/
│   │   ├── error_injection.py          # Error Injection (ỹ) swap types
│   │   ├── equivalence_rewriting.py    # Equivalence Rewriting (ŷ) swap types
│   │   ├── combined.py                 # All four Error Injection types in one pass
│   │   ├── rate_controller.py          # Token-level augmentation rate ρ
│   │   ├── pubchem.py                  # Rate-limited PubChem client with cache
│   │   ├── llm_convert.py              # LLM fallback for PubChem-unresolvable entities
│   │   ├── run.py                      # CLI entry point
│   │   ├── data/                       # Materials science swap dictionaries
│   │   └── data_bio/                   # Biology swap dictionaries
│   ├── judge/
│   │   ├── evaluate.py                 # LLM-as-a-Judge evaluation
│   │   ├── score_parser.py             # Score extraction from judge output
│   │   ├── prompt.txt                  # Reference-free judge prompt
│   │   ├── prompt_bio.txt              # Biological protocol judge prompt
│   │   └── knowledge/                  # ChemDict and RAG baselines
│   ├── metrics/                        # Validity, Stability, Accuracy
│   └── training/                       # Contrastive pair construction
├── scripts/
│   ├── run_augmentation.sh             # Both axes plus per-type ablations
│   ├── make_anchor_input.sh            # Anchor input for J(x, y*)
│   ├── run_judge.sh                    # Judge evaluation over one or more files
│   ├── run_metrics.sh                  # Validity / Stability / Accuracy
│   └── run_dpo_build.sh                # Contrastive pair construction
├── examples/sample_input.jsonl
└── requirements.txt
```

## Data

Input JSONL records:

```json
{
    "id": 1,
    "sample_id": 1,
    "contribution": "Target material description...",
    "recipe": "Ground truth synthesis recipe...",
    "Material_Name": "TiO2",
    "process": "hydrothermal",
    "domain": "photocatalysis"
}
```

Each stage appends fields. Augmentation adds `prediction` and `perturbation_meta`; the judge adds `judge_result` and `judge_model`.

| Dataset | Role | Size |
|---------|------|------|
| AlchemyBench (high-impact test) | Judge evaluation | 169 recipes |
| AlchemyBench (train) | RAG corpus, contrastive pairs | 16,026 recipes |
| BioProBench | Biology transfer | 200 protocols |
