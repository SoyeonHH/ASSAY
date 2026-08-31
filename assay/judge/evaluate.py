"""LLM-as-a-Judge evaluation via OpenRouter API.

Usage:
    python -m assay.judge.evaluate input.jsonl --model openai/gpt-4o
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import fire
import jsonlines
from litellm import completion, acompletion
from tqdm import tqdm

_PROMPT_DIR = Path(__file__).resolve().parent
_PROMPT_FILE = _PROMPT_DIR / "prompt.txt"

USER_PROMPT = """
Please evaluate the following:

Target Material:
{objective}

AI-Generated Recipe:
{prediction}

Ground Truth Recipe:
{gt_recipe}"""

USER_PROMPT_REF_FREE = """
Please evaluate the following:

Target Material:
{objective}

AI-Generated Recipe:
{prediction}"""


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Set it with: export OPENROUTER_API_KEY=your_key"
        )
    return key


def _strip_thinking_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class RecipeJudge:
    """Judge for evaluating recipe predictions using OpenRouter API."""

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        reference_free: bool = False,
        prompt_file: str | Path | None = None,
        augment_fn=None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reference_free = reference_free
        self.augment_fn = augment_fn
        self.system_prompt = Path(prompt_file or _PROMPT_FILE).read_text()

        os.environ["OPENROUTER_API_KEY"] = _get_api_key()

    def _build_user_content(self, item: dict) -> str:
        if self.reference_free:
            user_content = USER_PROMPT_REF_FREE.format(
                objective=item["contribution"],
                prediction=item["prediction"],
            )
        else:
            user_content = USER_PROMPT.format(
                objective=item["contribution"],
                prediction=item["prediction"],
                gt_recipe=item["recipe"],
            )

        if self.augment_fn:
            extra = self.augment_fn(item)
            if extra:
                user_content = extra + "\n\n" + user_content

        return user_content

    def _judge_openrouter(self, item: dict) -> str:
        model = self.model
        if not model.startswith("openrouter/"):
            model = f"openrouter/{model}"

        user_content = self._build_user_content(item)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = completion(
            model=model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        text = response["choices"][0]["message"]["content"]
        return _strip_thinking_tags(text)

    async def _judge_openrouter_async(self, item: dict) -> str:
        model = self.model
        if not model.startswith("openrouter/"):
            model = f"openrouter/{model}"

        user_content = self._build_user_content(item)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = await acompletion(
            model=model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        text = response["choices"][0]["message"]["content"]
        return _strip_thinking_tags(text)

    def judge(self, item: dict) -> str:
        return self._judge_openrouter(item)


def main(
    filename: str,
    model: str = "openai/gpt-4o",
    reference_free: bool = False,
    max_samples: int | None = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    concurrency: int = 1,
    output_dir: str | None = None,
    prompt_file: str | None = None,
    augment: str = "",
    rag_corpus: str = "",
    rag_top_k: int = 5,
):
    """Run recipe evaluation using LLM-as-a-Judge.

    Args:
        filename: Input JSONL file with predictions
        model: Model name (OpenRouter format, e.g. openai/gpt-4o)
        reference_free: Evaluate without ground truth reference
        max_samples: Maximum number of samples to evaluate
        temperature: Sampling temperature
        max_tokens: Maximum output tokens
        concurrency: Number of concurrent API requests (default 1)
        output_dir: Output directory (default: judge_results/)
        prompt_file: Judge prompt file (default: assay/judge/prompt.txt;
            use assay/judge/prompt_bio.txt for biological protocols)
        augment: Knowledge augmentation applied at inference time. One of
            "chem_dict", "rag", or "chem_dict,rag" (empty disables it).
        rag_corpus: Parquet corpus for RAG retrieval. Requires columns
            contribution, recipe, and contributions_embedding.
        rag_top_k: Number of recipes retrieved per query (default 5)
    """
    if isinstance(reference_free, str):
        reference_free = reference_free.lower() in ('true', '1', 'yes')

    augment_modes = [m.strip() for m in str(augment).split(",") if m.strip()]
    augment_fn = None
    if augment_modes:
        builders = []
        if "chem_dict" in augment_modes:
            from assay.judge.knowledge.chem_dict import ChemDictBuilder
            builders.append(("chem_dict", ChemDictBuilder()))
            print("Knowledge augmentation: ChemDict enabled")
        if "rag" in augment_modes:
            if not rag_corpus:
                raise ValueError("--rag_corpus is required when --augment includes 'rag'")
            from assay.judge.knowledge.rag import RAGRetriever
            builders.append(("rag", RAGRetriever(rag_corpus, top_k=rag_top_k)))
            print(f"Knowledge augmentation: RAG enabled (top_k={rag_top_k})")
        unknown = set(augment_modes) - {"chem_dict", "rag"}
        if unknown:
            raise ValueError(f"Unknown augmentation mode(s): {sorted(unknown)}")

        def _build_augment_context(item, _builders=builders):
            parts = []
            for mode, builder in _builders:
                key = "prediction" if mode == "chem_dict" else "contribution"
                ctx = builder.build_context(item.get(key, ""))
                if ctx:
                    parts.append(ctx)
            return "\n\n".join(parts)

        augment_fn = _build_augment_context

    ds = list(jsonlines.open(filename))
    if max_samples is not None:
        ds = ds[:max_samples]

    judge = RecipeJudge(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        reference_free=reference_free,
        prompt_file=prompt_file,
        augment_fn=augment_fn,
    )

    model_name = model.split("/")[-1]
    if augment_modes:
        model_name = f"{model_name}_{'_'.join(augment_modes)}"

    if output_dir is None:
        output_dir = "judge_results"

    base_name = os.path.splitext(os.path.basename(filename))[0]
    out_dir = os.path.join(output_dir, base_name)
    os.makedirs(out_dir, exist_ok=True)

    output_filename = os.path.join(out_dir, f"{model_name}.jsonl")

    # Resume from existing progress
    total = len(ds)
    skip = 0
    if os.path.exists(output_filename):
        with open(output_filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pos = 0
                while pos < len(line):
                    try:
                        json.loads(line[pos:])
                        skip += 1
                        break
                    except json.JSONDecodeError as e:
                        if e.msg == "Extra data" and e.pos:
                            json.loads(line[pos:pos+e.pos])
                            skip += 1
                            pos = pos + e.pos
                        else:
                            break
        ds = ds[skip:]
        print(f"Skipping {skip} items, {len(ds)} remaining")

    if not ds:
        print("All items already processed")
        return

    pbar = tqdm(total=total, initial=skip, desc=f"Judging ({model_name})")

    if concurrency > 1:
        print(f"Using async concurrency: {concurrency}")

        async def _run_concurrent():
            sem = asyncio.Semaphore(concurrency)
            MAX_RETRIES = 3

            async def process_one(item):
                async with sem:
                    for attempt in range(MAX_RETRIES):
                        try:
                            return await judge._judge_openrouter_async(item)
                        except Exception as e:
                            if attempt < MAX_RETRIES - 1:
                                wait = 2 ** attempt
                                print(f"\n  Retry {attempt+1}/{MAX_RETRIES}: {e}\n  Waiting {wait}s...")
                                await asyncio.sleep(wait)
                            else:
                                raise

            for i in range(0, len(ds), concurrency):
                chunk = ds[i : i + concurrency]
                tasks = [process_one(item) for item in chunk]
                results = await asyncio.gather(*tasks)
                with jsonlines.open(output_filename, "a") as fout:
                    for item, result in zip(chunk, results):
                        item["judge_result"] = result
                        item["judge_model"] = model
                        fout.write(item)
                pbar.update(len(chunk))

        asyncio.run(_run_concurrent())
    else:
        with jsonlines.open(output_filename, "a") as fout:
            for item in ds:
                result = judge._judge_openrouter(item)
                item["judge_result"] = result
                item["judge_model"] = model
                fout.write(item)
                pbar.update(1)

    pbar.close()
    print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    fire.Fire(main)
