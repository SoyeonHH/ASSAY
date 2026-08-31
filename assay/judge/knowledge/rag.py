"""Retrieval for the RAG knowledge augmentation baseline.

Retrieves the top-K most similar recipes from a training corpus by cosine
similarity over text-embedding-3-large embeddings of the target material
description. Retrieved recipes are prepended to the judge prompt as in-context
references; no ground-truth recipe for the evaluated target is ever supplied.

The corpus parquet must provide ``contribution``, ``recipe``, and a
``contributions_embedding`` column holding one embedding vector per row.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from litellm import embedding as litellm_embedding


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Set it with: export OPENROUTER_API_KEY=your_key"
        )
    return key


class RAGRetriever:
    """Retrieve similar recipes from a training corpus via embedding similarity."""

    def __init__(
        self,
        corpus_path: str,
        top_k: int = 5,
        cache_path: str | Path = "outputs/rag_query_embeddings.json",
        embedding_model: str = "openai/text-embedding-3-large",
    ):
        self.top_k = top_k
        self.cache_path = Path(cache_path)
        self.embedding_model = embedding_model
        self._emb_cache: dict[str, list[float]] = self._load_cache()

        df = pd.read_parquet(corpus_path)
        self._contributions = df["contribution"].tolist()
        self._recipes = df["recipe"].tolist()

        matrix = np.vstack(df["contributions_embedding"].tolist()).astype(np.float32)
        norms = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-10)
        self._emb_matrix = matrix / norms

        print(
            f"RAGRetriever: {self._emb_matrix.shape[0]} recipes from {corpus_path}, "
            f"{len(self._emb_cache)} cached query embeddings"
        )

    def _load_cache(self) -> dict[str, list[float]]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._emb_cache, f)

    def _get_embedding(self, text: str) -> np.ndarray:
        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if cache_key in self._emb_cache:
            return np.array(self._emb_cache[cache_key], dtype=np.float32)

        os.environ["OPENROUTER_API_KEY"] = _get_api_key()
        response = litellm_embedding(
            model=f"openrouter/{self.embedding_model}", input=[text],
        )
        vec = response["data"][0]["embedding"]

        self._emb_cache[cache_key] = vec
        self._save_cache()
        return np.array(vec, dtype=np.float32)

    def build_context(self, contribution_text: str, max_recipe_chars: int = 2000) -> str:
        """Return the top-K retrieved recipes formatted as in-context references."""
        query_vec = self._get_embedding(contribution_text)
        norm = np.linalg.norm(query_vec)
        if norm > 1e-10:
            query_vec = query_vec / norm

        similarities = self._emb_matrix @ query_vec
        top_indices = np.argsort(similarities)[::-1][: self.top_k]

        parts = ["Reference Recipes from Similar Materials:"]
        for rank, idx in enumerate(top_indices, 1):
            recipe = self._recipes[idx]
            if len(recipe) > max_recipe_chars:
                recipe = recipe[:max_recipe_chars] + " ..."
            parts.append(f"\n[Reference {rank}] (similarity: {float(similarities[idx]):.3f})")
            parts.append(f"Target Material:\n{self._contributions[idx]}")
            parts.append(f"\nRecipe:\n{recipe}")

        return "\n".join(parts)
