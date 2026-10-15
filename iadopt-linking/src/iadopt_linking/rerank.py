"""The baseline reranker: the same Qwen3-Reranker-0.6B and prompt the legacy pipeline uses.

The prompt text matches ``misc/benchmarking_example/phaseOneThreeMerged.py`` verbatim (a
contract test compares them). Only the description cap is new: Wikidata descriptions are short,
QUDT's can run to 1,500 characters, and one cap keeps the documents comparable.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

MODEL_NAME = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
DEFAULT_MAX_DESCRIPTION_CHARS = 300

QWEN3_RERANK_PREFIX = (
    "<|im_start|>system\n"
    " Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
QWEN3_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_RERANK_TASK = "Given a web search query, retrieve relevant passages that answer the query"


class ScoreError(ValueError):
    """A scorer returned something that cannot be a set of scores."""


class Scorer(Protocol):
    """Anything that scores documents against one query."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Score each document for the query.

        Args: query: the plain query text; documents: plain document texts.
        Returns: one score per document, in order.
        Raises: implementation-specific.
        Side effects: implementation-specific.
        """


def build_query(term: str, definition: str) -> str:
    """Form the plain query the legacy pipeline sends.

    Args: term: the slot text; definition: the variable's full definition.
    Returns: ``Definition of "<term>" in context: "<definition>"``.
    Raises: None.
    Side effects: none.
    """
    return f'Definition of "{term}" in context: "{definition}"'


def truncate_description(text: str, max_chars: int | None) -> str:
    """Cap a description at a word boundary.

    Args: text: description; max_chars: cap, or None for no cap.
    Returns: the text unchanged if short enough, else its first words plus an ellipsis.
    Raises: None.
    Side effects: none.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:.") + "…"


def build_document(label: str, description: str, max_chars: int | None) -> str:
    """Form the plain candidate document the legacy pipeline sends.

    Args: label: candidate label; description: candidate description; max_chars: description cap.
    Returns: ``label: "<label>", description: "<description>"``.
    Raises: None.
    Side effects: none.
    """
    return f'label: "{label}", description: "{truncate_description(description, max_chars)}"'


def format_query(query: str, task: str = DEFAULT_RERANK_TASK) -> str:
    """Wrap a plain query in the Qwen3 reranker template.

    Args: query: plain query; task: instruction line.
    Returns: the templated query text.
    Raises: None.
    Side effects: none.
    """
    return f"{QWEN3_RERANK_PREFIX}<Instruct>: {task}\n<Query>: {query}\n"


def format_document(document: str) -> str:
    """Wrap a plain document in the Qwen3 reranker template.

    Args: document: plain document.
    Returns: the templated document text.
    Raises: None.
    Side effects: none.
    """
    return f"<Document>: {document}{QWEN3_RERANK_SUFFIX}"


def validate_scores(scores: Sequence[float], expected: int) -> list[float]:
    """Refuse scores that cannot be right.

    Args: scores: model output; expected: how many documents were sent.
    Returns: the scores as floats.
    Raises: ScoreError for a wrong count, a non-finite score or one outside [0, 1].
    Side effects: none.
    """
    values = [float(s) for s in scores]
    if len(values) != expected:
        raise ScoreError(f"Scorer returned {len(values)} scores for {expected} documents")
    if any(not math.isfinite(v) or v < 0.0 or v > 1.0 for v in values):
        raise ScoreError(f"Scorer returned a score outside [0, 1]: {values}")
    return values


class Qwen3Reranker:
    """Qwen3-Reranker-0.6B cross-encoder, loaded from the local Hugging Face cache only."""

    def __init__(self, device: str | None = None, batch_size: int = 16) -> None:
        """Load the cached model; fail rather than download.

        Args: device: "mps", "cpu" or None to prefer mps; batch_size: pairs per forward pass.
        Returns: None.
        Raises: the loader's error when the weights are not in the local cache.
        Side effects: sets HF_HUB_OFFLINE=1; loads about 2.4 GB of weights.
        """
        os.environ["HF_HUB_OFFLINE"] = "1"
        import torch
        from huggingface_hub import snapshot_download
        from sentence_transformers import CrossEncoder

        self.revision = Path(snapshot_download(MODEL_NAME, local_files_only=True)).name
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.batch_size = batch_size
        self._model = CrossEncoder(MODEL_NAME, device=self.device, local_files_only=True)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Score each document for the query.

        Args: query: plain query text; documents: plain document texts.
        Returns: one probability-like score in [0, 1] per document.
        Raises: ValueError if the model returns an impossible score.
        Side effects: model inference.
        """
        if not documents:
            return []
        pairs = [[format_query(query), format_document(d)] for d in documents]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return validate_scores(list(scores), len(documents))

    def describe(self) -> dict[str, Any]:
        """Identify the model and runtime for the run manifest.

        Args: none.
        Returns: model name, revision hash, device, batch size and library versions.
        Raises: None.
        Side effects: none.
        """
        import sentence_transformers
        import torch
        import transformers

        return {"model": MODEL_NAME, "revision": self.revision, "device": self.device,
                "batch_size": self.batch_size, "torch": torch.__version__,
                "sentence_transformers": sentence_transformers.__version__,
                "transformers": transformers.__version__}
