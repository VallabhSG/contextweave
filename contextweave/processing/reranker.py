"""Optional local cross-encoder reranking.

Fused retrieval (vector + FTS + graph) is fast but scores query and memory
independently. A cross-encoder reads the (query, memory) pair together and is a
much sharper relevance judge — so we let it reorder the top fused candidates
before they reach the LLM. It runs fully locally via fastembed's ONNX rerankers
(no API, no new dependency), and is off unless ``CW_RERANK_MODEL`` is set.
"""

from __future__ import annotations

import logging

from contextweave.config import settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Wraps a fastembed ONNX cross-encoder; loads the model on first use."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Relevance score per document for the query (higher = more relevant)."""
        return list(self._get_model().rerank(query, documents))


# Cache one instance per model name so the ONNX weights load only once.
_cache: dict[str, CrossEncoderReranker] = {}


def get_reranker() -> CrossEncoderReranker | None:
    """The configured reranker, or None when reranking is disabled/unavailable."""
    name = settings.rerank_model
    if not name:
        return None
    if name not in _cache:
        try:
            _cache[name] = CrossEncoderReranker(name)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Reranker unavailable (%s); continuing without it: %s", name, e)
            return None
    return _cache[name]
