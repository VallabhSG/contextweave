"""Embedding service using fastembed (local, no API key required)."""

from __future__ import annotations

import logging

from contextweave.config import settings
from contextweave.schemas import Chunk

logger = logging.getLogger(__name__)


class GeminiEmbedder:
    """Local embedder using fastembed (BAAI/bge-small-en-v1.5, 384-dim)."""

    def __init__(self, model: str | None = None, **_kwargs):
        self._model_name = model or settings.embedding_model
        self._model = None

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        model = self._get_model()
        embeddings = list(model.embed([text]))
        return embeddings[0].tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a query string."""
        return self.embed_text(query)

    def embed_chunks(self, chunks: list[Chunk], batch_size: int = 50) -> list[Chunk]:
        """Embed chunks in batches, returning new Chunk objects with embeddings."""
        if not chunks:
            return []

        model = self._get_model()
        texts = [c.content for c in chunks]
        embedded = []

        try:
            embeddings = list(model.embed(texts))
            for chunk, emb in zip(chunks, embeddings):
                embedded.append(chunk.model_copy(update={"embedding": emb.tolist()}))
        except Exception as e:
            logger.error("Batch embedding failed: %s", e)
            for chunk in chunks:
                try:
                    emb = self.embed_text(chunk.content)
                    embedded.append(chunk.model_copy(update={"embedding": emb}))
                except Exception as inner_e:
                    logger.error("Single embed failed for chunk %s: %s", chunk.id, inner_e)
                    embedded.append(chunk)

        return embedded
