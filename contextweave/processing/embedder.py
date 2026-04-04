"""Embedding service using Google Gemini text-embedding-004."""

from __future__ import annotations

import logging

from contextweave.config import settings
from contextweave.schemas import Chunk

logger = logging.getLogger(__name__)


class GeminiEmbedder:
    """Wraps Google's Gemini embedding API for batch and single embedding."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or settings.gemini_api_key
        self._model = model or settings.embedding_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            self._client = genai
        return self._client

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        client = self._get_client()
        result = client.embed_content(
            model=self._model,
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]

    def embed_query(self, query: str) -> list[float]:
        """Embed a query string (uses retrieval_query task type)."""
        client = self._get_client()
        result = client.embed_content(
            model=self._model,
            content=query,
            task_type="retrieval_query",
        )
        return result["embedding"]

    def embed_chunks(self, chunks: list[Chunk], batch_size: int = 50) -> list[Chunk]:
        """Embed a list of chunks in batches, returning new Chunk objects with embeddings."""
        client = self._get_client()
        embedded = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]

            try:
                result = client.embed_content(
                    model=self._model,
                    content=texts,
                    task_type="retrieval_document",
                )
                embeddings = result["embedding"]

                for chunk, emb in zip(batch, embeddings):
                    embedded.append(chunk.model_copy(update={"embedding": emb}))

            except Exception as e:
                logger.error("Embedding batch %d failed: %s", i // batch_size, e)
                # Fall back to individual embedding
                for chunk in batch:
                    try:
                        emb = self.embed_text(chunk.content)
                        embedded.append(chunk.model_copy(update={"embedding": emb}))
                    except Exception as inner_e:
                        logger.error("Single embed failed for chunk %s: %s", chunk.id, inner_e)
                        embedded.append(chunk)

        return embedded
