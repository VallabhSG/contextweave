"""ChromaDB vector store for semantic search over chunks."""

from __future__ import annotations

import logging

import chromadb
from chromadb.config import Settings as ChromaSettings

from contextweave.config import settings
from contextweave.schemas import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Wraps ChromaDB for storing and querying chunk embeddings."""

    def __init__(self, persist_dir: str | None = None, collection_name: str = "chunks"):
        self._persist_dir = persist_dir or settings.chroma_persist_dir
        self._collection_name = collection_name
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            self._client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Store chunks with their embeddings. Returns count added."""
        collection = self._get_collection()
        added = 0

        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning("Skipping chunk %s — no embedding", chunk.id)
                continue

            try:
                collection.upsert(
                    ids=[chunk.id],
                    embeddings=[chunk.embedding],
                    documents=[chunk.content],
                    metadatas=[
                        {
                            "event_id": chunk.event_id,
                            "source": chunk.source.value,
                            "timestamp": chunk.timestamp.isoformat(),
                            "entities": ",".join(chunk.entities),
                        }
                    ],
                )
                added += 1
            except Exception as e:
                if "dimensionality" in str(e).lower() or "dimension" in str(e).lower():
                    logger.warning("Dimension mismatch — recreating collection: %s", e)
                    self._reset_collection()
                    collection = self._get_collection()
                    collection.upsert(
                        ids=[chunk.id],
                        embeddings=[chunk.embedding],
                        documents=[chunk.content],
                        metadatas=[
                            {
                                "event_id": chunk.event_id,
                                "source": chunk.source.value,
                                "timestamp": chunk.timestamp.isoformat(),
                                "entities": ",".join(chunk.entities),
                            }
                        ],
                    )
                    added += 1
                else:
                    logger.error("Failed to upsert chunk %s: %s", chunk.id, e)

        return added

    def _reset_collection(self) -> None:
        """Delete and recreate the collection (handles embedding dimension changes)."""
        if self._client is not None:
            try:
                self._client.delete_collection(self._collection_name)
            except Exception:
                pass
        self._collection = None
        self._client = None

    def query(
        self,
        embedding: list[float],
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[dict]:
        """Query similar chunks by embedding vector."""
        collection = self._get_collection()
        k = top_k or settings.retrieval_top_k

        kwargs = {
            "query_embeddings": [embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)

        items = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                items.append(
                    {
                        "chunk_id": chunk_id,
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "score": 1.0 - results["distances"][0][i],  # cosine: distance=1-similarity
                    }
                )

        return items

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Remove chunks by ID."""
        collection = self._get_collection()
        collection.delete(ids=chunk_ids)

    def count(self) -> int:
        """Total chunks stored."""
        return self._get_collection().count()
