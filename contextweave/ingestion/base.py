"""Base adapter interface for context ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from contextweave.schemas import ContextEvent


class BaseAdapter(ABC):
    """Abstract base for all ingestion adapters."""

    @abstractmethod
    def ingest_file(self, path: Path) -> list[ContextEvent]:
        """Parse a file and return normalized ContextEvents."""

    @abstractmethod
    def ingest_text(self, raw: str, metadata: dict | None = None) -> list[ContextEvent]:
        """Parse raw text and return normalized ContextEvents."""

    @property
    @abstractmethod
    def supported_formats(self) -> list[str]:
        """File extensions this adapter handles."""
