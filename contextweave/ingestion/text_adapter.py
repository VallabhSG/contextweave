"""Adapter for plain text and markdown files."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from contextweave.ingestion.base import BaseAdapter
from contextweave.schemas import ContextEvent, SourceType


class TextAdapter(BaseAdapter):
    """Ingests plain text and markdown files as context events."""

    @property
    def supported_formats(self) -> list[str]:
        return [".txt", ".md", ".markdown"]

    def ingest_file(self, path: Path) -> list[ContextEvent]:
        content = path.read_text(encoding="utf-8", errors="replace")
        stat = path.stat()
        timestamp = datetime.fromtimestamp(stat.st_mtime)

        return self.ingest_text(
            content,
            metadata={
                "filename": path.name,
                "path": str(path),
                "size_bytes": stat.st_size,
            },
            timestamp=timestamp,
        )

    def ingest_text(
        self,
        raw: str,
        metadata: dict | None = None,
        timestamp: datetime | None = None,
    ) -> list[ContextEvent]:
        if not raw.strip():
            return []

        meta = metadata or {}
        ts = timestamp or datetime.utcnow()

        # Split markdown by H1/H2 headings into logical sections
        sections = self._split_sections(raw)

        events = []
        for section in sections:
            if not section["content"].strip():
                continue
            events.append(
                ContextEvent(
                    source=SourceType.NOTE,
                    content=section["content"],
                    timestamp=ts,
                    metadata={**meta, "heading": section.get("heading", "")},
                    raw_format="markdown" if meta.get("filename", "").endswith(".md") else "text",
                )
            )

        return events if events else [
            ContextEvent(
                source=SourceType.NOTE,
                content=raw,
                timestamp=ts,
                metadata=meta,
                raw_format="text",
            )
        ]

    def _split_sections(self, text: str) -> list[dict]:
        """Split markdown by headings into logical sections."""
        heading_pattern = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(text))

        if not matches:
            return [{"content": text, "heading": ""}]

        sections = []

        # Content before first heading
        pre = text[: matches[0].start()].strip()
        if pre:
            sections.append({"content": pre, "heading": ""})

        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            sections.append({"content": content, "heading": match.group(2)})

        return sections
