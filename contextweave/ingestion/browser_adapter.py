"""Adapter for browser history JSON exports."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from contextweave.ingestion.base import BaseAdapter
from contextweave.schemas import ContextEvent, SourceType


class BrowserAdapter(BaseAdapter):
    """Ingests browser history exports (Chrome JSON format)."""

    @property
    def supported_formats(self) -> list[str]:
        return [".json"]

    def ingest_file(self, path: Path) -> list[ContextEvent]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        return self.ingest_text(raw, metadata={"filename": path.name, "path": str(path)})

    def ingest_text(self, raw: str, metadata: dict | None = None) -> list[ContextEvent]:
        meta = metadata or {}
        data = json.loads(raw)

        entries = data if isinstance(data, list) else data.get("Browser History", data.get("history", []))

        events = []
        window: list[dict] = []

        for entry in entries:
            title = entry.get("title", "")
            url = entry.get("url", "")
            ts_raw = entry.get("time_usec") or entry.get("timestamp") or entry.get("visitTime", "")

            if not title and not url:
                continue

            ts = self._parse_browser_ts(ts_raw)
            window.append({"title": title, "url": url, "ts": ts})

            # Group browsing sessions: 15 entries per window
            if len(window) >= 15:
                events.append(self._window_to_event(window, meta))
                window = []

        if window:
            events.append(self._window_to_event(window, meta))

        return events

    def _window_to_event(self, window: list[dict], meta: dict) -> ContextEvent:
        """Convert browsing window into a ContextEvent."""
        content_lines = []
        domains = set()

        for entry in window:
            url = entry["url"]
            domain = self._extract_domain(url)
            domains.add(domain)
            content_lines.append(f"Visited: {entry['title']} ({domain})")

        content = "\n".join(content_lines)
        earliest = min(e["ts"] for e in window)

        return ContextEvent(
            source=SourceType.BROWSER,
            content=content,
            timestamp=earliest,
            metadata={**meta, "domains": sorted(domains), "visit_count": len(window)},
            raw_format="browser_history",
        )

    @staticmethod
    def _parse_browser_ts(raw) -> datetime:
        """Parse Chrome's microsecond timestamps or standard formats."""
        if not raw:
            return datetime.utcnow()

        try:
            usec = int(raw)
            # Chrome uses microseconds since 1601-01-01
            if usec > 1e16:
                epoch_delta = 11644473600
                return datetime.fromtimestamp(usec / 1e6 - epoch_delta)
            if usec > 1e12:
                return datetime.fromtimestamp(usec / 1000)
            return datetime.fromtimestamp(usec)
        except (ValueError, OSError):
            pass

        return datetime.utcnow()

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        url = url.split("://", 1)[-1] if "://" in url else url
        return url.split("/", 1)[0].split("?", 1)[0]
