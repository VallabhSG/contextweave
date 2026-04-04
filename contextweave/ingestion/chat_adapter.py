"""Adapter for chat exports (WhatsApp, Telegram, Slack, generic JSON)."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from contextweave.ingestion.base import BaseAdapter
from contextweave.schemas import ContextEvent, SourceType


class ChatAdapter(BaseAdapter):
    """Ingests chat export files and normalizes into ContextEvents."""

    @property
    def supported_formats(self) -> list[str]:
        return [".json", ".txt"]

    def ingest_file(self, path: Path) -> list[ContextEvent]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        meta = {"filename": path.name, "path": str(path)}

        if path.suffix == ".json":
            return self._parse_json_chat(raw, meta)
        return self._parse_whatsapp_txt(raw, meta)

    def ingest_text(self, raw: str, metadata: dict | None = None) -> list[ContextEvent]:
        meta = metadata or {}
        try:
            json.loads(raw)
            return self._parse_json_chat(raw, meta)
        except (json.JSONDecodeError, ValueError):
            return self._parse_whatsapp_txt(raw, meta)

    def _parse_json_chat(self, raw: str, meta: dict) -> list[ContextEvent]:
        """Parse JSON chat exports (Slack, Telegram, generic)."""
        data = json.loads(raw)
        messages = data if isinstance(data, list) else data.get("messages", [])

        events = []
        window: list[dict] = []

        for msg in messages:
            text = msg.get("text") or msg.get("content") or msg.get("body", "")
            if not isinstance(text, str) or not text.strip():
                continue

            sender = msg.get("from") or msg.get("user") or msg.get("sender", "unknown")
            ts_raw = msg.get("date") or msg.get("ts") or msg.get("timestamp", "")
            ts = self._parse_timestamp(ts_raw)

            window.append({"sender": sender, "text": text, "ts": ts})

            # Group into conversation windows of ~10 messages
            if len(window) >= 10:
                events.append(self._window_to_event(window, meta))
                window = []

        if window:
            events.append(self._window_to_event(window, meta))

        return events

    def _parse_whatsapp_txt(self, raw: str, meta: dict) -> list[ContextEvent]:
        """Parse WhatsApp plain-text exports."""
        pattern = re.compile(
            r"\[?(\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\]?\s*-?\s*([^:]+):\s*(.*)",
        )

        window: list[dict] = []
        events = []

        for line in raw.splitlines():
            match = pattern.match(line)
            if match:
                ts = self._parse_timestamp(match.group(1))
                sender = match.group(2).strip()
                text = match.group(3).strip()
                if text:
                    window.append({"sender": sender, "text": text, "ts": ts})

                if len(window) >= 10:
                    events.append(self._window_to_event(window, meta))
                    window = []

        if window:
            events.append(self._window_to_event(window, meta))

        return events

    def _window_to_event(self, window: list[dict], meta: dict) -> ContextEvent:
        """Convert a conversation window into a single ContextEvent."""
        speakers = {m["sender"] for m in window}
        content = "\n".join(f"[{m['sender']}]: {m['text']}" for m in window)
        earliest = min(m["ts"] for m in window)

        return ContextEvent(
            source=SourceType.CONVERSATION,
            content=content,
            timestamp=earliest,
            metadata={**meta, "speakers": sorted(speakers), "message_count": len(window)},
            raw_format="chat",
        )

    @staticmethod
    def _parse_timestamp(raw: str) -> datetime:
        """Best-effort timestamp parsing."""
        if not raw:
            return datetime.utcnow()

        # Handle Unix timestamps
        try:
            ts_float = float(raw)
            if ts_float > 1e12:  # milliseconds
                ts_float /= 1000
            return datetime.fromtimestamp(ts_float)
        except (ValueError, OSError):
            pass

        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%m/%d/%y, %I:%M %p",
            "%m/%d/%Y, %I:%M %p",
            "%d/%m/%Y, %H:%M",
            "%m/%d/%y, %H:%M:%S",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(raw.strip(), fmt)
            except ValueError:
                continue

        return datetime.utcnow()
