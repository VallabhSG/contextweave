"""Adapter for iCalendar (.ics) files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from contextweave.ingestion.base import BaseAdapter
from contextweave.schemas import ContextEvent, SourceType


class CalendarAdapter(BaseAdapter):
    """Ingests iCalendar (.ics) files into ContextEvents."""

    @property
    def supported_formats(self) -> list[str]:
        return [".ics"]

    def ingest_file(self, path: Path) -> list[ContextEvent]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        return self.ingest_text(raw, metadata={"filename": path.name, "path": str(path)})

    def ingest_text(self, raw: str, metadata: dict | None = None) -> list[ContextEvent]:
        meta = metadata or {}

        try:
            from icalendar import Calendar
        except ImportError:
            return self._fallback_parse(raw, meta)

        cal = Calendar.from_ical(raw)
        events = []

        for component in cal.walk():
            if component.name != "VEVENT":
                continue

            summary = str(component.get("summary", "Untitled Event"))
            description = str(component.get("description", ""))
            location = str(component.get("location", ""))
            dtstart = component.get("dtstart")
            attendees = component.get("attendee", [])

            ts = datetime.utcnow()
            if dtstart:
                dt = dtstart.dt
                ts = dt if isinstance(dt, datetime) else datetime.combine(dt, datetime.min.time())

            if not isinstance(attendees, list):
                attendees = [attendees]
            attendee_list = [str(a).replace("mailto:", "") for a in attendees if a]

            content_parts = [f"Event: {summary}"]
            if description:
                content_parts.append(f"Description: {description}")
            if location:
                content_parts.append(f"Location: {location}")
            if attendee_list:
                content_parts.append(f"Attendees: {', '.join(attendee_list)}")

            events.append(
                ContextEvent(
                    source=SourceType.CALENDAR,
                    content="\n".join(content_parts),
                    timestamp=ts,
                    metadata={
                        **meta,
                        "event_title": summary,
                        "location": location,
                        "attendees": attendee_list,
                    },
                    raw_format="ical",
                )
            )

        return events

    def _fallback_parse(self, raw: str, meta: dict) -> list[ContextEvent]:
        """Simple regex-based ICS parsing when icalendar is unavailable."""
        events = []
        current: dict = {}

        for line in raw.splitlines():
            line = line.strip()
            if line == "BEGIN:VEVENT":
                current = {}
            elif line == "END:VEVENT" and current:
                content = f"Event: {current.get('SUMMARY', 'Untitled')}"
                if current.get("DESCRIPTION"):
                    content += f"\nDescription: {current['DESCRIPTION']}"

                events.append(
                    ContextEvent(
                        source=SourceType.CALENDAR,
                        content=content,
                        timestamp=datetime.utcnow(),
                        metadata={**meta, "event_title": current.get("SUMMARY", "")},
                        raw_format="ical",
                    )
                )
                current = {}
            elif ":" in line and current is not None:
                key, _, value = line.partition(":")
                key = key.split(";")[0]  # Strip parameters
                current[key] = value

        return events
