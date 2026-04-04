"""Tests for ingestion adapters."""

from __future__ import annotations

import json


from contextweave.ingestion.browser_adapter import BrowserAdapter
from contextweave.ingestion.calendar_adapter import CalendarAdapter
from contextweave.ingestion.chat_adapter import ChatAdapter
from contextweave.ingestion.text_adapter import TextAdapter
from contextweave.schemas import SourceType


class TestTextAdapter:
    def test_ingest_plain_text(self):
        adapter = TextAdapter()
        events = adapter.ingest_text("This is a test note.")
        assert len(events) == 1
        assert events[0].source == SourceType.NOTE
        assert "test note" in events[0].content

    def test_ingest_markdown_splits_sections(self):
        adapter = TextAdapter()
        md = "## Section A\n\nContent of A.\n\n## Section B\n\nContent of B."
        events = adapter.ingest_text(md, metadata={"filename": "test.md"})
        assert len(events) == 2
        assert any("Content of A" in e.content for e in events)
        assert any("Content of B" in e.content for e in events)

    def test_ingest_file(self, tmp_path):
        p = tmp_path / "note.md"
        p.write_text("## My Note\n\nSome content here.", encoding="utf-8")
        adapter = TextAdapter()
        events = adapter.ingest_file(p)
        assert len(events) >= 1
        assert any("Some content" in e.content for e in events)

    def test_empty_content_returns_empty(self):
        adapter = TextAdapter()
        events = adapter.ingest_text("")
        assert len(events) == 0

    def test_metadata_preserved(self):
        adapter = TextAdapter()
        events = adapter.ingest_text("Hello", metadata={"tag": "test"})
        assert events[0].metadata.get("tag") == "test"


class TestChatAdapter:
    def test_parse_json_chat(self):
        adapter = ChatAdapter()
        messages = [
            {"from": "Alice", "text": f"Message {i}", "date": "2024-01-15T10:00:00"}
            for i in range(5)
        ]
        raw = json.dumps(messages)
        events = adapter.ingest_text(raw)
        assert len(events) >= 1
        assert events[0].source == SourceType.CONVERSATION

    def test_parse_whatsapp_format(self):
        adapter = ChatAdapter()
        wa_text = (
            "[1/15/24, 10:00 AM] Alice: Hello there\n"
            "[1/15/24, 10:01 AM] Bob: Hi Alice!\n"
            "[1/15/24, 10:02 AM] Alice: How are you?\n"
        )
        events = adapter.ingest_text(wa_text)
        assert len(events) >= 1
        assert events[0].source == SourceType.CONVERSATION

    def test_speakers_in_metadata(self):
        adapter = ChatAdapter()
        messages = [{"from": "Alice", "text": "Hi", "date": "2024-01-15T10:00:00"}]
        events = adapter.ingest_text(json.dumps(messages))
        assert len(events) >= 1

    def test_window_grouping(self):
        adapter = ChatAdapter()
        messages = [
            {"from": "Alice", "text": f"Message {i}", "date": "2024-01-15T10:00:00"}
            for i in range(30)
        ]
        events = adapter.ingest_text(json.dumps(messages))
        # 30 messages at window size 10 → 3 events
        assert len(events) == 3


class TestBrowserAdapter:
    def test_parse_chrome_history(self):
        adapter = BrowserAdapter()
        history = {
            "Browser History": [
                {"title": f"Page {i}", "url": f"https://example.com/page{i}", "time_usec": "13369977600000000"}
                for i in range(5)
            ]
        }
        events = adapter.ingest_text(json.dumps(history))
        assert len(events) >= 1
        assert events[0].source == SourceType.BROWSER

    def test_extract_domain(self):
        assert BrowserAdapter._extract_domain("https://github.com/user/repo") == "github.com"
        assert BrowserAdapter._extract_domain("http://localhost:8080/api") == "localhost:8080"

    def test_domains_in_metadata(self):
        adapter = BrowserAdapter()
        history = [
            {"title": "GitHub", "url": "https://github.com", "time_usec": "13369977600000000"},
            {"title": "Google", "url": "https://google.com", "time_usec": "13369977600000000"},
        ]
        events = adapter.ingest_text(json.dumps(history))
        assert len(events) >= 1


class TestCalendarAdapter:
    ICS_CONTENT = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
SUMMARY:Team Standup
DESCRIPTION:Daily sync meeting
DTSTART:20240115T100000Z
DTEND:20240115T103000Z
LOCATION:Zoom
END:VEVENT
END:VCALENDAR"""

    def test_parse_ics(self):
        adapter = CalendarAdapter()
        events = adapter.ingest_text(self.ICS_CONTENT)
        assert len(events) >= 1
        assert events[0].source == SourceType.CALENDAR
        assert "Team Standup" in events[0].content

    def test_fallback_parse(self):
        adapter = CalendarAdapter()
        events = adapter._fallback_parse(self.ICS_CONTENT, {})
        assert len(events) >= 1
        assert "Team Standup" in events[0].content

    def test_location_in_content(self):
        adapter = CalendarAdapter()
        events = adapter.ingest_text(self.ICS_CONTENT)
        assert any("Zoom" in e.content for e in events)
