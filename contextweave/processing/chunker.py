"""Semantic chunking that respects conversation boundaries and topic shifts."""

from __future__ import annotations

import re
import uuid

from contextweave.config import settings
from contextweave.schemas import Chunk, ContextEvent, SourceType


class SemanticChunker:
    """Splits ContextEvents into semantically coherent chunks.

    Unlike naive fixed-size splitting, this chunker respects:
    - Conversation turn boundaries
    - Paragraph/section breaks
    - Topic shifts (via sentence-level heuristics)
    - Temporal coherence (keeps timestamps attached)
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        overlap_sentences: int | None = None,
    ):
        self.max_tokens = max_tokens or settings.chunk_max_tokens
        self.overlap_sentences = overlap_sentences or settings.chunk_overlap_sentences

    def chunk_event(self, event: ContextEvent) -> list[Chunk]:
        """Split a single ContextEvent into chunks."""
        if event.source == SourceType.CONVERSATION:
            return self._chunk_conversation(event)
        if event.source == SourceType.BROWSER:
            return self._chunk_browser(event)
        return self._chunk_prose(event)

    def chunk_events(self, events: list[ContextEvent]) -> list[Chunk]:
        """Chunk multiple events."""
        chunks = []
        for event in events:
            chunks.extend(self.chunk_event(event))
        return chunks

    def _chunk_conversation(self, event: ContextEvent) -> list[Chunk]:
        """Chunk chat conversations by turn windows."""
        lines = event.content.split("\n")
        chunks = []
        window: list[str] = []
        window_start = 0
        token_count = 0

        for i, line in enumerate(lines):
            line_tokens = self._estimate_tokens(line)

            if token_count + line_tokens > self.max_tokens and window:
                chunk_text = "\n".join(window)
                chunks.append(self._make_chunk(event, chunk_text, window_start, i))

                # Overlap: keep last N lines
                overlap = window[-self.overlap_sentences :] if self.overlap_sentences else []
                window = overlap
                window_start = max(0, i - len(overlap))
                token_count = sum(self._estimate_tokens(ln) for ln in window)

            window.append(line)
            token_count += line_tokens

        if window:
            chunk_text = "\n".join(window)
            chunks.append(self._make_chunk(event, chunk_text, window_start, len(lines)))

        return chunks

    def _chunk_prose(self, event: ContextEvent) -> list[Chunk]:
        """Chunk prose text by paragraphs, then sentences."""
        paragraphs = re.split(r"\n\s*\n", event.content)

        chunks = []
        buffer: list[str] = []
        buffer_start = 0
        token_count = 0
        char_offset = 0

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)

            # Single paragraph exceeds limit — split by sentences
            if para_tokens > self.max_tokens:
                if buffer:
                    chunks.append(
                        self._make_chunk(event, "\n\n".join(buffer), buffer_start, char_offset)
                    )
                    buffer = []
                    token_count = 0

                sentence_chunks = self._split_sentences(para, event, char_offset)
                chunks.extend(sentence_chunks)
                char_offset += len(para) + 2
                buffer_start = char_offset
                continue

            if token_count + para_tokens > self.max_tokens and buffer:
                chunks.append(
                    self._make_chunk(event, "\n\n".join(buffer), buffer_start, char_offset)
                )

                # Overlap
                overlap = buffer[-1:] if self.overlap_sentences else []
                buffer = overlap
                buffer_start = char_offset - len(overlap[0]) if overlap else char_offset
                token_count = sum(self._estimate_tokens(b) for b in buffer)

            buffer.append(para)
            token_count += para_tokens
            char_offset += len(para) + 2

        if buffer:
            chunks.append(self._make_chunk(event, "\n\n".join(buffer), buffer_start, char_offset))

        return chunks

    def _chunk_browser(self, event: ContextEvent) -> list[Chunk]:
        """Chunk browser history — already windowed, usually fits in one chunk."""
        tokens = self._estimate_tokens(event.content)
        if tokens <= self.max_tokens:
            return [self._make_chunk(event, event.content, 0, len(event.content))]
        return self._chunk_prose(event)

    def _split_sentences(self, text: str, event: ContextEvent, base_offset: int) -> list[Chunk]:
        """Split a long paragraph into sentence-based chunks."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        window: list[str] = []
        token_count = 0

        for sentence in sentences:
            s_tokens = self._estimate_tokens(sentence)
            if token_count + s_tokens > self.max_tokens and window:
                chunk_text = " ".join(window)
                chunks.append(
                    self._make_chunk(event, chunk_text, base_offset, base_offset + len(chunk_text))
                )

                overlap = window[-self.overlap_sentences :]
                window = overlap
                token_count = sum(self._estimate_tokens(s) for s in window)

            window.append(sentence)
            token_count += s_tokens

        if window:
            chunk_text = " ".join(window)
            chunks.append(
                self._make_chunk(event, chunk_text, base_offset, base_offset + len(chunk_text))
            )

        return chunks

    def _make_chunk(self, event: ContextEvent, content: str, start: int, end: int) -> Chunk:
        return Chunk(
            id=str(uuid.uuid4()),
            event_id=event.id,
            content=content,
            start_idx=start,
            end_idx=end,
            timestamp=event.timestamp,
            source=event.source,
            metadata=dict(event.metadata),
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token for English."""
        return max(1, len(text) // 4)
