"""Token-budgeted, redundancy-aware context assembly.

The retriever returns the most relevant memories; this layer decides which of
them actually enter the language model's context window. A naive "stuff top-K
into the prompt" approach ignores three problems this module solves:

1. **Token budget** — a context window is finite. Pack memories by relevance
   until a budget is spent instead of dumping an unbounded number of
   full-length memories. When the query is known, a long memory is compressed to
   its most query-relevant sentences (not just its opening) so the budget holds
   more on-point context; without a query it is front-truncated to fit.

2. **Redundancy** — near-duplicate memories (the same recurring thought captured
   five times) waste the budget and bias the model toward whatever it repeats.
   A candidate that is lexically near-identical to something already packed is
   skipped. This is a cheap, dependency-free take on Maximal Marginal Relevance:
   keep relevance high while forcing diversity.

3. **Honest confidence** — how much to trust an answer depends on how relevant
   the packed context actually is and how broadly it is corroborated, not merely
   on how many rows were returned.

Selection lives here, deliberately separate from the retriever's recall+ranking
job: the retriever must still *return* near-duplicate memories (temporal-decay
behaviour depends on it), while the prompt should not *contain* them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from contextweave.schemas import QueryResult

# Rough English token estimate: ~4 characters per token. Documented as an
# estimate; `ContextBudgeter` accepts an injected counter for exact accounting.
_CHARS_PER_TOKEN = 4

_WORD_RE = re.compile(r"[a-z0-9]+")

# Split on sentence-ending punctuation followed by whitespace. Good enough for
# the prose and conversational memories this handles; no NLP dependency.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    """Heuristic token count for budgeting (chars / 4, floored at 1 for non-empty)."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _tokenize_words(text: str) -> set[str]:
    """Lowercased alphanumeric word set, for lexical similarity."""
    return set(_WORD_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard overlap of two word sets, in [0, 1]."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if intersection == 0:
        return 0.0
    return intersection / len(a | b)


def _split_sentences(text: str) -> list[str]:
    """Split prose into sentences on terminal punctuation; drops empties."""
    return [s for s in (part.strip() for part in _SENTENCE_RE.split(text)) if s]


@dataclass(frozen=True)
class AssembledContext:
    """The outcome of packing retrieved memories into a budget.

    ``results`` is the ordered subset that should go into the prompt. The counts
    explain what was left out and why, so callers can surface or log it.
    """

    results: list[QueryResult]
    token_estimate: int
    confidence: float
    dropped_for_budget: int = 0
    dropped_for_redundancy: int = 0
    truncated_ids: list[str] = field(default_factory=list)


class ContextBudgeter:
    """Selects and packs retrieved memories under a token budget.

    Args:
        token_budget: Maximum estimated tokens of memory content in the prompt.
        redundancy_threshold: Lexical Jaccard above which a candidate is treated
            as a near-duplicate of an already-packed memory and skipped.
        min_truncated_tokens: A memory that does not fit is only truncated (rather
            than dropped) if at least this many tokens of budget remain — a
            two-word fragment is not worth a context block.
        token_counter: Callable mapping text to a token count. Defaults to the
            chars/4 heuristic; inject a real tokenizer for exact accounting.
    """

    def __init__(
        self,
        token_budget: int = 3000,
        redundancy_threshold: float = 0.82,
        min_truncated_tokens: int = 64,
        max_tokens_per_memory: int = 200,
        token_counter: Callable[[str], int] | None = None,
    ):
        self.token_budget = token_budget
        self.redundancy_threshold = redundancy_threshold
        self.min_truncated_tokens = min_truncated_tokens
        self.max_tokens_per_memory = max_tokens_per_memory
        self._count = token_counter or estimate_tokens

    def assemble(self, results: list[QueryResult], query: str | None = None) -> AssembledContext:
        """Pack ``results`` (assumed ranked best-first) into the token budget.

        When ``query`` is given, a memory longer than ``max_tokens_per_memory`` is
        compressed to its most query-relevant sentences before packing — so the
        budget holds more distinct, on-point context instead of one memory's
        filler. Without a query, memories are packed whole (and front-truncated
        only at the budget boundary), preserving the query-agnostic behaviour.
        """
        if not results:
            return AssembledContext(results=[], token_estimate=0, confidence=0.0)

        query_words = _tokenize_words(query) if query else set()
        packed: list[QueryResult] = []
        packed_word_sets: list[set[str]] = []
        used_tokens = 0
        dropped_budget = 0
        dropped_redundant = 0
        reduced_ids: list[str] = []

        for idx, candidate in enumerate(results):
            content = candidate.content
            reduced = False

            # Proactive query-aware compression of a long memory to its core.
            if query_words and self._count(content) > self.max_tokens_per_memory:
                content = self._compress(content, query_words, self.max_tokens_per_memory)
                reduced = content != candidate.content

            words = _tokenize_words(content)

            # Redundancy: skip a near-duplicate of something already packed.
            # Never skip the first pick — an empty prompt helps no one.
            if packed and self._is_redundant(words, packed_word_sets):
                dropped_redundant += 1
                continue

            remaining = self.token_budget - used_tokens
            cost = self._count(content)

            if cost <= remaining:
                item = (
                    candidate
                    if content == candidate.content
                    else candidate.model_copy(update={"content": content})
                )
                packed.append(item)
                packed_word_sets.append(words)
                used_tokens += cost
                if reduced:
                    reduced_ids.append(candidate.chunk_id)
                continue

            # Does not fit whole. Shrink to the remaining budget if that leaves a
            # substantive fragment (and always include at least the top pick).
            if remaining >= self.min_truncated_tokens or not packed:
                fit = max(remaining, self.min_truncated_tokens)
                shrunk = self._fit_to_budget(content, query_words, fit)
                packed.append(candidate.model_copy(update={"content": shrunk}))
                packed_word_sets.append(_tokenize_words(shrunk))
                used_tokens += self._count(shrunk)
                reduced_ids.append(candidate.chunk_id)
                # Budget is effectively spent; remaining candidates can't fit.
                dropped_budget += len(results) - idx - 1
                break

            dropped_budget += 1

        return AssembledContext(
            results=packed,
            token_estimate=used_tokens,
            confidence=self._confidence(packed),
            dropped_for_budget=dropped_budget,
            dropped_for_redundancy=dropped_redundant,
            truncated_ids=reduced_ids,
        )

    def _is_redundant(self, words: set[str], packed_word_sets: list[set[str]]) -> bool:
        return any(
            _jaccard(words, prior) >= self.redundancy_threshold for prior in packed_word_sets
        )

    def _fit_to_budget(self, content: str, query_words: set[str], max_tokens: int) -> str:
        """Shrink content to max_tokens: query-aware if possible, else front-cut."""
        if query_words:
            return self._compress(content, query_words, max_tokens)
        return self._truncate(content, max_tokens)

    def _compress(self, content: str, query_words: set[str], max_tokens: int) -> str:
        """Keep a memory's most query-relevant sentences, in original order.

        The first sentence (framing/topic) is always kept; the rest are ranked by
        word overlap with the query and added until the token budget is reached.
        Falls back to a front-truncation when the text is a single long sentence.
        """
        if self._count(content) <= max_tokens:
            return content
        sentences = _split_sentences(content)
        if len(sentences) <= 1:
            return self._truncate(content, max_tokens)

        kept_indices = {0}
        used = self._count(sentences[0])
        ranked = sorted(
            enumerate(sentences[1:], start=1),
            key=lambda pair: len(_tokenize_words(pair[1]) & query_words),
            reverse=True,
        )
        for i, sentence in ranked:
            if len(_tokenize_words(sentence) & query_words) == 0:
                continue  # no relevance signal — don't spend budget on it
            cost = self._count(sentence)
            if used + cost > max_tokens:
                continue
            kept_indices.add(i)
            used += cost

        kept = [s for i, s in enumerate(sentences) if i in kept_indices]
        text = " ".join(kept).strip()
        if len(kept) < len(sentences):
            text = f"{text} …[trimmed]"
        # Safety net: if even the framing sentence overflowed, hard-truncate.
        if self._count(text) > max_tokens + 4:
            return self._truncate(content, max_tokens)
        return text

    def _truncate(self, content: str, max_tokens: int) -> str:
        """Cut content to ~max_tokens on a word boundary, marking the elision."""
        char_budget = max_tokens * _CHARS_PER_TOKEN
        if len(content) <= char_budget:
            return content
        clipped = content[:char_budget].rsplit(" ", 1)[0].rstrip()
        return f"{clipped} …[truncated]"

    @staticmethod
    def _confidence(packed: list[QueryResult]) -> float:
        """Confidence from packed relevance, not result count or ranking score.

        Reads ``relevance`` (pre-decay fused match quality), not ``score`` (which
        carries temporal decay and intent tuning) — so confidence answers "how
        well does this context match the query?" and does not swing just because
        a query was classified as temporal. Driven by the strongest match
        (``peak``) and the mean of the top few, nudged up modestly by breadth
        (distinct corroborating memories). Eight weakly-relevant memories yield
        low confidence — the whole point of not tying confidence to raw count.
        """
        if not packed:
            return 0.0
        scores = sorted((r.relevance for r in packed), reverse=True)
        top = scores[:3]
        peak = top[0]
        mean_top = sum(top) / len(top)
        relevance = 0.6 * peak + 0.4 * mean_top
        breadth = min(1.0, len(packed) / 5)
        return round(min(1.0, relevance * (0.7 + 0.3 * breadth)), 3)
