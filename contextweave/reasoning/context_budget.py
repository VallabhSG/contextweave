"""Token-budgeted, redundancy-aware context assembly.

The retriever returns the most relevant memories; this layer decides which of
them actually enter the language model's context window. A naive "stuff top-K
into the prompt" approach ignores three problems this module solves:

1. **Token budget** — a context window is finite. Pack memories by relevance
   until a budget is spent instead of dumping an unbounded number of
   full-length memories. An over-long single memory is truncated to fit rather
   than dropped whole, so its opening (usually the most on-topic part) survives.

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
        token_counter: Callable[[str], int] | None = None,
    ):
        self.token_budget = token_budget
        self.redundancy_threshold = redundancy_threshold
        self.min_truncated_tokens = min_truncated_tokens
        self._count = token_counter or estimate_tokens

    def assemble(self, results: list[QueryResult]) -> AssembledContext:
        """Pack ``results`` (assumed ranked best-first) into the token budget."""
        if not results:
            return AssembledContext(results=[], token_estimate=0, confidence=0.0)

        packed: list[QueryResult] = []
        packed_word_sets: list[set[str]] = []
        used_tokens = 0
        dropped_budget = 0
        dropped_redundant = 0
        truncated_ids: list[str] = []

        for idx, candidate in enumerate(results):
            words = _tokenize_words(candidate.content)

            # Redundancy: skip a near-duplicate of something already packed.
            # Never skip the first pick — an empty prompt helps no one.
            if packed and self._is_redundant(words, packed_word_sets):
                dropped_redundant += 1
                continue

            remaining = self.token_budget - used_tokens
            cost = self._count(candidate.content)

            if cost <= remaining:
                packed.append(candidate)
                packed_word_sets.append(words)
                used_tokens += cost
                continue

            # Does not fit whole. Truncate to the remaining budget if that leaves
            # a substantive fragment (and always include at least the top pick).
            if remaining >= self.min_truncated_tokens or not packed:
                fit = max(remaining, self.min_truncated_tokens)
                truncated = self._truncate(candidate.content, fit)
                packed.append(candidate.model_copy(update={"content": truncated}))
                packed_word_sets.append(_tokenize_words(truncated))
                used_tokens += self._count(truncated)
                truncated_ids.append(candidate.chunk_id)
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
            truncated_ids=truncated_ids,
        )

    def _is_redundant(self, words: set[str], packed_word_sets: list[set[str]]) -> bool:
        return any(
            _jaccard(words, prior) >= self.redundancy_threshold for prior in packed_word_sets
        )

    def _truncate(self, content: str, max_tokens: int) -> str:
        """Cut content to ~max_tokens on a word boundary, marking the elision."""
        char_budget = max_tokens * _CHARS_PER_TOKEN
        if len(content) <= char_budget:
            return content
        clipped = content[:char_budget].rsplit(" ", 1)[0].rstrip()
        return f"{clipped} …[truncated]"

    @staticmethod
    def _confidence(packed: list[QueryResult]) -> float:
        """Confidence from packed relevance, not result count.

        Driven by the strongest match (``peak``) and the mean of the top few,
        then nudged up modestly by breadth (distinct corroborating memories).
        Eight weakly-relevant memories now yield low confidence — the whole
        point of not tying confidence to raw count.
        """
        if not packed:
            return 0.0
        scores = sorted((r.score for r in packed), reverse=True)
        top = scores[:3]
        peak = top[0]
        mean_top = sum(top) / len(top)
        relevance = 0.6 * peak + 0.4 * mean_top
        breadth = min(1.0, len(packed) / 5)
        return round(min(1.0, relevance * (0.7 + 0.3 * breadth)), 3)
