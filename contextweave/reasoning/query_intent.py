"""Query-intent detection, shared by retrieval and reasoning.

The *kind* of question changes what good context looks like. "How has my
thinking evolved?" wants history preserved, not decayed away; "what connects X
and Y?" leans on the graph. Detecting intent once, in one place, lets both the
retriever (which tunes ranking) and the reasoning engine (which picks a prompt)
agree on the query type instead of each re-deriving it.
"""

from __future__ import annotations

# Keyword hints per query type. A query is classified as the type with the most
# hint hits; ties fall back to insertion order; no hits means "general".
QUERY_TYPE_HINTS: dict[str, list[str]] = {
    "patterns": ["pattern", "trend", "recurring", "often", "usually", "tend to"],
    "gaps": ["avoiding", "missing", "overlooking", "neglecting", "forgot"],
    "temporal": ["evolved", "changed", "over time", "progression", "shift"],
    "cross_reference": ["think about", "opinion on", "what does", "relationship between"],
    "priorities": ["focus", "prioritize", "this week", "next", "should I", "what's important"],
}

GENERAL = "general"


def detect_query_type(query: str) -> str:
    """Classify a query into one of QUERY_TYPE_HINTS keys, or 'general'."""
    query_lower = query.lower()
    scores: dict[str, int] = {}
    for qtype, keywords in QUERY_TYPE_HINTS.items():
        hits = sum(1 for kw in keywords if kw in query_lower)
        if hits > 0:
            scores[qtype] = hits
    if scores:
        return max(scores, key=scores.get)
    return GENERAL
