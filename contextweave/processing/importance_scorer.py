"""Importance scoring with temporal decay for memories."""

from __future__ import annotations

import math
from datetime import datetime


class ImportanceScorer:
    """Scores memories using temporal decay, access frequency, and connection density.

    Formula: importance = base × recency_decay × access_boost × connection_boost

    Where:
    - recency_decay = exp(-ln(2) × days_elapsed / half_life)
    - access_boost = 1 + log(1 + access_count) × boost_factor
    - connection_boost = 1 + connection_count × density_weight
    """

    def __init__(
        self,
        half_life_days: float = 30.0,
        access_boost_factor: float = 1.2,
        connection_density_weight: float = 0.3,
    ):
        self.half_life_days = half_life_days
        self.access_boost_factor = access_boost_factor
        self.connection_density_weight = connection_density_weight

    def score(
        self,
        base_importance: float,
        timestamp: datetime,
        access_count: int = 0,
        connection_count: int = 0,
        now: datetime | None = None,
    ) -> float:
        """Calculate the current importance score for a memory."""
        now = now or datetime.utcnow()

        recency = self._recency_decay(timestamp, now)
        access = self._access_boost(access_count)
        connections = self._connection_boost(connection_count)

        return min(1.0, base_importance * recency * access * connections)

    def _recency_decay(self, timestamp: datetime, now: datetime) -> float:
        """Exponential decay based on age."""
        days_elapsed = max(0, (now - timestamp).total_seconds() / 86400)
        return math.exp(-math.log(2) * days_elapsed / self.half_life_days)

    def _access_boost(self, access_count: int) -> float:
        """Logarithmic boost from access frequency."""
        return 1.0 + math.log1p(access_count) * self.access_boost_factor

    def _connection_boost(self, connection_count: int) -> float:
        """Linear boost from graph connectivity."""
        return 1.0 + connection_count * self.connection_density_weight

    def estimate_base_importance(self, content: str, source: str) -> float:
        """Heuristic base importance from content signals."""
        score = 0.5

        # Longer content tends to be more substantive
        word_count = len(content.split())
        if word_count > 200:
            score += 0.1
        elif word_count < 20:
            score -= 0.1

        # Source weighting
        source_weights = {
            "conversation": 0.05,
            "note": 0.1,
            "journal": 0.15,
            "calendar": 0.0,
            "browser": -0.1,
        }
        score += source_weights.get(source, 0)

        # Content signals
        importance_signals = [
            "important",
            "critical",
            "decision",
            "agreed",
            "deadline",
            "action item",
            "follow up",
            "remember",
            "key takeaway",
        ]
        content_lower = content.lower()
        signal_count = sum(1 for s in importance_signals if s in content_lower)
        score += min(0.2, signal_count * 0.05)

        return max(0.1, min(1.0, score))
