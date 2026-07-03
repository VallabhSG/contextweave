"""Shared per-client-IP rate limiter (in-memory, single-process)."""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

INGEST_LIMIT = "10/minute"
BATCH_LIMIT = "5/minute"
QUERY_LIMIT = "20/minute"

limiter = Limiter(key_func=get_remote_address)
