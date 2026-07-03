"""Proactive nudge endpoint: cached digest of focus, commitments, gaps."""

import json
from datetime import timedelta

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from contextweave.api.deps import get_workspace
from contextweave.api.rate_limit import limiter
from contextweave.config import settings
from contextweave.reasoning.digest import DigestEngine
from contextweave.timeutils import utcnow
from contextweave.workspaces import Workspace

router = APIRouter()

DIGEST_LIMIT = "12/hour"

_engine = DigestEngine()


@router.get("/digest")
@limiter.limit(DIGEST_LIMIT)
async def get_digest(request: Request, force: bool = False, ws: Workspace = Depends(get_workspace)):
    """Today's nudge — regenerated at most every CW_DIGEST_CACHE_HOURS unless forced."""
    if not force:
        latest = ws.memory_store.latest_digest()
        if latest is not None:
            generated_at, payload = latest
            if utcnow() - generated_at < timedelta(hours=settings.digest_cache_hours):
                return {**json.loads(payload), "cached": True}

    memories = await run_in_threadpool(ws.memory_store.list_recent, 15)
    digest = await run_in_threadpool(_engine.generate, memories)
    payload = digest.model_dump_json()
    ws.memory_store.save_digest(payload)
    return {**json.loads(payload), "cached": False}
