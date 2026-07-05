"""Proactive nudge endpoints: the cached digest and its email subscription."""

import json
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from contextweave.api.deps import get_workspace
from contextweave.api.rate_limit import limiter
from contextweave.config import settings
from contextweave.notify import mailer
from contextweave.notify.subscriptions import get_subscription_store
from contextweave.reasoning.digest import DigestEngine
from contextweave.timeutils import utcnow
from contextweave.workspaces import Workspace

router = APIRouter()

DIGEST_LIMIT = "12/hour"

# Deliberately simple: enough to stop typos and garbage, not RFC 5322
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

_engine = DigestEngine()


def build_digest_payload(ws: Workspace, force: bool = False) -> dict:
    """Cached digest for a workspace, regenerating past CW_DIGEST_CACHE_HOURS.

    Shared by the API route and the email scheduler — both must see the
    same nudge for the same day.
    """
    if not force:
        latest = ws.memory_store.latest_digest()
        if latest is not None:
            generated_at, payload = latest
            if utcnow() - generated_at < timedelta(hours=settings.digest_cache_hours):
                return {**json.loads(payload), "cached": True}

    memories = ws.memory_store.list_recent(15)
    digest = _engine.generate(memories)
    payload = digest.model_dump_json()
    ws.memory_store.save_digest(payload)
    return {**json.loads(payload), "cached": False}


@router.get("/digest")
@limiter.limit(DIGEST_LIMIT)
async def get_digest(request: Request, force: bool = False, ws: Workspace = Depends(get_workspace)):
    """Today's nudge — regenerated at most every CW_DIGEST_CACHE_HOURS unless forced."""
    return await run_in_threadpool(build_digest_payload, ws, force)


# ── Email subscription ───────────────────────────────────────


class SubscribeRequest(BaseModel):
    email: str = Field(pattern=EMAIL_PATTERN, max_length=254)
    send_hour_utc: int = Field(default=3, ge=0, le=23)


@router.get("/digest/subscription")
def digest_subscription(ws: Workspace = Depends(get_workspace)):
    """Whether email delivery is available, and the caller's subscription."""
    available = mailer.smtp_configured()
    sub = None if ws.is_demo else get_subscription_store().get(ws.user_id)
    if not sub:
        return {"available": available, "subscribed": False}
    return {
        "available": available,
        "subscribed": True,
        "email": sub["email"],
        "send_hour_utc": sub["send_hour_utc"],
    }


@router.post("/digest/subscribe")
@limiter.limit(DIGEST_LIMIT)
def subscribe_digest(
    request: Request, req: SubscribeRequest, ws: Workspace = Depends(get_workspace)
):
    """Opt in to the daily digest email for this workspace."""
    if ws.is_demo:
        raise HTTPException(403, "Register or sign in first — the demo space has no inbox.")
    if not mailer.smtp_configured():
        raise HTTPException(503, "Email delivery is not configured on this server.")
    token = get_subscription_store().subscribe(ws.user_id, req.email, req.send_hour_utc)
    return {
        "status": "subscribed",
        "email": req.email,
        "send_hour_utc": req.send_hour_utc,
        "unsubscribe_token": token,
    }


@router.delete("/digest/subscribe")
def unsubscribe_digest(ws: Workspace = Depends(get_workspace)):
    """Opt out again (authenticated path)."""
    if ws.is_demo:
        raise HTTPException(403, "The demo space has no subscription to remove.")
    removed = get_subscription_store().unsubscribe(ws.user_id)
    return {"status": "unsubscribed" if removed else "not-subscribed"}


@router.get("/digest/unsubscribe")
def unsubscribe_by_token(token: str = ""):
    """One-click unsubscribe from the email footer — no sign-in required."""
    if not get_subscription_store().unsubscribe_by_token(token):
        raise HTTPException(404, "Unknown or already-used unsubscribe link.")
    return HTMLResponse(
        "<html><body style='font-family:Georgia,serif;max-width:480px;"
        "margin:80px auto;color:#0e0e0c'>"
        "<h2 style='font-weight:400'>Unsubscribed.</h2>"
        "<p style='color:#6b6a64'>No more daily digests. You can re-subscribe "
        "anytime from your space.</p></body></html>"
    )
