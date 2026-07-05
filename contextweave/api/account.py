"""Account, privacy, and data-control endpoints."""

import logging
import re

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from contextweave.api.deps import get_workspace
from contextweave.api.rate_limit import limiter
from contextweave.auth.users import get_user_store
from contextweave.config import settings
from contextweave.workspaces import Workspace

logger = logging.getLogger(__name__)

router = APIRouter()

REGISTER_LIMIT = "5/hour"


class RegisterRequest(BaseModel):
    name: str = Field(default="", max_length=80)


@router.post("/auth/register")
@limiter.limit(REGISTER_LIMIT)
def register(request: Request, req: RegisterRequest | None = None):
    """Create a private workspace; the API key is returned exactly once."""
    user_id, api_key = get_user_store().create_user(name=req.name if req else "")
    logger.info("Registered private workspace %s", user_id)
    return {
        "user_id": user_id,
        "api_key": api_key,
        "message": (
            "Store this key now — it is shown only once. "
            "Send it as an X-API-Key header on every request."
        ),
    }


def _supabase_base_url(url: str) -> str:
    """Reduce a pasted Supabase URL to the project base supabase-js expects.

    The dashboard surfaces per-service endpoints (…/rest/v1, …/auth/v1)
    prominently, so that is what people configure; the client appends the
    service paths itself.
    """
    url = url.strip().rstrip("/")
    return re.sub(r"/(rest|auth|realtime|storage|functions)/v\d+$", "", url)


@router.get("/auth/config")
def auth_config():
    """Public auth discovery for the web UI.

    The Supabase URL and anon key are public by design (they ship in every
    Supabase frontend); sign-in is only offered when the backend also holds
    the JWT secret needed to verify the resulting session tokens.
    """
    enabled = bool(
        settings.supabase_url and settings.supabase_anon_key and settings.supabase_jwt_secret
    )
    if not enabled:
        return {"supabase": {"enabled": False}}
    return {
        "supabase": {
            "enabled": True,
            "url": _supabase_base_url(settings.supabase_url),
            "anon_key": settings.supabase_anon_key,
        }
    }


@router.get("/me")
def me(ws: Workspace = Depends(get_workspace)):
    """Identify the caller's workspace and its size."""
    return {"user_id": ws.user_id, "private": not ws.is_demo, **ws.memory_store.stats()}


@router.get("/export")
def export_memory(ws: Workspace = Depends(get_workspace)):
    """Full portable JSON export of everything in the caller's workspace."""
    return ws.export_data()


@router.delete("/memory")
def wipe_memory(ws: Workspace = Depends(get_workspace)):
    """Erase every memory in the caller's private workspace."""
    if ws.is_demo:
        raise HTTPException(
            403,
            "The shared demo space cannot be wiped. "
            "Register a private space to control your own data.",
        )
    ws.wipe()
    logger.info("Workspace %s wiped by owner", ws.user_id)
    return {"status": "wiped", "user_id": ws.user_id}
