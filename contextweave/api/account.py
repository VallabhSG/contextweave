"""Account, privacy, and data-control endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from contextweave.api.deps import get_workspace
from contextweave.api.rate_limit import limiter
from contextweave.auth.users import UserStore
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
    user_id, api_key = UserStore().create_user(name=req.name if req else "")
    logger.info("Registered private workspace %s", user_id)
    return {
        "user_id": user_id,
        "api_key": api_key,
        "message": (
            "Store this key now — it is shown only once. "
            "Send it as an X-API-Key header on every request."
        ),
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
