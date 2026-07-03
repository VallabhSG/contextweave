"""Request dependencies: API-key extraction and workspace resolution."""

from fastapi import HTTPException, Request

from contextweave.auth.users import UserStore
from contextweave.workspaces import DEMO_USER_ID, Workspace, manager


def _extract_api_key(request: Request) -> str | None:
    key = request.headers.get("x-api-key")
    if key:
        return key.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def get_workspace(request: Request) -> Workspace:
    """Valid API key → private workspace; no key → shared demo space."""
    api_key = _extract_api_key(request)
    if api_key is None:
        return manager.get(DEMO_USER_ID)
    user_id = UserStore().verify(api_key)
    if user_id is None:
        raise HTTPException(401, "Invalid API key")
    return manager.get(user_id)
