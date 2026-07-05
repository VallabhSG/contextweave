"""Request dependencies: credential extraction and workspace resolution."""

from fastapi import HTTPException, Request

from contextweave.auth.supabase import verify_supabase_jwt
from contextweave.auth.users import get_user_store
from contextweave.workspaces import DEMO_USER_ID, Workspace, manager


def _extract_credential(request: Request) -> str | None:
    key = request.headers.get("x-api-key")
    if key:
        return key.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def get_workspace(request: Request) -> Workspace:
    """Resolve the caller's workspace.

    - no credential      → shared demo space
    - cw_* API key       → private workspace (programmatic access)
    - Supabase JWT       → private workspace derived from the auth user id
    """
    credential = _extract_credential(request)
    if credential is None:
        return manager.get(DEMO_USER_ID)

    if credential.startswith("cw_"):
        user_id = get_user_store().verify(credential)
        if user_id is None:
            raise HTTPException(401, "Invalid API key")
        return manager.get(user_id)

    if "." in credential:  # JWT shape — a Supabase session token
        user_id = verify_supabase_jwt(credential)
        if user_id is None:
            raise HTTPException(401, "Invalid or expired session token")
        return manager.get(user_id)

    raise HTTPException(401, "Unrecognized credential")
