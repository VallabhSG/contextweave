"""Supabase Auth: verify GoTrue session JWTs and map them to workspace ids."""

from __future__ import annotations

import logging

from contextweave.config import settings

logger = logging.getLogger(__name__)


def verify_supabase_jwt(token: str) -> str | None:
    """Return a stable workspace user id for a valid Supabase JWT, else None.

    Supabase signs session tokens with the project's JWT secret (HS256,
    audience "authenticated"). The sub claim is the auth user's UUID; we
    derive the workspace id from it so the same person always lands in
    the same memory space regardless of session.
    """
    secret = settings.supabase_jwt_secret
    if not secret:
        return None

    import jwt

    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            audience="authenticated",
            options={"require": ["sub", "exp"]},
        )
    except jwt.PyJWTError as e:
        logger.debug("Supabase JWT rejected: %s", e)
        return None

    sub = str(payload.get("sub", ""))
    if not sub:
        return None
    return "sb_" + sub.replace("-", "")[:24]
