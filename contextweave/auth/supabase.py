"""Supabase Auth: verify GoTrue session JWTs and map them to workspace ids."""

from __future__ import annotations

import logging
import re
import threading

from contextweave.config import settings

logger = logging.getLogger(__name__)

_ASYMMETRIC_ALGS = {"ES256", "RS256"}

_jwks_lock = threading.Lock()
_jwks_client = None
_jwks_url = ""


def supabase_base_url(url: str | None = None) -> str:
    """Reduce a pasted Supabase URL to the project base.

    Dashboards surface per-service endpoints (…/rest/v1, …/auth/v1)
    prominently, so that is what people configure; clients append the
    service paths themselves.
    """
    raw = (settings.supabase_url if url is None else url).strip().rstrip("/")
    return re.sub(r"/(rest|auth|realtime|storage|functions)/v\d+$", "", raw)


def _signing_key(token: str):
    """Resolve the public key for an asymmetric token from the project JWKS.

    The JWKS location derives from the *configured* project URL — never from
    the token's own iss claim, which an attacker controls.
    """
    global _jwks_client, _jwks_url
    from jwt import PyJWKClient

    url = supabase_base_url() + "/auth/v1/.well-known/jwks.json"
    with _jwks_lock:
        if _jwks_client is None or _jwks_url != url:
            _jwks_client = PyJWKClient(url)
            _jwks_url = url
        client = _jwks_client
    return client.get_signing_key_from_jwt(token).key


def verify_supabase_jwt(token: str) -> str | None:
    """Return a stable workspace user id for a valid Supabase JWT, else None.

    Legacy projects sign session tokens HS256 with the project JWT secret;
    projects on the newer signing-keys system use ES256/RS256 keys published
    at the project's JWKS endpoint. The two paths never share key material
    (algorithm-confusion guard). The sub claim is the auth user's UUID; the
    workspace id derives from it so the same person always lands in the same
    memory space regardless of session.
    """
    import jwt

    try:
        alg = str(jwt.get_unverified_header(token).get("alg", ""))
    except jwt.PyJWTError as e:
        logger.debug("Supabase JWT rejected (bad header): %s", e)
        return None

    try:
        if alg == "HS256":
            secret = settings.supabase_jwt_secret
            if not secret:
                return None
            payload = jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                audience="authenticated",
                options={"require": ["sub", "exp"]},
            )
        elif alg in _ASYMMETRIC_ALGS:
            if not settings.supabase_url:
                return None  # no trusted JWKS source configured
            payload = jwt.decode(
                token,
                _signing_key(token),
                algorithms=[alg],
                audience="authenticated",
                issuer=supabase_base_url() + "/auth/v1",
                options={"require": ["sub", "exp"]},
            )
        else:
            return None
    except Exception as e:  # signature/claim failures and JWKS fetch errors
        logger.debug("Supabase JWT rejected: %s", e)
        return None

    sub = str(payload.get("sub", ""))
    if not sub:
        return None
    return "sb_" + sub.replace("-", "")[:24]
