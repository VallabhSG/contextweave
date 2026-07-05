"""Unit tests for Supabase JWT verification.

Legacy projects sign session tokens HS256 with a shared secret; projects
created after the 2025 signing-keys rollout use asymmetric keys (ES256)
published via JWKS. verify_supabase_jwt must handle both — and refuse
anything else.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from contextweave.auth import supabase as sb
from contextweave.config import settings

SUB = "11111111-2222-3333-4444-555555555555"
WORKSPACE = "sb_" + SUB.replace("-", "")[:24]
BASE = "https://proj.supabase.co"


def _claims(**over):
    payload = {
        "sub": SUB,
        "aud": "authenticated",
        "iss": f"{BASE}/auth/v1",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    payload.update(over)
    return payload


@pytest.fixture
def ec_key():
    return ec.generate_private_key(ec.SECP256R1())


class TestHs256Legacy:
    def test_valid_token_maps_to_workspace(self, monkeypatch):
        monkeypatch.setattr(settings, "supabase_jwt_secret", "legacy-secret")
        token = pyjwt.encode(_claims(), "legacy-secret", algorithm="HS256")
        assert sb.verify_supabase_jwt(token) == WORKSPACE

    def test_rejected_without_secret(self, monkeypatch):
        monkeypatch.setattr(settings, "supabase_jwt_secret", "")
        token = pyjwt.encode(_claims(), "anything", algorithm="HS256")
        assert sb.verify_supabase_jwt(token) is None


class TestEs256Jwks:
    def test_valid_token_maps_to_workspace(self, monkeypatch, ec_key):
        monkeypatch.setattr(settings, "supabase_url", BASE)
        monkeypatch.setattr(sb, "_signing_key", lambda token: ec_key.public_key())
        token = pyjwt.encode(_claims(), ec_key, algorithm="ES256")
        assert sb.verify_supabase_jwt(token) == WORKSPACE

    def test_url_with_service_suffix_still_verifies(self, monkeypatch, ec_key):
        # People paste …/rest/v1 — issuer pinning must use the normalized base
        monkeypatch.setattr(settings, "supabase_url", f"{BASE}/rest/v1/")
        monkeypatch.setattr(sb, "_signing_key", lambda token: ec_key.public_key())
        token = pyjwt.encode(_claims(), ec_key, algorithm="ES256")
        assert sb.verify_supabase_jwt(token) == WORKSPACE

    def test_rejected_without_configured_url(self, monkeypatch, ec_key):
        # No trusted JWKS source — never derive it from the token itself
        monkeypatch.setattr(settings, "supabase_url", "")
        token = pyjwt.encode(_claims(), ec_key, algorithm="ES256")
        assert sb.verify_supabase_jwt(token) is None

    def test_wrong_key_rejected(self, monkeypatch, ec_key):
        monkeypatch.setattr(settings, "supabase_url", BASE)
        other = ec.generate_private_key(ec.SECP256R1())
        monkeypatch.setattr(sb, "_signing_key", lambda token: other.public_key())
        token = pyjwt.encode(_claims(), ec_key, algorithm="ES256")
        assert sb.verify_supabase_jwt(token) is None

    def test_foreign_issuer_rejected(self, monkeypatch, ec_key):
        monkeypatch.setattr(settings, "supabase_url", BASE)
        monkeypatch.setattr(sb, "_signing_key", lambda token: ec_key.public_key())
        claims = _claims(iss="https://evil.supabase.co/auth/v1")
        token = pyjwt.encode(claims, ec_key, algorithm="ES256")
        assert sb.verify_supabase_jwt(token) is None


class TestAlgorithmConfusion:
    def test_alg_none_rejected(self, monkeypatch):
        monkeypatch.setattr(settings, "supabase_jwt_secret", "legacy-secret")
        monkeypatch.setattr(settings, "supabase_url", BASE)
        token = pyjwt.encode(_claims(), None, algorithm="none")
        assert sb.verify_supabase_jwt(token) is None

    def test_hs256_signed_with_public_key_material_rejected(self, monkeypatch, ec_key):
        # Classic confusion attack: HS256 token must only ever be checked
        # against the shared secret, never against JWKS material
        monkeypatch.setattr(settings, "supabase_jwt_secret", "legacy-secret")
        monkeypatch.setattr(settings, "supabase_url", BASE)
        token = pyjwt.encode(_claims(), "not-the-secret", algorithm="HS256")
        assert sb.verify_supabase_jwt(token) is None

    def test_garbage_token_rejected(self):
        assert sb.verify_supabase_jwt("not.a.jwt") is None
        assert sb.verify_supabase_jwt("") is None
