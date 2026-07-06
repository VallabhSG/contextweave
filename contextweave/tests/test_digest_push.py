"""Pushed daily digest: subscriptions, the hourly sweep, and email delivery.

SMTP is never touched — the sender is monkeypatched and captured. The
sweep logic is tested against a pinned clock.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from contextweave.config import settings


@pytest.fixture
def smtp_on(monkeypatch):
    monkeypatch.setattr(settings, "smtp_host", "smtp.test")
    monkeypatch.setattr(settings, "smtp_username", "mailer@test")
    monkeypatch.setattr(settings, "smtp_password", "pw")
    monkeypatch.setattr(settings, "public_base_url", "https://cw.test")


@pytest.fixture
def outbox(monkeypatch):
    sent: list[dict] = []
    from contextweave.notify import mailer

    monkeypatch.setattr(
        mailer,
        "send_email",
        lambda to, subject, text, html: sent.append(
            {"to": to, "subject": subject, "text": text, "html": html}
        ),
    )
    return sent


def _register(client):
    body = client.post("/api/auth/register", json={"name": "push"}).json()
    return body["user_id"], {"X-API-Key": body["api_key"]}


class TestResendTransport:
    """HTTP delivery for hosts that block SMTP egress (e.g. HF Spaces)."""

    def test_resend_key_alone_enables_delivery(self, monkeypatch):
        from contextweave.notify import mailer

        monkeypatch.setattr(settings, "smtp_host", "")
        monkeypatch.setattr(settings, "resend_api_key", "re_test_123")
        assert mailer.email_configured() is True

    def test_send_goes_through_resend_api(self, monkeypatch):
        from contextweave.notify import mailer

        monkeypatch.setattr(settings, "smtp_host", "")
        monkeypatch.setattr(settings, "resend_api_key", "re_test_123")
        monkeypatch.setattr(settings, "digest_from_email", "")

        captured = {}

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                pass

        def fake_post(url, json=None, headers=None, timeout=None):
            captured.update({"url": url, "json": json, "headers": headers})
            return FakeResponse()

        import httpx

        monkeypatch.setattr(httpx, "post", fake_post)
        mailer.send_email("me@example.com", "Sub", "text body", "<p>html</p>")

        assert captured["url"] == "https://api.resend.com/emails"
        assert captured["headers"]["Authorization"] == "Bearer re_test_123"
        assert captured["json"]["to"] == ["me@example.com"]
        assert captured["json"]["subject"] == "Sub"
        assert captured["json"]["html"] == "<p>html</p>"
        # Resend's shared onboarding sender is the no-config default
        assert captured["json"]["from"] == "ContextWeave <onboarding@resend.dev>"

    def test_resend_error_raises(self, monkeypatch):
        from contextweave.notify import mailer

        monkeypatch.setattr(settings, "smtp_host", "")
        monkeypatch.setattr(settings, "resend_api_key", "re_test_123")

        import httpx

        def fake_post(url, json=None, headers=None, timeout=None):
            raise httpx.ConnectError("boom")

        monkeypatch.setattr(httpx, "post", fake_post)
        with pytest.raises(Exception):
            mailer.send_email("me@example.com", "Sub", "t", "<p>h</p>")


class TestSubscriptionEndpoints:
    def test_demo_space_cannot_subscribe(self, client, smtp_on):
        r = client.post("/api/digest/subscribe", json={"email": "a@b.co", "send_hour_utc": 3})
        assert r.status_code == 403

    def test_unavailable_without_smtp_config(self, client):
        _, h = _register(client)
        r = client.post(
            "/api/digest/subscribe", json={"email": "a@b.co", "send_hour_utc": 3}, headers=h
        )
        assert r.status_code == 503
        assert client.get("/api/digest/subscription", headers=h).json()["available"] is False

    def test_subscribe_status_unsubscribe(self, client, smtp_on):
        _, h = _register(client)
        r = client.post(
            "/api/digest/subscribe", json={"email": "me@example.com", "send_hour_utc": 6}, headers=h
        )
        assert r.status_code == 200

        status = client.get("/api/digest/subscription", headers=h).json()
        assert status == {
            "available": True,
            "subscribed": True,
            "email": "me@example.com",
            "send_hour_utc": 6,
        }

        assert client.delete("/api/digest/subscribe", headers=h).status_code == 200
        assert client.get("/api/digest/subscription", headers=h).json()["subscribed"] is False

    def test_invalid_email_rejected(self, client, smtp_on):
        _, h = _register(client)
        for bad in ("not-an-email", "a@b", "a b@c.co", ""):
            r = client.post(
                "/api/digest/subscribe", json={"email": bad, "send_hour_utc": 3}, headers=h
            )
            assert r.status_code == 422, bad

    def test_tokenized_unsubscribe_link_needs_no_auth(self, client, smtp_on):
        _, h = _register(client)
        token = client.post(
            "/api/digest/subscribe", json={"email": "me@example.com", "send_hour_utc": 3}, headers=h
        ).json()["unsubscribe_token"]

        r = client.get(f"/api/digest/unsubscribe?token={token}")
        assert r.status_code == 200
        assert client.get("/api/digest/subscription", headers=h).json()["subscribed"] is False

        # replay / garbage tokens do nothing loudly
        assert client.get(f"/api/digest/unsubscribe?token={token}").status_code == 404


class TestSweep:
    def _subscribed_workspace(self, client, hour=6, with_memory=True):
        _, h = _register(client)
        if with_memory:
            client.post(
                "/api/ingest/text",
                json={"content": "Ship the digest mailer by Friday — I promised Meera."},
                headers=h,
            )
        client.post(
            "/api/digest/subscribe",
            json={"email": "me@example.com", "send_hour_utc": hour},
            headers=h,
        )
        return h

    def test_sends_at_the_subscribed_hour_once_per_day(self, client, smtp_on, outbox):
        from contextweave.notify.scheduler import send_due_digests

        self._subscribed_workspace(client, hour=6)
        at_six = datetime(2026, 7, 6, 6, 4)

        assert send_due_digests(now=at_six) == 1
        assert len(outbox) == 1
        assert outbox[0]["to"] == "me@example.com"
        assert "unsubscribe" in outbox[0]["html"].lower()

        # same hour again → already sent today
        assert send_due_digests(now=at_six) == 0
        # next day → sends again
        assert send_due_digests(now=datetime(2026, 7, 7, 6, 30)) == 1

    def test_wrong_hour_sends_nothing(self, client, smtp_on, outbox):
        from contextweave.notify.scheduler import send_due_digests

        self._subscribed_workspace(client, hour=6)
        assert send_due_digests(now=datetime(2026, 7, 6, 9, 0)) == 0
        assert outbox == []

    def test_empty_workspace_skipped(self, client, smtp_on, outbox):
        from contextweave.notify.scheduler import send_due_digests

        self._subscribed_workspace(client, hour=6, with_memory=False)
        assert send_due_digests(now=datetime(2026, 7, 6, 6, 0)) == 0
        assert outbox == []

    def test_one_failing_send_does_not_block_the_rest(self, client, smtp_on, monkeypatch):
        from contextweave.notify import mailer
        from contextweave.notify.scheduler import send_due_digests

        self._subscribed_workspace(client, hour=6)
        h2 = self._subscribed_workspace(client, hour=6)
        del h2

        calls = {"n": 0}

        def flaky(to, subject, text, html):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("smtp down")

        monkeypatch.setattr(mailer, "send_email", flaky)
        assert send_due_digests(now=datetime(2026, 7, 6, 6, 0)) == 1
        assert calls["n"] == 2
