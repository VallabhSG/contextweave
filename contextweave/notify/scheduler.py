"""Hourly sweep that emails due digest subscriptions."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from starlette.concurrency import run_in_threadpool

from contextweave import workspaces
from contextweave.config import settings
from contextweave.notify import mailer
from contextweave.notify.subscriptions import get_subscription_store
from contextweave.timeutils import utcnow

logger = logging.getLogger(__name__)


def _unsubscribe_url(token: str) -> str | None:
    base = settings.public_base_url.strip().rstrip("/")
    if not base:
        return None
    return f"{base}/api/digest/unsubscribe?token={token}"


def send_due_digests(now: datetime | None = None) -> int:
    """Email every subscription due this hour; returns how many went out.

    Idempotent per day (last_sent_on guard), and one failing recipient
    never blocks the rest of the sweep.
    """
    if not mailer.smtp_configured():
        return 0
    from contextweave.api.digest_routes import build_digest_payload

    now = now or utcnow()
    today = now.date().isoformat()
    store = get_subscription_store()
    sent = 0
    for sub in store.due(now.hour, today):
        try:
            ws = workspaces.manager.get(sub["user_id"])
            if not ws.memory_store.list_recent(1):
                continue  # nothing to say yet — an empty nudge trains people to ignore it
            payload = build_digest_payload(ws)
            subject, text, html = mailer.render_digest_email(
                payload, _unsubscribe_url(sub["unsubscribe_token"])
            )
            mailer.send_email(sub["email"], subject, text, html)
            store.mark_sent(sub["user_id"], today)
            sent += 1
            logger.info("Digest emailed for workspace %s", sub["user_id"])
        except Exception:
            logger.exception("Digest delivery failed for workspace %s", sub["user_id"])
    return sent


async def scheduler_loop() -> None:
    """Sweep shortly after boot (restarts must not miss the current hour),
    then once past every hour boundary."""
    await asyncio.sleep(20)
    while True:
        try:
            sent = await run_in_threadpool(send_due_digests)
            if sent:
                logger.info("Digest sweep delivered %d email(s)", sent)
        except Exception:
            logger.exception("Digest sweep crashed; continuing")
        now = utcnow()
        await asyncio.sleep(3600 - (now.minute * 60 + now.second) + 30)
