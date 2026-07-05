"""Digest email rendering and SMTP delivery (stdlib only, provider-agnostic)."""

from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from html import escape

from contextweave.config import settings

logger = logging.getLogger(__name__)


def smtp_configured() -> bool:
    return bool(settings.smtp_host)


def _from_address() -> str:
    return settings.digest_from_email or settings.smtp_username


def send_email(to: str, subject: str, text: str, html: str) -> None:
    """Deliver one message; raises on failure so callers can react."""
    msg = EmailMessage()
    msg["From"] = _from_address()
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")

    if settings.smtp_port == 465:
        with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=30) as smtp:
            if settings.smtp_username:
                smtp.login(settings.smtp_username, settings.smtp_password)
            smtp.send_message(msg)
    else:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as smtp:
            smtp.starttls()
            if settings.smtp_username:
                smtp.login(settings.smtp_username, settings.smtp_password)
            smtp.send_message(msg)


def _section_text(title: str, items: list[str]) -> str:
    if not items:
        return ""
    lines = "\n".join(f"  - {item}" for item in items)
    return f"\n{title}\n{lines}\n"


def _section_html(title: str, items: list[str], accent: str = "#0e0e0c") -> str:
    if not items:
        return ""
    lis = "".join(
        f'<li style="margin:4px 0;color:#3a3a36;font-size:14px">{escape(item)}</li>'
        for item in items
    )
    return (
        f'<p style="margin:18px 0 6px;font-size:11px;letter-spacing:.08em;'
        f'text-transform:uppercase;color:{accent}">{escape(title)}</p>'
        f'<ul style="margin:0;padding-left:18px">{lis}</ul>'
    )


def render_digest_email(digest: dict, unsubscribe_url: str | None) -> tuple[str, str, str]:
    """Return (subject, text_body, html_body) for one digest payload."""
    headline = digest.get("headline") or "Your memory has something for you"
    focus = digest.get("focus") or []
    commitments = digest.get("commitments") or []
    gaps = digest.get("gaps") or []

    subject = f"ContextWeave nudge — {headline[:70]}"

    text = f"{headline}\n"
    text += _section_text("Focus", focus)
    text += _section_text("Commitments", commitments)
    text += _section_text("Slipping", gaps)
    if unsubscribe_url:
        text += f"\nUnsubscribe: {unsubscribe_url}\n"

    unsub_html = (
        f'<p style="margin-top:28px;font-size:12px;color:#aaa99f">'
        f'<a href="{escape(unsubscribe_url)}" style="color:#aaa99f">Unsubscribe</a> '
        f"from the daily digest.</p>"
        if unsubscribe_url
        else ""
    )
    html = f"""\
<!DOCTYPE html>
<html>
  <body style="margin:0;padding:0;background:#f7f6f2">
    <div style="max-width:560px;margin:0 auto;padding:32px 24px;
                font-family:Georgia,'Times New Roman',serif">
      <p style="font-size:11px;letter-spacing:.14em;text-transform:uppercase;
                color:#c4692a;margin:0 0 10px">ContextWeave &middot; today's nudge</p>
      <h1 style="font-size:26px;font-weight:400;color:#0e0e0c;margin:0 0 8px;
                 line-height:1.25">{escape(headline)}</h1>
      <div style="font-family:Helvetica,Arial,sans-serif">
        {_section_html("Focus", focus)}
        {_section_html("Commitments", commitments)}
        {_section_html("Slipping", gaps, accent="#b8760a")}
        {unsub_html}
      </div>
    </div>
  </body>
</html>
"""
    return subject, text, html
