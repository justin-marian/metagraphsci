"""Send a training-finished notification email via Gmail SMTP.

Reads credentials from environment variables so secrets never live in the repo:
  GMAIL_USER         — sender Gmail address (e.g. you@gmail.com)
  GMAIL_APP_PASSWORD — 16-char App Password from https://myaccount.google.com/apppasswords

Usage:
    python scripts/notify_email.py <status> <run_dir>
        status:   "ok" or "fail"
        run_dir:  path to the training run output directory
"""
from __future__ import annotations

import json
import os
import socket
import ssl
import sys
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from smtplib import SMTP_SSL

RECIPIENT = "daachirita@gmail.com"


def build_body(status: str, run_dir: Path) -> str:
    lines = [
        f"Status:    {status.upper()}",
        f"Host:      {socket.gethostname()}",
        f"Finished:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Run dir:   {run_dir}",
        "",
    ]
    summary = run_dir / "artifacts" / "run_summary.json"
    if summary.exists():
        try:
            metrics = json.loads(summary.read_text())
            lines.append("Run summary:")
            for k, v in metrics.items():
                lines.append(f"  {k}: {v}")
        except Exception as exc:
            lines.append(f"(could not parse run_summary.json: {exc})")
    else:
        for nested in run_dir.rglob("run_summary.json"):
            lines.append(f"Found summary: {nested}")
            try:
                lines.append(nested.read_text())
            except Exception:
                pass
            break
    return "\n".join(lines)


def main() -> int:
    status = sys.argv[1] if len(sys.argv) > 1 else "ok"
    run_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "runs/openalex_ai_intel")

    user = os.environ.get("GMAIL_USER")
    pwd = os.environ.get("GMAIL_APP_PASSWORD")
    if not user or not pwd:
        print("[notify_email] GMAIL_USER / GMAIL_APP_PASSWORD not set; skipping email.")
        return 0

    msg = EmailMessage()
    msg["Subject"] = f"[MetaGraphSci] training {status.upper()} — {run_dir.name}"
    msg["From"] = user
    msg["To"] = RECIPIENT
    msg.set_content(build_body(status, run_dir))

    try:
        with SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as smtp:
            smtp.login(user, pwd)
            smtp.send_message(msg)
        print(f"[notify_email] sent to {RECIPIENT}")
        return 0
    except Exception as exc:
        print(f"[notify_email] FAILED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
