"""
notify_teams.py
---------------
Minimal Teams webhook notifier for pipeline run status.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


def send_teams_alert(webhook_url: str | None, title: str, facts: dict[str, Any], ok: bool) -> bool:
    if not webhook_url:
        return False

    color = "2EB886" if ok else "D64545"
    fact_items = [{"name": str(k), "value": str(v)} for k, v in facts.items()]
    payload = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": color,
        "summary": title,
        "title": title,
        "sections": [
            {
                "facts": fact_items,
                "markdown": True,
            }
        ],
    }

    try:
        r = requests.post(webhook_url, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=10)
        r.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("Teams alert failed: %s", exc)
        return False

