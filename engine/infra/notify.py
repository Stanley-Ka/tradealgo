from __future__ import annotations


import requests


def send_slack(webhook_url: str, text: str) -> None:
    payload = {"text": text}
    resp = requests.post(webhook_url, json=payload, timeout=10)
    resp.raise_for_status()


def send_discord(webhook_url: str, text: str) -> None:
    # Allow @everyone mentions explicitly to ensure pings work when enabled
    payload = {
        "content": text,
        "allowed_mentions": {"parse": ["everyone"]},
    }
    resp = requests.post(webhook_url, json=payload, timeout=10)
    resp.raise_for_status()
