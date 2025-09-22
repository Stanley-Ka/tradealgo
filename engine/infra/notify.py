from __future__ import annotations

import json
from typing import Any

import requests


def send_slack(webhook_url: str, text: str) -> None:
    payload = {"text": text}
    resp = requests.post(webhook_url, json=payload, timeout=10)
    resp.raise_for_status()


def send_discord(webhook_url: str, text: str) -> None:
    payload = {"content": text}
    resp = requests.post(webhook_url, json=payload, timeout=10)
    resp.raise_for_status()

