from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import pandas as pd  # type: ignore
import requests  # type: ignore
import yaml  # type: ignore


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate preset/config readiness for overnight run"
    )
    p.add_argument("--config", required=True, help="YAML preset/config path")
    p.add_argument(
        "--send-discord-test",
        action="store_true",
        help="Send a small test message to the webhook if present",
    )
    p.add_argument(
        "--discord-webhook", default="", help="Override webhook for test message"
    )
    return p.parse_args(argv)


def _ok(msg: str) -> None:
    print(f"[ok] {msg}")


def _warn(msg: str) -> None:
    print(f"[warn] {msg}")


def _err(msg: str) -> None:
    print(f"[err] {msg}")


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        _err(f"failed to load YAML: {e}")
        sys.exit(1)

    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    cal = cfg.get("calibration", {}) if isinstance(cfg.get("calibration"), dict) else {}
    alert = cfg.get("alert", {}) if isinstance(cfg.get("alert"), dict) else {}

    # Check files
    missing = False
    for key in ("features", "meta_model", "oof", "calibrators_pkl"):
        val = paths.get(key, "")
        if not val:
            _warn(f"paths.{key} not set")
            continue
        if os.path.exists(val):
            _ok(f"found paths.{key}: {val}")
        else:
            _err(f"missing paths.{key}: {val}")
            missing = True
    mc = cal.get("meta_calibrator_pkl", "")
    if mc:
        if os.path.exists(mc):
            _ok(f"found calibration.meta_calibrator_pkl: {mc}")
        else:
            _warn(f"calibration.meta_calibrator_pkl missing: {mc}")

    # Feature freshness
    feat = paths.get("features", "")
    if feat and os.path.exists(feat):
        try:
            dts = pd.read_parquet(feat, columns=["date"])  # type: ignore
            dts["date"] = pd.to_datetime(dts["date"]).dt.tz_localize(None)  # type: ignore
            last = pd.to_datetime(dts["date"]).max()  # type: ignore
            now_naive = pd.Timestamp.utcnow().tz_localize(None).normalize()
            days = (now_naive - last.normalize()).days
            note = f"(stale {days}d)" if days > 2 else ""
            if days > 30:
                _warn(f"features last date: {last.date()} {note}")
            else:
                _ok(f"features last date: {last.date()} {note}")
        except Exception as e:
            _warn(f"could not read features date: {e}")

    # Env checks
    poly = os.environ.get("POLYGON_API_KEY", "")
    if poly:
        _ok("POLYGON_API_KEY is set")
        # Light API probe
        try:
            r = requests.get(
                "https://api.polygon.io/v3/reference/dividends",
                params={"limit": 1},
                headers={"Authorization": f"Bearer {poly}"},
                timeout=10,
            )
            if r.status_code == 200:
                _ok("Polygon API reachable")
            else:
                _warn(f"Polygon API status {r.status_code}: {r.text[:120]}")
        except Exception as e:
            _warn(f"Polygon API probe failed: {e}")
    else:
        _warn("POLYGON_API_KEY not set in environment")

    # Discord webhooks (alerts and trades)
    def _resolve_placeholder(val: str) -> str:
        # Resolve ${VAR} from environment if present; otherwise return empty to allow fallback
        if (
            isinstance(val, str)
            and val.strip().startswith("${")
            and val.strip().endswith("}")
        ):
            key = val.strip()[2:-1]
            return os.environ.get(key, "")
        return val

    raw_alert = alert.get("discord_webhook", "")
    raw_alert = _resolve_placeholder(raw_alert)
    hook_alert = args.discord_webhook or (
        raw_alert
        or os.environ.get(
            "DISCORD_ALERTS_WEBHOOK_URL", os.environ.get("DISCORD_WEBHOOK_URL", "")
        )
    )
    paper = cfg.get("paper", {}) if isinstance(cfg.get("paper"), dict) else {}
    raw_trades = paper.get("discord_webhook", "")
    raw_trades = _resolve_placeholder(raw_trades)
    hook_trades = raw_trades or os.environ.get("DISCORD_TRADES_WEBHOOK_URL", "")
    if hook_alert:
        _ok("Discord alerts webhook configured")
    else:
        _warn("Discord alerts webhook not set (alerts will not send)")
    if hook_trades:
        _ok("Discord trades webhook configured")
    else:
        _warn("Discord trades webhook not set (trade entries will not send)")
    if args.send_discord_test:
        for name, hook in (("alerts", hook_alert), ("trades", hook_trades)):
            if not hook:
                continue
            try:
                resp = requests.post(
                    hook,
                    json={
                        "content": f"Engine test: {name} webhook OK",
                        "allowed_mentions": {"parse": ["everyone"]},
                    },
                    timeout=10,
                )
                if resp.status_code // 100 == 2:
                    _ok(f"Discord {name} test message sent")
                else:
                    _warn(f"Discord {name} test HTTP {resp.status_code}")
            except Exception as e:
                _warn(f"Discord {name} test failed: {e}")

    # Universe file
    uni = alert.get("universe_file", "")
    if uni:
        if os.path.exists(uni):
            _ok(f"universe file: {uni}")
        else:
            _warn(f"universe file missing: {uni}")

    if missing:
        sys.exit(2)


if __name__ == "__main__":
    main()
