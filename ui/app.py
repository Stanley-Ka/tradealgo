from __future__ import annotations

import os
import io
import glob
import subprocess
import time
import sys
from typing import Dict, Any

import pandas as pd
import requests  # type: ignore
import yaml  # type: ignore
import re
import streamlit as st  # type: ignore

import pathlib

# Ensure project root is on sys.path so `engine` is importable when running `streamlit run ui/app.py`
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.infra.yaml_config import load_yaml_config
from engine.tools.summarize import summarize_preset


st.set_page_config(page_title="Trading Engine UI", layout="wide")
st.title("Trading Engine – Research & Alerts")
try:
    from engine.infra.market_time import today_session, is_open, DEFAULT_CAL

    now = pd.Timestamp.now(tz="UTC").astimezone()
    sess = today_session(DEFAULT_CAL)
    tzname = now.tzname() or "local"
    if sess is not None:
        open_s = sess.open_ts.strftime("%Y-%m-%d %H:%M:%S %Z")
        close_s = sess.close_ts.strftime("%Y-%m-%d %H:%M:%S %Z")
        badge = "🟢 Open" if is_open(DEFAULT_CAL) else "🔴 Closed"
        st.caption(
            f"Timezone: {tzname} • {badge} • Today: open {open_s} → close {close_s}"
        )
    else:
        st.caption(f"Timezone: {tzname} • Market closed today")
except Exception:
    pass


def _parse_percent(text: str, default_frac: float) -> float:
    try:
        s = (text or "").strip()
        if not s:
            return float(default_frac)
        has_pct = s.endswith("%")
        if has_pct:
            s = s[:-1]
        s = re.sub(r"[,\s]", "", s)
        val = float(s)
        return max(
            0.0, min(1.0, val / 100.0 if (has_pct or val > 1.0) else val / 100.0)
        )
    except Exception:
        return float(default_frac)


@st.cache_data(show_spinner=False)
def load_config(path: str) -> Dict[str, Any]:
    return load_yaml_config(path)


def run_module(module: str, args: list[str]) -> str:
    cmd = [sys.executable, "-m", module] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stdout + ("\n" + proc.stderr if proc.stderr else "")


def run_with_progress(
    module: str, args: list[str], label: str = "Working…", est_seconds: int = 60
) -> str:
    """Run a Python module with a rough progress indicator.

    This does not parse fine‑grained progress from the tool; instead, it
    advances up to 95% over `est_seconds` while the process runs, then
    completes to 100% when done. It returns combined stdout/stderr.
    """
    st.write(label)
    bar = st.progress(0)
    start = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-m", module, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = "", ""
    # Rough, time‑based progress
    while proc.poll() is None:
        elapsed = time.time() - start
        pct = min(95, int(100 * elapsed / max(1, est_seconds)))
        bar.progress(pct)
        time.sleep(0.25)
    stdout, stderr = proc.communicate()
    bar.progress(100)
    return (stdout or "") + ("\n" + stderr if stderr else "")


def run_ps(script_path: str, args: list[str]) -> str:
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        script_path,
        *args,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, shell=False)
    return proc.stdout + ("\n" + proc.stderr if proc.stderr else "")


def run_ps_cmd(cmdline: str) -> str:
    """Run a raw PowerShell command line (for schtasks) and return combined output."""
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        cmdline,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, shell=False)
    return proc.stdout + ("\n" + proc.stderr if proc.stderr else "")


# Persistent UI settings (e.g., account)
UI_SETTINGS_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "data" / "ui_settings.yaml"
)


def load_ui_settings() -> dict:
    try:
        if UI_SETTINGS_PATH.exists():
            with open(UI_SETTINGS_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        return {}
    return {}


def save_ui_settings(cfg: dict) -> None:
    try:
        UI_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(UI_SETTINGS_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except Exception:
        pass


def _count_universe_symbols(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for ln in f if ln.strip() and not ln.strip().startswith("#"))
    except Exception:
        return 0


def _task_next_run(task_name: str) -> str:
    try:
        ps = (
            f"$t = Get-ScheduledTask -TaskName '{task_name}' -ErrorAction SilentlyContinue;"
            "$i = if ($t) { Get-ScheduledTaskInfo $t } else { $null };"
            "if ($i) { $i.NextRunTime } else { 'N/A' }"
        )
        out = run_ps_cmd(ps).strip()
        return out.splitlines()[-1].strip() if out else "N/A"
    except Exception:
        return "N/A"


def _pid_running_windows(pid: int) -> bool:
    try:
        out = run_ps_cmd(f"tasklist /FI 'PID eq {pid}'")
        return str(pid) in out
    except Exception:
        return True  # if unsure, assume running


def _prefer_local_yaml(rel_path: str) -> str:
    base = ROOT / rel_path
    if base.suffix != ".yaml":
        return rel_path
    local = base.with_name(base.stem + ".local.yaml")
    if local.exists():
        return local.relative_to(ROOT).as_posix()
    return rel_path


friendly_presets = {
    "Swing – Aggressive": _prefer_local_yaml("engine/presets/swing_aggressive.yaml"),
    "Swing – Conservative": _prefer_local_yaml(
        "engine/presets/swing_conservative.yaml"
    ),
    # Advanced users can still select a full config explicitly if desired:
    "Config (advanced)": _prefer_local_yaml("engine/config.research.yaml"),
}

with st.sidebar:
    st.header("Preset")
    preset_label = st.selectbox(
        "Choose a trading style",
        list(friendly_presets.keys()),
        index=0,
        help="Pick a simple preset; advanced users can edit the YAML later",
    )
    preset = friendly_presets[preset_label]
    st.caption("Using: " + preset)
    mode = st.radio(
        "Mode",
        ["Optimise", "Trade", "Analysis", "Settings", "Logs", "Overview", "Guided"],
        index=0,
    )
    st.divider()
    default_hook = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if mode == "Settings":
        st.header("Notifications")
        webhook = st.text_input("Discord webhook (optional)", value=default_hook)
        test_msg = st.text_input("Test message", value="test")
        if st.button("Send Discord test"):
            try:
                if not webhook:
                    raise RuntimeError("No webhook provided")
                resp = requests.post(
                    webhook, json={"content": test_msg or "test"}, timeout=10
                )
                resp.raise_for_status()
                st.success("Test message sent to Discord.")
            except Exception as e:
                st.error(f"Discord test failed: {e}")
    else:
        webhook = default_hook


cfg = load_config(preset)
features = cfg.get("paths", {}).get(
    "features", "data/datasets/features_daily_1D.parquet"
)
model_out = cfg.get("paths", {}).get("meta_model", "data/models/meta_hgb.pkl")
meta_pred = cfg.get("paths", {}).get("meta", "data/datasets/meta_predictions.parquet")
oof_path_default = cfg.get("paths", {}).get(
    "oof", "data/datasets/oof_specialists.parquet"
)
calibs_out_default = cfg.get("calibration", {}).get(
    "calibrators_pkl", "data/models/spec_calibrators.pkl"
)
universe_default = cfg.get("alert", {}).get(
    "universe_file", "engine/data/universe/nasdaq100.example.txt"
)
universe_map = {
    "NASDAQ‑100 (example)": "engine/data/universe/nasdaq100.example.txt",
    "All US stocks": "engine/data/universe/us_all.txt",
    "Your watchlist": "engine/data/universe/watchlist.txt",
    "Custom…": None,
}


def _default_universe_label(p: str) -> str:
    for k, v in universe_map.items():
        if v == p:
            return k
    return "NASDAQ‑100 (example)"


if mode == "Overview":
    st.subheader("Preset Summary")
    st.info(
        "Start the unattended overnight workflow with a couple of clicks. Use Parity mode for deterministic tests (no live sends)."
    )
    st.code(summarize_preset(cfg))
    st.subheader("Quick Start Overnight")
    ov_col1, ov_col2 = st.columns(2)
    with ov_col1:
        start_intr = st.checkbox("Start Intraday Snapshot", value=True)
        every = st.number_input("Alert every (mins)", value=5, min_value=1, step=1)
        minp = st.number_input("Min price", value=1.0, min_value=0.0)
        maxp = st.number_input("Max price (0=none)", value=20.0, min_value=0.0)
        heartbeat = st.checkbox("Discord heartbeat", value=True)
        build_wl = st.checkbox("Build Watchlist", value=True)
        wl_top = st.number_input("Watchlist Top-K", value=500, min_value=50, step=50)
        wl_minadv = st.number_input(
            "Watchlist min ADV (USD)", value=5_000_000, step=500_000
        )
        rebuild_wl = st.checkbox("Rebuild watchlist per trigger", value=True)
        wl_bucket = st.checkbox("Diversify watchlist by ADV buckets", value=True)
        wl_buckets = st.number_input("ADV buckets", value=3, min_value=2, step=1)
        mix_intr = st.checkbox("Mix intraday into ranking", value=False)
        mix_w = st.slider(
            "Mix weight (0..1)", min_value=0.0, max_value=1.0, value=0.3, step=0.05
        )
        parity = st.checkbox(
            "Parity mode (deterministic test)",
            value=False,
            help="Disables cooldown/exploration, uses feature prices, and suppresses sends",
        )
        if st.button("Start Overnight"):
            try:
                script = str((ROOT / "scripts" / "start-overnight.ps1").resolve())
                base = [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    f'"{script}"',
                ]
                args = [
                    "-Config",
                    f'"{preset}"',
                    "-Features",
                    f'"{features}"',
                    "-Model",
                    f'"{model_out}"',
                    "-Universe",
                    f'"{cfg.get("alert", {}).get("universe_file", "engine/data/universe/nasdaq100.example.txt")}"',
                    "-AlertEvery",
                    str(int(every)),
                    "-MinPrice",
                    str(float(minp)),
                    "-MaxPrice",
                    str(float(maxp)),
                ]
                if heartbeat:
                    args += ["-Heartbeat"]
                if start_intr:
                    args += ["-StartIntraday", "-IntradayEvery", str(int(every))]
                if build_wl:
                    args += [
                        "-BuildWatchlist",
                        "-WatchlistOut",
                        f'"{str((ROOT / "engine" / "data" / "universe" / "watchlist.txt").resolve())}"',
                        "-WatchTopK",
                        str(int(wl_top)),
                        "-WatchMinPrice",
                        str(float(minp)),
                        "-WatchMinADV",
                        str(float(wl_minadv)),
                    ]
                if rebuild_wl:
                    args += ["-RebuildWatchlist"]
                    if wl_bucket:
                        args += [
                            "-WatchBucketAdv",
                            "-WatchBuckets",
                            str(int(wl_buckets)),
                        ]
                if mix_intr:
                    # start-overnight passes these to rt-alert
                    args += ["-MixIntraday", "-MixWeight", str(float(mix_w))]
                if parity:
                    args += ["-Parity"]
                # Launch via a single command string to avoid Win32 arg parsing issues
                cmd = " ".join(base + args)
                subprocess.Popen(cmd, shell=True)
                st.success("Overnight processes started.")
            except Exception as e:
                st.error(f"Failed to start overnight: {e}")
    with ov_col2:
        st.subheader("Run Presets")
        st.caption("One‑click start with preset YAMLs.")
        aggr_options = list(
            dict.fromkeys(
                [
                    _prefer_local_yaml("engine/presets/swing_aggressive.yaml"),
                    "engine/presets/swing_aggressive.yaml",
                ]
            )
        )
        c_aggr = st.selectbox(
            "Aggressive preset",
            options=aggr_options,
            index=0,
        )
        cons_options = list(
            dict.fromkeys(
                [
                    _prefer_local_yaml("engine/presets/swing_conservative.yaml"),
                    "engine/presets/swing_conservative.yaml",
                ]
            )
        )
        c_cons = st.selectbox(
            "Conservative preset",
            options=cons_options,
            index=0,
        )
        q_every = st.selectbox("Alert every (mins)", options=[5, 10, 15], index=0)
        q_times = st.selectbox(
            "Entry times", options=["09:35,15:55", "10:00,15:30"], index=0
        )
        colp1, colp2 = st.columns(2)
        with colp1:
            if st.button("Start Overnight (Aggressive)"):
                try:
                    script = str((ROOT / "scripts" / "overnight.ps1").resolve())
                    base = [
                        "powershell.exe",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        f'"{script}"',
                    ]
                    args = [
                        "-Config",
                        f'"{c_aggr}"',
                        "-AlertEvery",
                        str(int(q_every)),
                        "-EntryTimes",
                        f'"{q_times}"',
                        "-StartManager",
                    ]
                    cmd = " ".join(base + args)
                    subprocess.Popen(cmd, shell=True)
                    st.success("Aggressive overnight started.")
                except Exception as e:
                    st.error(f"Failed: {e}")
        st.divider()
        if st.button("Start Parity Overnight (Current preset)"):
            try:
                script = str((ROOT / "scripts" / "start-overnight.ps1").resolve())
                base = [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    f'"{script}"',
                ]
                args = [
                    "-Config",
                    f'"{preset}"',
                    "-AlertEvery",
                    "5",
                    "-EntryTimes",
                    '"09:35,15:55"',
                    "-Parity",
                ]
                cmd = " ".join(base + args)
                subprocess.Popen(cmd, shell=True)
                st.success("Parity overnight started (no sends, deterministic).")
            except Exception as e:
                st.error(f"Failed to start parity overnight: {e}")
        with colp2:
            if st.button("Start Overnight (Conservative)"):
                try:
                    script = str((ROOT / "scripts" / "overnight.ps1").resolve())
                    base = [
                        "powershell.exe",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        f'"{script}"',
                    ]
                    args = [
                        "-Config",
                        f'"{c_cons}"',
                        "-AlertEvery",
                        str(int(q_every)),
                        "-EntryTimes",
                        f'"{q_times}"',
                        "-StartManager",
                    ]
                    cmd = " ".join(base + args)
                    subprocess.Popen(cmd, shell=True)
                    st.success("Conservative overnight started.")
                except Exception as e:
                    st.error(f"Failed: {e}")
    with st.expander("Help", expanded=False):
        st.markdown(
            """
        - Pick a preset, then click Start Overnight to launch the alert loop and entry scheduler.
        - Use Parity to test deterministically (no live sends, no randomness).
        - Run Presets buttons start with common configs in one click.
        """
        )
    st.subheader("Monitoring")
    mon1, mon2, mon3 = st.columns(3)
    with mon1:
        st.caption("Real‑time status")
        pid_path = os.path.join("data/alerts", "rt.pid")
        if os.path.exists(pid_path):
            try:
                pid = int(open(pid_path, "r", encoding="utf-8").read().strip())
                running = _pid_running_windows(pid) if os.name == "nt" else True
                st.metric("RT PID", f"{pid}", delta=("running" if running else "stale"))
            except Exception:
                st.caption("PID file present")
        else:
            st.caption("No PID file (RT not started)")
    with mon2:
        st.caption("Next Overnight task run")
        next_run = _task_next_run("Engine-Overnight")
        st.metric("Next run", next_run)
    with mon3:
        st.caption("Last picks")
        diag = os.path.join("data/alerts", "alert_diag.csv")
        try:
            if os.path.exists(diag):
                df = pd.read_csv(diag)
                last = df.tail(5)
                st.dataframe(last)
            else:
                st.caption("No diagnostics yet.")
        except Exception as e:
            st.caption(f"Failed to load: {e}")


if mode == "Data":
    st.subheader("Build Dataset & Features")
    st.info(
        "Build your historical dataset (with cost‑aware labels) and feature parquet. Most users only need Provider, Universe, and Start date. Advanced lets you set End and custom output paths."
    )
    colA, colB = st.columns(2)
    with colA:
        prov = st.selectbox(
            "Data source",
            options=["yahoo", "alphavantage", "polygon"],
            index=0,
            help="Where to download daily prices from",
        )
        uni_label = st.selectbox(
            "Which symbols?",
            options=list(universe_map.keys()),
            index=list(universe_map.keys()).index(
                _default_universe_label(universe_default)
            ),
            help="Pick a built‑in list or choose Custom",
        )
        uni = (
            st.text_input(
                "Custom list (one symbol per line)",
                value=universe_default,
                help="Path to your own list of symbols",
            )
            if universe_map[uni_label] is None
            else universe_map[uni_label]
        )
        start_choice = st.selectbox(
            "Start date",
            options=["2010-01-01", "2015-01-01", "2020-01-01", "Custom…"],
            index=1,
            help="How far back to build data",
        )
        start = (
            st.text_input("Custom start (YYYY-MM-DD)", value="2015-01-01")
            if start_choice == "Custom…"
            else start_choice
        )
        label_thr_bps = st.selectbox(
            "Minimum move to count as 'up' (bps)",
            options=[0, 5, 10, 15, 20],
            index=2,
            help="Protects against tiny moves counting as wins",
        )
        if st.button("Build Dataset"):
            args = [
                "--provider",
                prov,
                "--universe-file",
                uni,
                "--start",
                start,
                "--label-horizon",
                "1",
                "--label-threshold-bps",
                str(int(label_thr_bps)),
            ]
            with st.spinner("Building dataset…"):
                out = run_module("engine.data.build_dataset", args)
            st.text(out)
        with st.expander("Advanced", expanded=False):
            end = st.text_input("End (YYYY-MM-DD, blank=all)", value="")
            if st.button("Build Dataset (with end)"):
                args = [
                    "--provider",
                    prov,
                    "--universe-file",
                    uni,
                    "--start",
                    start,
                    *(["--end", end] if end.strip() else []),
                    "--label-horizon",
                    "1",
                    "--label-threshold-bps",
                    str(int(label_thr_bps)),
                ]
                with st.spinner("Building dataset…"):
                    st.text(run_module("engine.data.build_dataset", args))
    with colB:
        if st.button("Build Features"):
            args = [
                "--universe-file",
                uni,
                "--start",
                start,
                "--out",
                features,
            ]
            with st.spinner("Building features…"):
                out = run_module("engine.features.build_features", args)
            st.text(out)
        with st.expander("Advanced", expanded=False):
            feat_out = st.text_input("Features out", value=features)
            if st.button("Build Features (custom out)"):
                args = ["--universe-file", uni, "--start", start, "--out", feat_out]
                with st.spinner("Building features…"):
                    st.text(run_module("engine.features.build_features", args))


if mode == "Research":
    st.info(
        "Run cross‑validation + calibration and train the meta‑learner (HGB) with one click. Then backtest and build a watchlist."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Weekly Train (CV + Meta)")
        oof_path = oof_path_default
        calibs_out = calibs_out_default
        news = cfg.get("paths", {}).get("news_sentiment", "")
        if st.button("Run Weekly Train", type="primary"):
            # CV
            args_cv = [
                "--features",
                features,
                "--label",
                "label_up_1d",
                "--cv-scheme",
                "time_kfold",
                "--kfolds",
                "5",
                "--purge-days",
                "3",
                "--embargo-days",
                "3",
                "--calibration",
                cfg.get("calibration", {}).get("kind", "isotonic"),
                "--out",
                oof_path,
                "--calibrators-out",
                calibs_out,
                "--spec-config",
                preset,
            ]
            if news:
                args_cv += ["--news-sentiment", news]
            with st.spinner("Running CV…"):
                out1 = run_module("engine.models.run_cv", args_cv)
            # Meta
            args_meta = [
                "--oof",
                oof_path,
                "--train-folds",
                "all-but-last:1",
                "--test-folds",
                "last:1",
                "--model",
                "hgb",
                "--hgb-learning-rate",
                "0.05",
                "--hgb-max-iter",
                "400",
                "--meta-calibration",
                "isotonic",
                "--replace-prob-with-calibrated",
                "--out",
                meta_pred,
                "--model-out",
                model_out,
            ]
            with st.spinner("Training meta…"):
                out2 = run_module("engine.models.train_meta", args_meta)
            st.text(out1 + "\n" + out2)
        with st.expander("Advanced (CV only)", expanded=False):
            if st.button("Run CV only"):
                args = [
                    "--features",
                    features,
                    "--label",
                    "label_up_1d",
                    "--cv-scheme",
                    "time_kfold",
                    "--kfolds",
                    "5",
                    "--purge-days",
                    "3",
                    "--embargo-days",
                    "3",
                    "--calibration",
                    cfg.get("calibration", {}).get("kind", "isotonic"),
                    "--out",
                    oof_path,
                    "--calibrators-out",
                    calibs_out,
                    "--spec-config",
                    preset,
                ]
                if news:
                    args += ["--news-sentiment", news]
                with st.spinner("Running CV…"):
                    st.text(run_module("engine.models.run_cv", args))
    with col2:
        st.subheader("Backtest")
        topk = st.selectbox(
            "Top-K", options=[10, 15, 20, 25, 30], index=2, help="Names per rebalance"
        )
        cost = st.selectbox(
            "Cost (bps)", options=[0, 2, 5, 10], index=2, help="Transaction cost"
        )
        if st.button("Run Backtest"):
            args = [
                "--features",
                features,
                "--pred",
                meta_pred,
                "--prob-col",
                "meta_prob",
                "--top-k",
                str(topk),
                "--cost-bps",
                str(cost),
                "--rebalance",
                "weekly",
                "--rebal-weekday",
                "MON",
                "--report-html",
                "data/backtests/daily_topk_report.html",
            ]
            with st.spinner("Backtesting…"):
                out = run_module("engine.backtest.simple_daily", args)
            st.text(out)
    with st.expander("Build Watchlist", expanded=False):
        wl_out = st.text_input("Out path", value="engine/data/universe/watchlist.txt")
        wl_top = st.number_input("Top-K", value=500, min_value=10, step=10)
        wl_minp = st.number_input("Min price", value=1.0, min_value=0.0)
        wl_minadv = st.number_input("Min ADV (USD)", value=5_000_000, step=500_000)
        use_pred = st.checkbox("Use predictions parquet if available", value=False)
        wl_pred = (
            st.text_input(
                "Pred parquet (optional)", value=cfg.get("paths", {}).get("meta", "")
            )
            if use_pred
            else ""
        )
        if st.button("Build watchlist"):
            args = [
                "--features",
                features,
                *(
                    ["--pred", wl_pred]
                    if (use_pred and wl_pred)
                    else ["--model-pkl", model_out]
                ),
                *(["--oof", oof_path_default] if not use_pred else []),
                "--calibrators-pkl",
                calibs_out_default,
                "--out",
                wl_out,
                "--top-k",
                str(int(wl_top)),
                "--min-price",
                str(float(wl_minp)),
                "--min-adv-usd",
                str(float(wl_minadv)),
            ]
            with st.spinner("Building watchlist…"):
                out = run_module("engine.tools.build_watchlist", args)
            st.text(out)

    with st.expander("Intraday (1m) Demo – Advanced", expanded=False):
        bars_root = st.text_input("Bars root", value="data/equities/polygon")
        interval = st.selectbox("Interval", options=["1m", "5m"], index=0)
        lookback = st.number_input("Lookback bars", value=200, min_value=50, step=10)
        snap_out = st.text_input(
            "Snapshot out", value="data/datasets/features_intraday_latest.parquet"
        )
        symbols = st.text_input("Scaffold symbols (dry-run)", value="MSFT,AAPL")
        if st.button("Scaffold bars (dry-run)"):
            out = run_module(
                "engine.data.polygon_stream_bars",
                [
                    "--symbols",
                    symbols,
                    "--interval",
                    interval,
                    "--out-root",
                    bars_root,
                    "--dry-run",
                ],
            )
            st.text(out)
        if st.button("Build snapshot"):
            out = run_module(
                "engine.tools.build_intraday_latest",
                [
                    "--bars-root",
                    bars_root,
                    "--interval",
                    interval,
                    "--lookback-bars",
                    str(lookback),
                    "--out",
                    snap_out,
                ],
            )
            st.text(out)


if mode == "Alerts":
    st.subheader("Trade Alert")
    st.info(
        "Preview or send today’s top‑K picks based on your model. Choose a price source: feature (deterministic) or live (for sizing). Advanced holds optional filters and intraday mix."
    )
    provider = st.selectbox(
        "News provider",
        options=["none", "polygon", "finnhub"],
        index=0,
        help="Source for fresh news sentiment",
    )
    topk_alert = st.selectbox(
        "Top-K", options=[1, 2, 3, 4, 5], index=2, help="Number of symbols to alert"
    )
    price_src = st.selectbox(
        "Price source",
        options=["feature", "live"],
        index=0,
        help="Reference price for sizing/messages",
    )
    live_prov = st.selectbox(
        "Live provider",
        options=["yahoo", "polygon"],
        index=1,
        help="Provider for live prices if enabled",
    )
    acct_eq = int(os.environ.get("ACCOUNT_EQUITY", 100000))
    risk_pct = 0.005
    stop_mult = 1.0
    entry_px = "close"
    colA, colB = st.columns(2)
    with st.expander("Advanced (filters & options)", expanded=False):
        min_price = st.selectbox(
            "Min price ($)",
            options=[0, 1, 5, 10, 20],
            index=0,
            key="al_min_price",
            help="Filter out low-priced names",
        )
        max_price = st.selectbox(
            "Max price ($)",
            options=[0, 50, 100, 200, 500, 1000],
            index=0,
            key="al_max_price",
            help="0 = no maximum",
        )
        heartbeat = st.checkbox(
            "Send heartbeat if empty",
            value=True,
            key="al_heartbeat",
            help="Notify even when no picks pass risk gates",
        )
        mix_intr = st.checkbox(
            "Mix intraday snapshot",
            value=False,
            key="al_mix_intr",
            help="Blend simple intraday score with daily meta prob",
        )
        mix_w = st.slider(
            "Mix weight",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            key="al_mix_w",
            help="0 = ignore intraday; 1 = intraday only",
        )
        intr_feat = st.text_input(
            "Intraday features parquet",
            value="data/datasets/features_intraday_latest.parquet",
            key="al_intr_feat",
            help="Latest intraday snapshot parquet path",
        )
        if st.button("Reset Advanced"):
            st.session_state.al_min_price = 0
            st.session_state.al_max_price = 0
            st.session_state.al_heartbeat = True
            st.session_state.al_mix_intr = False
            st.session_state.al_mix_w = 0.3
            st.session_state.al_intr_feat = (
                "data/datasets/features_intraday_latest.parquet"
            )
    with colA:
        if st.button("Run Alert", type="primary"):
            args = [
                "--config",
                preset,
                "--features",
                features,
                "--model-pkl",
                model_out,
                "--oof",
                oof_path_default,
                "--calibrators-pkl",
                calibs_out_default,
                "--universe-file",
                cfg.get("alert", {}).get(
                    "universe_file", "engine/data/universe/nasdaq100.example.txt"
                ),
                "--provider",
                provider,
                "--top-k",
                str(topk_alert),
                "--price-source",
                price_src,
                "--live-provider",
                live_prov,
                "--debug-risk",
                "--account-equity",
                str(acct_eq),
                "--risk-pct",
                str(risk_pct),
                "--stop-atr-mult",
                str(stop_mult),
                "--entry-price",
                entry_px,
            ]
            if min_price and float(min_price) > 0:
                args += ["--min-price", str(min_price)]
            if max_price and float(max_price) > 0:
                args += ["--max-price", str(max_price)]
            if heartbeat:
                args += ["--heartbeat-on-empty"]
            if mix_intr:
                args += [
                    "--mix-intraday",
                    str(float(mix_w)),
                    "--intraday-features",
                    intr_feat,
                ]
            if webhook:
                args += ["--discord-webhook", webhook]
            with st.spinner("Running alert…"):
                out = run_module("engine.tools.trade_alert", args)
            st.text(out)
        if st.button("Preview Alert"):
            args = [
                "--config",
                preset,
                "--features",
                features,
                "--model-pkl",
                model_out,
                "--oof",
                oof_path_default,
                "--calibrators-pkl",
                calibs_out_default,
                "--universe-file",
                cfg.get("alert", {}).get(
                    "universe_file", "engine/data/universe/nasdaq100.example.txt"
                ),
                "--provider",
                provider,
                "--top-k",
                str(topk_alert),
                "--price-source",
                price_src,
                "--live-provider",
                live_prov,
                "--debug-risk",
                "--account-equity",
                str(acct_eq),
                "--risk-pct",
                str(risk_pct),
                "--stop-atr-mult",
                str(stop_mult),
                "--entry-price",
                entry_px,
                "--dry-run",
            ]
            if min_price and float(min_price) > 0:
                args += ["--min-price", str(min_price)]
            if max_price and float(max_price) > 0:
                args += ["--max-price", str(max_price)]
            if heartbeat:
                args += ["--heartbeat-on-empty"]
            if mix_intr:
                args += [
                    "--mix-intraday",
                    str(float(mix_w)),
                    "--intraday-features",
                    intr_feat,
                ]
            with st.spinner("Building preview…"):
                out = run_module("engine.tools.trade_alert", args)
            with st.expander("Discord message preview"):
                st.text(out)
    with colB:
        st.info("Tip: Set FINNHUB_API_KEY or POLYGON_API_KEY in env for news/prices.")


if mode == "Paper":
    st.subheader("Paper Trader")
    st.info(
        "Update or simulate your paper portfolio using the current model. See ledger and reset if you want a fresh start."
    )
    col1, col2 = st.columns(2)
    with col1:
        state_dir = st.text_input("State dir", value="data/paper")
        topk = int(cfg.get("top_k", cfg.get("alert", {}).get("top_k", 20)))
        cost_bps = 5
        turnover_cap = 0.3
        initial_equity = int(os.environ.get("ACCOUNT_EQUITY", 1_000_000))
        if st.button("Step Paper Trader"):
            args = [
                "--features",
                features,
                "--universe-file",
                cfg.get("alert", {}).get(
                    "universe_file", "engine/data/universe/nasdaq100.example.txt"
                ),
                "--top-k",
                str(topk),
                "--state-dir",
                state_dir,
                "--cost-bps",
                str(cost_bps),
                "--initial-equity",
                str(initial_equity),
                "--config",
                preset,
            ]
            pred_path = cfg.get("paths", {}).get(
                "meta", "data/datasets/meta_predictions.parquet"
            )
            if os.path.exists(pred_path):
                args += ["--pred", pred_path]
            else:
                args += [
                    "--model-pkl",
                    model_out,
                    "--oof",
                    cfg.get("paths", {}).get("oof", ""),
                    "--calibrators-pkl",
                    cfg.get("calibration", {}).get("calibrators_pkl", ""),
                ]
            r = cfg.get("risk", {})
            args += [
                "--min-adv-usd",
                str(r.get("min_adv_usd", 1e7)),
                "--max-atr-pct",
                str(r.get("max_atr_pct", 0.05)),
            ]
            if turnover_cap and turnover_cap > 0:
                args += ["--turnover-cap", str(turnover_cap)]
            if webhook:
                args += ["--discord-webhook", webhook]
            with st.spinner("Stepping paper trader…"):
                out = run_module("engine.tools.paper_trader", args)
            with st.expander("Paper trader log"):
                st.text(out)
        if st.button("Run to end"):
            args = [
                "--features",
                features,
                "--universe-file",
                cfg.get("alert", {}).get(
                    "universe_file", "engine/data/universe/nasdaq100.example.txt"
                ),
                "--top-k",
                str(topk),
                "--state-dir",
                state_dir,
                "--cost-bps",
                str(cost_bps),
                "--initial-equity",
                str(initial_equity),
                "--config",
                preset,
                "--pred",
                cfg.get("paths", {}).get(
                    "meta", "data/datasets/meta_predictions.parquet"
                ),
                "--run-to-end",
            ]
            r = cfg.get("risk", {})
            args += [
                "--min-adv-usd",
                str(r.get("min_adv_usd", 1e7)),
                "--max-atr-pct",
                str(r.get("max_atr_pct", 0.05)),
            ]
            with st.spinner("Running to end…"):
                out = run_module("engine.tools.paper_trader", args)
            with st.expander("Paper trader log"):
                st.text(out)
    with col2:
        if os.path.exists(os.path.join(state_dir, "ledger.parquet")):
            led = pd.read_parquet(os.path.join(state_dir, "ledger.parquet"))
            st.line_chart(led.set_index("date")["equity"])
            st.dataframe(led.tail(10))
        if st.button("Reset state"):
            try:
                for fn in ["ledger.parquet", "weights.parquet"]:
                    p = os.path.join(state_dir, fn)
                    if os.path.exists(p):
                        os.remove(p)
                st.success("State reset.")
            except Exception as e:
                st.error(f"Reset failed: {e}")


if mode == "Real-Time":
    st.subheader("Real-Time Alerts")
    st.info(
        "Run the market‑hours scheduler that triggers alerts at specific times or intervals. Parity mode disables live sends and randomness for testing."
    )
    col1, col2 = st.columns(2)
    with col1:
        times = st.text_input(
            "Alert times (HH:MM, comma-separated)", value="09:35,15:55"
        )
        rt_poll = st.number_input("Poll seconds", value=15, min_value=1, step=1)
        rt_state = st.text_input(
            "Alert log path", value="data/alerts/alert_log.parquet"
        )
        rt_provider = st.selectbox(
            "News provider", options=["none", "polygon", "finnhub"], index=0
        )
        rt_price_src = st.selectbox(
            "Price source", options=["feature", "live"], index=0
        )
        rt_live_provider = st.selectbox(
            "Live provider", options=["yahoo", "polygon"], index=1
        )
        rt_topk = st.number_input(
            "Top-K per trigger",
            value=int(cfg.get("alert", {}).get("top_k", 3)),
            min_value=1,
            step=1,
        )
        rt_cooldown = st.number_input(
            "Cooldown minutes", value=120, min_value=0, step=5
        )
        rt_explore = st.slider(
            "Explore prob", min_value=0.0, max_value=0.2, value=0.0, step=0.01
        )
        rt_parity = st.checkbox(
            "Parity mode (deterministic)",
            value=False,
            help="Sets provider=none, price=feature, cooldown=0, explore=0, no sends",
        )
        if st.button("Start real-time alerts"):
            args = [
                "--features",
                features,
                "--model-pkl",
                model_out,
                "--oof",
                oof_path_default,
                "--calibrators-pkl",
                calibs_out_default,
                "--universe-file",
                cfg.get("alert", {}).get(
                    "universe_file", "engine/data/universe/nasdaq100.example.txt"
                ),
                "--provider",
                ("none" if rt_parity else rt_provider),
                "--top-k",
                str(int(rt_topk)),
                "--times",
                times,
                "--poll",
                str(rt_poll),
                "--state",
                rt_state,
                "--price-source",
                ("feature" if rt_parity else rt_price_src),
                "--live-provider",
                rt_live_provider,
                "--cooldown-mins",
                str(0 if rt_parity else int(rt_cooldown)),
                "--explore-prob",
                str(0.0 if rt_parity else float(rt_explore)),
            ]
            if (not rt_parity) and webhook:
                args += ["--discord-webhook", webhook]
            cmd = [sys.executable, "-m", "engine.tools.real_time_alert", *args]
            try:
                os.makedirs(os.path.dirname(rt_state), exist_ok=True)
                proc = subprocess.Popen(cmd)
                pid_path = os.path.join(os.path.dirname(rt_state), "rt.pid")
                with open(pid_path, "w", encoding="utf-8") as f:
                    f.write(str(proc.pid))
                st.success(
                    f"Real-time alerts started (PID {proc.pid}). Times are local."
                )
            except Exception as e:
                st.error(f"Failed to start real-time alerts: {e}")
    with col2:
        if st.button("Stop real-time alerts"):
            try:
                pid_path = os.path.join(os.path.dirname(rt_state), "rt.pid")
                if os.path.exists(pid_path):
                    pid = int(open(pid_path, "r", encoding="utf-8").read().strip())
                    try:
                        if os.name == "posix":
                            os.kill(pid, 15)
                        else:
                            import signal

                            os.kill(pid, signal.SIGTERM)
                        st.success(f"Sent stop signal to PID {pid}.")
                    except Exception as e:
                        st.warning(f"Could not signal PID {pid}: {e}")
                else:
                    st.info("No PID file found; process may not be running.")
            except Exception as e:
                st.error(f"Stop failed: {e}")

    st.subheader("Single Trigger (Parity Test)")
    par_col1, _ = st.columns(2)
    with par_col1:
        if st.button("Trigger once now --force"):
            diag_path = "data/alerts/ui_parity_rt.csv"
            args = [
                "--config",
                preset,
                "--features",
                features,
                "--model-pkl",
                model_out,
                "--oof",
                oof_path_default,
                "--calibrators-pkl",
                calibs_out_default,
                "--universe-file",
                cfg.get("alert", {}).get(
                    "universe_file", "engine/data/universe/nasdaq100.example.txt"
                ),
                "--provider",
                "none",
                "--top-k",
                str(int(cfg.get("alert", {}).get("top_k", 3))),
                "--alert-log-csv",
                diag_path,
                "--cooldown-mins",
                "0",
                "--price-source",
                "feature",
                "--force",
            ]
            with st.spinner("Triggering…"):
                out = run_module("engine.tools.real_time_alert", args)
            st.text(out)


if mode == "Positions":
    st.subheader("Open Positions")
    st.info(
        "Inspect open positions, compute unrealized PnL via Polygon, review recent actions, and view the paper equity curve."
    )
    pos_path = st.text_input("Positions CSV", value="data/paper/positions.csv")
    log_path = st.text_input("Trade Log CSV", value="data/paper/trade_log.csv")
    ledger_path = st.text_input(
        "Ledger Parquet (paper)", value="data/paper/ledger.parquet"
    )
    if os.path.exists(pos_path):
        pos = pd.read_csv(pos_path)
        st.dataframe(pos)
        syms = [
            str(s).upper()
            for s in pos.get("symbol", pd.Series([])).dropna().unique().tolist()
        ]
        if st.button("Compute unrealized PnL (Polygon)"):
            try:
                from engine.infra.http import HttpClient, HttpConfig
                import numpy as np

                api_key = os.environ.get("POLYGON_API_KEY", "")
                if not api_key:
                    st.warning("POLYGON_API_KEY not set.")
                else:
                    client = HttpClient(
                        HttpConfig(requests_per_second=5.0, timeout=10.0)
                    )
                    prices = {}
                    for sym in syms:
                        try:
                            url = f"https://api.polygon.io/v2/last/trade/{sym}"
                            data = (
                                client.get_json(url, params={"apiKey": api_key}) or {}
                            )
                            p = (data.get("results", {}) or {}).get("p")
                            if p:
                                prices[sym] = float(p)
                        except Exception:
                            continue
                    pos2 = pos.copy()
                    pos2["last_price"] = pos2["symbol"].map(
                        lambda s: prices.get(str(s).upper(), float("nan"))
                    )
                    pos2["unrealized"] = (
                        pos2["last_price"].astype(float)
                        - pos2["entry_price"].astype(float)
                    ) * pos2.get("shares", pd.Series(1, index=pos2.index)).astype(float)
                    st.dataframe(pos2)
                    st.metric("Unrealized PnL", f"${pos2['unrealized'].sum():,.0f}")
            except Exception as e:
                st.error(f"Failed to fetch prices: {e}")
    else:
        st.info("No positions file found.")
    st.subheader("Recent Actions")
    if os.path.exists(log_path):
        logs = pd.read_csv(log_path)
        st.dataframe(logs.tail(50))
        if not logs.empty:
            # Simple realized PnL estimate (sell actions only, using logged price and qty vs known entry price if present is not available here)
            # Display action counts
            st.caption(f"Actions: {len(logs)} (last 50 shown)")
    else:
        st.info("No trade log yet.")
    st.subheader("Equity Curve (Paper)")
    if os.path.exists(ledger_path):
        try:
            led = pd.read_parquet(ledger_path)
            if "date" in led.columns and "equity" in led.columns:
                led = led.sort_values("date")
                st.line_chart(led.set_index("date")["equity"])
                st.metric("Final Equity", f"${float(led['equity'].iloc[-1]):,.0f}")
        except Exception as e:
            st.error(f"Failed to load ledger: {e}")

if mode == "Settings":
    st.subheader("Account Settings")
    st.info(
        "Set your account equity and defaults used by tools (risk %, cost, turnover cap). These are session settings for the UI."
    )
    if "acct" not in st.session_state:
        # Load persisted settings if available
        _saved = load_ui_settings().get("account", {})
        st.session_state.acct = {
            "equity": int(
                _saved.get("equity", os.environ.get("ACCOUNT_EQUITY", "100000"))
            ),
            "risk_pct": float(_saved.get("risk_pct", 0.005)),
            "stop_atr_mult": float(_saved.get("stop_atr_mult", 1.0)),
            "turnover_cap": float(_saved.get("turnover_cap", 0.3)),
            "cost_bps": int(_saved.get("cost_bps", 5)),
        }
    with st.form("acct_settings"):
        a_equity = st.number_input(
            "Account equity ($)",
            value=int(st.session_state.acct.get("equity", 100000)),
            step=5000,
        )
        a_risk_txt = st.text_input(
            "Risk per trade (%)",
            value=f"{100*float(st.session_state.acct.get('risk_pct', 0.005)):.2f}",
        )
        a_stop = st.number_input(
            "Stop distance (ATR×)",
            value=float(st.session_state.acct.get("stop_atr_mult", 1.0)),
            min_value=0.1,
            max_value=5.0,
            step=0.1,
        )
        a_turn = st.number_input(
            "Default turnover cap",
            value=float(st.session_state.acct.get("turnover_cap", 0.3)),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
        )
        a_cost = st.number_input(
            "Default cost (bps)",
            value=int(st.session_state.acct.get("cost_bps", 5)),
            step=1,
        )
        colsa, colsb = st.columns(2)
        with colsa:
            saved = st.form_submit_button("Save settings")
        with colsb:
            reset = st.form_submit_button("Reset to defaults")
        if saved:
            a_risk = _parse_percent(
                a_risk_txt, st.session_state.acct.get("risk_pct", 0.005)
            )
            st.session_state.acct = {
                "equity": int(a_equity),
                "risk_pct": float(a_risk),
                "stop_atr_mult": float(a_stop),
                "turnover_cap": float(a_turn),
                "cost_bps": int(a_cost),
            }
            cur = load_ui_settings()
            cur["account"] = st.session_state.acct
            save_ui_settings(cur)
            st.success("Account settings updated for this session.")
        if reset:
            st.session_state.acct = {
                "equity": 100000,
                "risk_pct": 0.005,
                "stop_atr_mult": 1.0,
                "turnover_cap": 0.3,
                "cost_bps": 5,
            }
            cur = load_ui_settings()
            cur["account"] = st.session_state.acct
            save_ui_settings(cur)
            st.success("Account settings reset to defaults.")

st.caption(
    f"Equity ${st.session_state.get('acct',{}).get('equity',100000):,} • Risk {100*float(st.session_state.get('acct',{}).get('risk_pct',0.005)):.2f}% • Stop {float(st.session_state.get('acct',{}).get('stop_atr_mult',1.0)):.1f}×ATR • Turnover cap {float(st.session_state.get('acct',{}).get('turnover_cap',0.3)):.2f} • Cost {int(st.session_state.get('acct',{}).get('cost_bps',5))} bps"
)

if mode == "Guided":
    st.subheader("Guided Setup – Research → Live")
    st.info(
        "Step through the full workflow from building data to starting overnight, with sensible defaults and one‑click actions."
    )
    st.markdown(
        "1) Build dataset with cost-aware labels → 2) Build features → 3) Weekly train (CV+meta) → 4) Backtest → 5) Preview alert → 6) Start overnight"
    )
    with st.expander("Step 1 — Build Dataset", expanded=False):
        prov = st.selectbox(
            "Provider",
            options=["yahoo", "alphavantage", "polygon"],
            index=0,
            key="g_prov",
        )
        uni = st.text_input(
            "Universe file", value="engine/data/universe/us_all.txt", key="g_uni"
        )
        start = st.text_input("Start", value="2015-01-01", key="g_start")
        thr = st.number_input(
            "Label threshold (bps)", value=10, min_value=0, key="g_thr"
        )
        if st.button("Run Step 1"):
            args = [
                "--provider",
                prov,
                "--universe-file",
                uni,
                "--start",
                start,
                "--label-horizon",
                "1",
                "--label-threshold-bps",
                str(int(thr)),
            ]
            with st.spinner("Building dataset…"):
                st.text(run_module("engine.data.build_dataset", args))
    with st.expander("Step 2 — Build Features", expanded=False):
        if st.button("Run Step 2"):
            args = [
                "--universe-file",
                st.session_state.get("g_uni", "engine/data/universe/us_all.txt"),
                "--start",
                st.session_state.get("g_start", "2015-01-01"),
                "--out",
                features,
            ]
            with st.spinner("Building features…"):
                st.text(run_module("engine.features.build_features", args))
    with st.expander("Step 3 — CV + Calibrators (isotonic)", expanded=True):
        if st.button("Run Step 3"):
            args = [
                "--features",
                features,
                "--label",
                "label_up_1d",
                "--cv-scheme",
                "time_kfold",
                "--kfolds",
                "5",
                "--purge-days",
                "3",
                "--embargo-days",
                "3",
                "--calibration",
                cfg.get("calibration", {}).get("kind", "isotonic"),
                "--out",
                oof_path_default,
                "--calibrators-out",
                calibs_out_default,
                "--spec-config",
                preset,
            ]
            with st.spinner("Running CV…"):
                st.text(run_module("engine.models.run_cv", args))
    with st.expander("Step 4 — Train Meta (HGB) + meta calibration", expanded=True):
        if st.button("Run Step 4"):
            args = [
                "--oof",
                oof_path_default,
                "--train-folds",
                "all-but-last:1",
                "--test-folds",
                "last:1",
                "--model",
                "hgb",
                "--hgb-learning-rate",
                "0.05",
                "--hgb-max-iter",
                "400",
                "--meta-calibration",
                "isotonic",
                "--replace-prob-with-calibrated",
                "--out",
                meta_pred,
                "--model-out",
                model_out,
            ]
            with st.spinner("Training meta…"):
                st.text(run_module("engine.models.train_meta", args))
    with st.expander("Step 5 — Backtest", expanded=True):
        topk = st.number_input("Top-K", value=20, min_value=1, step=1, key="g_topk")
        cost = st.number_input("Cost (bps)", value=5, min_value=0, step=1, key="g_cost")
        if st.button("Run Step 5"):
            args = [
                "--features",
                features,
                "--pred",
                meta_pred,
                "--prob-col",
                "meta_prob",
                "--top-k",
                str(int(topk)),
                "--cost-bps",
                str(int(cost)),
                "--rebalance",
                "weekly",
                "--rebal-weekday",
                "MON",
                "--report-html",
                "data/backtests/daily_topk_report.html",
            ]
            with st.spinner("Backtesting…"):
                st.text(run_module("engine.backtest.simple_daily", args))
    with st.expander("Step 6 — Preview Alert", expanded=True):
        if st.button("Run Step 6"):
            args = [
                "--config",
                preset,
                "--features",
                features,
                "--model-pkl",
                model_out,
                "--oof",
                oof_path_default,
                "--calibrators-pkl",
                calibs_out_default,
                "--universe-file",
                cfg.get("alert", {}).get(
                    "universe_file", "engine/data/universe/nasdaq100.example.txt"
                ),
                "--provider",
                cfg.get("alert", {}).get("news_provider", "none"),
                "--top-k",
                str(int(cfg.get("alert", {}).get("top_k", 3))),
                "--price-source",
                cfg.get("alert", {}).get("price_source", "feature"),
                "--dry-run",
            ]
            with st.spinner("Building alert preview…"):
                st.text(run_module("engine.tools.trade_alert", args))
    with st.expander("Step 7 — Start Overnight", expanded=False):
        if st.button("Run Step 7"):
            try:
                script = str((ROOT / "scripts" / "overnight.ps1").resolve())
                base = [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    f'"{script}"',
                ]
                args = [
                    "-Config",
                    f'"{preset}"',
                    "-AlertEvery",
                    "5",
                    "-EntryTimes",
                    '"09:35,15:55"',
                    "-StartManager",
                ]
                cmd = " ".join(base + args)
                subprocess.Popen(cmd, shell=True)
                st.success("Overnight started.")
            except Exception as e:
                st.error(f"Failed to start overnight: {e}")


if mode == "Optimise":
    st.subheader("Optimise — Build, Train, Test")
    st.info(
        "End‑to‑end research flow: build data, train models, and test with backtests & watchlists."
    )
    tab_build, tab_train, tab_test = st.tabs(["Build", "Train", "Test"])
    with tab_build:
        colA, colB = st.columns(2)
        with colA:
            prov = st.selectbox(
                "Data source",
                options=["yahoo", "alphavantage", "polygon"],
                index=0,
                help="Where to download daily prices from",
            )
            uni_label = st.selectbox(
                "Which symbols?",
                options=list(universe_map.keys()),
                index=list(universe_map.keys()).index(
                    _default_universe_label(universe_default)
                ),
                help="Pick a built‑in list or choose Custom",
            )
            uni = (
                st.text_input(
                    "Custom list (one symbol per line)",
                    value=universe_default,
                    help="Path to your own list of symbols",
                )
                if universe_map[uni_label] is None
                else universe_map[uni_label]
            )
            start_choice = st.selectbox(
                "Start date",
                options=["2010-01-01", "2015-01-01", "2020-01-01", "Custom…"],
                index=1,
            )
            start = (
                st.text_input("Custom start (YYYY-MM-DD)", value="2015-01-01")
                if start_choice == "Custom…"
                else start_choice
            )
            thr = st.selectbox(
                "Label threshold (bps)", options=[0, 5, 10, 15, 20], index=2
            )
            if st.button("Build Dataset", key="opt_build_ds"):
                est = 30
                if isinstance(uni, str) and os.path.exists(uni):
                    est = max(30, _count_universe_symbols(uni) * 2)
                out = run_with_progress(
                    "engine.data.build_dataset",
                    [
                        "--provider",
                        prov,
                        "--universe-file",
                        uni,
                        "--start",
                        start,
                        "--label-horizon",
                        "1",
                        "--label-threshold-bps",
                        str(int(thr)),
                    ],
                    label="Building dataset…",
                    est_seconds=int(est),
                )
                st.text(out)
        with colB:
            if st.button("Build Features", key="opt_build_feat"):
                est = 30
                if isinstance(uni, str) and os.path.exists(uni):
                    est = max(30, _count_universe_symbols(uni) * 1)
                out = run_with_progress(
                    "engine.features.build_features",
                    ["--universe-file", uni, "--start", start, "--out", features],
                    label="Building features…",
                    est_seconds=int(est),
                )
                st.text(out)
    with tab_train:
        st.caption(
            "CV with isotonic + time‑kfold (purge/embargo) and HGB meta with isotonic meta calibration"
        )
        if st.button("Run Weekly Train", type="primary", key="opt_train"):
            oof = run_with_progress(
                "engine.models.run_cv",
                [
                    "--features",
                    features,
                    "--label",
                    "label_up_1d",
                    "--cv-scheme",
                    "time_kfold",
                    "--kfolds",
                    "5",
                    "--purge-days",
                    "3",
                    "--embargo-days",
                    "3",
                    "--calibration",
                    "isotonic",
                    "--out",
                    oof_path_default,
                    "--calibrators-out",
                    calibs_out_default,
                    "--spec-config",
                    preset,
                ],
                label="Cross‑validating specialists…",
                est_seconds=120,
            )
            st.text(oof)
            meta = run_with_progress(
                "engine.models.train_meta",
                [
                    "--oof",
                    oof_path_default,
                    "--train-folds",
                    "all-but-last:1",
                    "--test-folds",
                    "last:1",
                    "--model",
                    "hgb",
                    "--hgb-learning-rate",
                    "0.05",
                    "--hgb-max-iter",
                    "400",
                    "--meta-calibration",
                    "isotonic",
                    "--replace-prob-with-calibrated",
                    "--out",
                    meta_pred,
                    "--model-out",
                    model_out,
                ],
                label="Training meta…",
                est_seconds=90,
            )
            st.text(meta)
    with tab_test:
        col1, col2 = st.columns(2)
        with col1:
            topk = st.selectbox(
                "How many symbols to hold?",
                options=[10, 15, 20, 25, 30],
                index=2,
                help="Equal‑weighted portfolio size",
            )
            cost = st.selectbox(
                "Estimated trading cost (bps)",
                options=[0, 2, 5, 10],
                index=2,
                help="Round‑trip cost in basis points",
            )
            if st.button("Run Backtest", key="opt_bt"):
                out = run_with_progress(
                    "engine.backtest.simple_daily",
                    [
                        "--features",
                        features,
                        "--pred",
                        meta_pred,
                        "--prob-col",
                        "meta_prob",
                        "--top-k",
                        str(topk),
                        "--cost-bps",
                        str(cost),
                        "--rebalance",
                        "weekly",
                        "--rebal-weekday",
                        "MON",
                        "--decision-log-csv",
                        "data/backtests/decision_log.csv",
                        "--report-html",
                        "data/backtests/daily_topk_report.html",
                        "--account-equity",
                        str(
                            int(st.session_state.get("acct", {}).get("equity", 100000))
                        ),
                    ],
                    label="Backtesting…",
                    est_seconds=60,
                )
                st.text(out)
            if st.button("Open latest backtest report"):
                paths = [
                    os.path.join("data", "backtests", "daily_topk_report.html"),
                    r"D:\\EngineData\\backtests\\daily_topk_report.html",
                ]
                opened = False
                for pth in paths:
                    if os.path.exists(pth):
                        try:
                            if os.name == "nt":
                                os.startfile(pth)  # type: ignore[attr-defined]
                            else:
                                import webbrowser

                                webbrowser.open(f"file://{os.path.abspath(pth)}")
                            opened = True
                            break
                        except Exception:
                            continue
                if not opened:
                    st.warning("No backtest report found.")
        with st.expander("Recent Results", expanded=False):
            # Show last backtest equity and simple stats
            bt_paths = []
            for pat in [
                os.path.join("data", "backtests", "*.parquet"),
                os.path.join("data", "backtests", "daily_topk_results.parquet"),
                r"D:\\EngineData\\backtests\\*.parquet",
            ]:
                bt_paths.extend(glob.glob(pat))
            bt_path = None
            if bt_paths:
                bt_paths = sorted(set(bt_paths), key=lambda p: os.path.getmtime(p))
                bt_path = bt_paths[-1]
            if bt_path and os.path.exists(bt_path):
                try:
                    dfbt = pd.read_parquet(bt_path)
                    if not dfbt.empty and "equity" in dfbt.columns:
                        tail = dfbt.tail(100)
                        st.line_chart(tail.set_index("date")["equity"])
                        # Simple stats
                        daily = dfbt.get("net_ret")
                        if daily is not None and len(daily) > 10:
                            import numpy as np

                            ann = 252
                            cagr = (
                                float(dfbt["equity"].iloc[-1])
                                ** (ann / max(1, len(dfbt)))
                                - 1.0
                            )
                            vol = float(np.std(daily)) * np.sqrt(ann)
                            sharpe = (float(np.mean(daily)) * ann) / (
                                vol if vol > 1e-12 else 1.0
                            )
                            peak = dfbt["equity"].cummax()
                            mdd = float((peak - dfbt["equity"]).max()) / float(
                                peak.max() if float(peak.max() or 0) > 0 else 1.0
                            )
                            m1, m2, m3 = st.columns(3)
                            m1.metric("CAGR", f"{100*cagr:.2f}%")
                            m2.metric("Sharpe", f"{sharpe:.2f}")
                            m3.metric("Max DD", f"{100*mdd:.2f}%")
                        st.caption(f"Backtest: {bt_path}")
                except Exception as e:
                    st.caption(f"Failed to load backtest: {e}")

if mode == "Analysis":
    st.subheader("Analysis — Watchlist & Alerts")
    st.info("Preview alerts (dry‑run) and inspect watchlists.")
    provider = st.selectbox(
        "News provider", options=["none", "polygon", "finnhub"], index=0
    )
    topk_alert = st.selectbox("Top-K", options=[1, 2, 3, 4, 5], index=2)
    price_src = st.selectbox("Price source", options=["feature", "live"], index=0)
    live_prov = st.selectbox("Live provider", options=["yahoo", "polygon"], index=1)
    acct_eq = int(os.environ.get("ACCOUNT_EQUITY", 100000))
    risk_pct = 0.005
    stop_mult = 1.0
    entry_px = "close"
    colA, colB = st.columns(2)
    with st.expander("Advanced (filters & options)", expanded=False):
        min_price = st.selectbox("Min price ($)", options=[0, 1, 5, 10, 20], index=0)
        max_price = st.selectbox(
            "Max price ($)", options=[0, 50, 100, 200, 500, 1000], index=0
        )
        heartbeat = st.checkbox("Send heartbeat if empty", value=True)
        mix_intr = st.checkbox("Mix intraday snapshot", value=False)
        mix_w = st.slider("Mix weight", 0.0, 1.0, 0.3, 0.05)
        intr_feat = st.text_input(
            "Intraday features parquet",
            value="data/datasets/features_intraday_latest.parquet",
        )
    with colA:
        if st.button("Run Alert (send)"):
            args = [
                "--config",
                preset,
                "--features",
                features,
                "--model-pkl",
                model_out,
                "--oof",
                oof_path_default,
                "--calibrators-pkl",
                calibs_out_default,
                "--universe-file",
                cfg.get("alert", {}).get(
                    "universe_file", "engine/data/universe/nasdaq100.example.txt"
                ),
                "--provider",
                provider,
                "--top-k",
                str(topk_alert),
                "--price-source",
                price_src,
                "--live-provider",
                live_prov,
                "--debug-risk",
                "--account-equity",
                str(acct_eq),
                "--risk-pct",
                str(risk_pct),
                "--stop-atr-mult",
                str(stop_mult),
                "--entry-price",
                entry_px,
            ]
            if min_price:
                args += ["--min-price", str(min_price)]
            if max_price:
                args += ["--max-price", str(max_price)]
            if heartbeat:
                args += ["--heartbeat-on-empty"]
            if mix_intr:
                args += [
                    "--mix-intraday",
                    str(float(mix_w)),
                    "--intraday-features",
                    intr_feat,
                ]
            if webhook:
                args += ["--discord-webhook", webhook]
            out = run_with_progress(
                "engine.tools.trade_alert", args, label="Sending alert…", est_seconds=10
            )
            st.text(out)
        if st.button("Preview Alert (dry-run)", type="primary"):
            args = [
                "--config",
                preset,
                "--features",
                features,
                "--model-pkl",
                model_out,
                "--oof",
                oof_path_default,
                "--calibrators-pkl",
                calibs_out_default,
                "--universe-file",
                cfg.get("alert", {}).get(
                    "universe_file", "engine/data/universe/nasdaq100.example.txt"
                ),
                "--provider",
                provider,
                "--top-k",
                str(topk_alert),
                "--price-source",
                price_src,
                "--live-provider",
                live_prov,
                "--debug-risk",
                "--account-equity",
                str(acct_eq),
                "--risk-pct",
                str(risk_pct),
                "--stop-atr-mult",
                str(stop_mult),
                "--entry-price",
                entry_px,
                "--dry-run",
            ]
            if min_price:
                args += ["--min-price", str(min_price)]
            if max_price:
                args += ["--max-price", str(max_price)]
            if heartbeat:
                args += ["--heartbeat-on-empty"]
            if mix_intr:
                args += [
                    "--mix-intraday",
                    str(float(mix_w)),
                    "--intraday-features",
                    intr_feat,
                ]
            out = run_with_progress(
                "engine.tools.trade_alert",
                args,
                label="Generating preview…",
                est_seconds=10,
            )
            with st.expander("Preview output"):
                st.text(out)
    with colB:
        st.subheader("Watchlist Summary")
        wl_path = st.text_input(
            "Watchlist file", value="engine/data/universe/watchlist.txt"
        )
        try:
            if os.path.exists(wl_path):
                syms = [
                    ln.strip()
                    for ln in open(wl_path, "r", encoding="utf-8").read().splitlines()
                    if ln.strip()
                ]
                st.metric("Symbols", len(syms))
                st.text(", ".join(syms[:50]) + (" …" if len(syms) > 50 else ""))
            else:
                st.caption("No watchlist file found yet.")
        except Exception as e:
            st.error(f"Failed to load watchlist: {e}")

if mode == "Trade":
    st.subheader("Trade — Real‑Time & Paper")
    st.info(
        "Start the live alert loop, or run paper trading. You can also schedule both here."
    )
    tab_rt, tab_paper, tab_sched = st.tabs(["Real‑Time", "Paper", "Scheduler"])
    with tab_rt:
        sched = st.selectbox(
            "Schedule",
            options=[
                "09:35 & 15:55",
                "Every 5 minutes",
                "Every 15 minutes",
                "At open (+5m)",
                "Before close (-5m)",
                "Custom times",
            ],
            index=0,
        )
        times = (
            st.text_input("Custom times", value="09:35,15:55")
            if sched == "Custom times"
            else ("09:35,15:55" if sched == "09:35 & 15:55" else "")
        )
        every = (
            0
            if sched in ("09:35 & 15:55", "Custom times")
            else (5 if sched == "Every 5 minutes" else 15)
        )
        at_open = sched == "At open (+5m)"
        before_close = 5 if sched == "Before close (-5m)" else 0
        rt_poll = st.selectbox("Poll seconds", options=[5, 10, 15, 30], index=2)
        rt_state = st.text_input(
            "Alert log path", value="data/alerts/alert_log.parquet"
        )
        rt_provider = st.selectbox(
            "News provider", options=["none", "polygon", "finnhub"], index=0
        )
        rt_price_src = st.selectbox(
            "Price source", options=["feature", "live"], index=0
        )
        rt_live_provider = st.selectbox(
            "Live provider", options=["yahoo", "polygon"], index=1
        )
        rt_topk = st.selectbox("Top-K per trigger", options=[1, 2, 3, 4, 5], index=2)
        rt_cooldown = st.selectbox("Cooldown (mins)", options=[0, 30, 60, 120], index=3)
        rt_explore = st.selectbox(
            "Explore prob", options=[0.0, 0.01, 0.02, 0.05, 0.1], index=0
        )
        rt_parity = st.checkbox("Parity mode (deterministic)", value=False)
        if st.button("Start real-time alerts"):
            args = [
                "--features",
                features,
                "--model-pkl",
                model_out,
                "--oof",
                oof_path_default,
                "--calibrators-pkl",
                calibs_out_default,
                "--universe-file",
                cfg.get("alert", {}).get(
                    "universe_file", "engine/data/universe/nasdaq100.example.txt"
                ),
                "--provider",
                ("none" if rt_parity else rt_provider),
                "--top-k",
                str(int(rt_topk)),
                *(["--times", times] if times else []),
                *(["--every-minutes", str(int(every))] if every else []),
                *(["--at-open"] if at_open else []),
                *(
                    ["--before-close-mins", str(int(before_close))]
                    if before_close
                    else []
                ),
                "--poll",
                str(rt_poll),
                "--state",
                rt_state,
                "--price-source",
                ("feature" if rt_parity else rt_price_src),
                "--live-provider",
                rt_live_provider,
                "--cooldown-mins",
                str(0 if rt_parity else int(rt_cooldown)),
                "--explore-prob",
                str(0.0 if rt_parity else float(rt_explore)),
            ]
            if (not rt_parity) and webhook:
                args += ["--discord-webhook", webhook]
            try:
                os.makedirs(os.path.dirname(rt_state), exist_ok=True)
                proc = subprocess.Popen(
                    [sys.executable, "-m", "engine.tools.real_time_alert", *args]
                )
                pid_path = os.path.join(os.path.dirname(rt_state), "rt.pid")
                with open(pid_path, "w", encoding="utf-8") as f:
                    f.write(str(proc.pid))
                st.success(f"Real-time alerts started (PID {proc.pid}).")
            except Exception as e:
                st.error(f"Failed to start real-time alerts: {e}")
        if st.button("Stop real-time alerts"):
            try:
                pid_path = os.path.join(os.path.dirname(rt_state), "rt.pid")
                if os.path.exists(pid_path):
                    pid = int(open(pid_path, "r", encoding="utf-8").read().strip())
                    import signal

                    os.kill(pid, signal.SIGTERM)
                    st.success(f"Sent stop signal to PID {pid}.")
                else:
                    st.info("No PID file found; process may not be running.")
            except Exception as e:
                st.error(f"Stop failed: {e}")
        st.subheader("Single Trigger (Parity Test)")
        if st.button("Trigger once now --force"):
            diag_path = "data/alerts/ui_parity_rt.csv"
            args = [
                "--config",
                preset,
                "--features",
                features,
                "--model-pkl",
                model_out,
                "--oof",
                oof_path_default,
                "--calibrators-pkl",
                calibs_out_default,
                "--universe-file",
                cfg.get("alert", {}).get(
                    "universe_file", "engine/data/universe/nasdaq100.example.txt"
                ),
                "--provider",
                "none",
                "--top-k",
                str(int(rt_topk)),
                "--alert-log-csv",
                diag_path,
                "--cooldown-mins",
                "0",
                "--price-source",
                "feature",
                "--force",
            ]
            out = run_with_progress(
                "engine.tools.real_time_alert", args, label="Triggering…", est_seconds=5
            )
            st.text(out)
    with tab_paper:
        state_dir = st.text_input("State dir", value="data/paper")
        topk = st.selectbox("Top‑K", options=[10, 15, 20, 25, 30], index=2)
        cost_bps = st.selectbox("Cost (bps)", options=[0, 2, 5, 10], index=2)
        initial_equity = int(os.environ.get("ACCOUNT_EQUITY", 1_000_000))
        if st.button("Step Paper Trader"):
            args = [
                "--features",
                features,
                "--universe-file",
                cfg.get("alert", {}).get(
                    "universe_file", "engine/data/universe/nasdaq100.example.txt"
                ),
                "--top-k",
                str(topk),
                "--state-dir",
                state_dir,
                "--cost-bps",
                str(cost_bps),
                "--initial-equity",
                str(initial_equity),
                "--config",
                preset,
            ]
            pred_path = cfg.get("paths", {}).get(
                "meta", "data/datasets/meta_predictions.parquet"
            )
            if os.path.exists(pred_path):
                args += ["--pred", pred_path]
            else:
                args += [
                    "--model-pkl",
                    model_out,
                    "--oof",
                    oof_path_default,
                    "--calibrators-pkl",
                    calibs_out_default,
                ]
            r = cfg.get("risk", {})
            args += [
                "--min-adv-usd",
                str(r.get("min_adv_usd", 1e7)),
                "--max-atr-pct",
                str(r.get("max_atr_pct", 0.05)),
            ]
            if webhook:
                args += ["--discord-webhook", webhook]
            out = run_with_progress(
                "engine.tools.paper_trader",
                args,
                label="Stepping paper trader…",
                est_seconds=10,
            )
            st.text(out)
        if st.button("Run to end"):
            args = [
                "--features",
                features,
                "--universe-file",
                cfg.get("alert", {}).get(
                    "universe_file", "engine/data/universe/nasdaq100.example.txt"
                ),
                "--top-k",
                str(topk),
                "--state-dir",
                state_dir,
                "--cost-bps",
                str(cost_bps),
                "--initial-equity",
                str(initial_equity),
                "--config",
                preset,
                "--pred",
                cfg.get("paths", {}).get(
                    "meta", "data/datasets/meta_predictions.parquet"
                ),
                "--run-to-end",
            ]
            r = cfg.get("risk", {})
            args += [
                "--min-adv-usd",
                str(r.get("min_adv_usd", 1e7)),
                "--max-atr-pct",
                str(r.get("max_atr_pct", 0.05)),
            ]
            out = run_with_progress(
                "engine.tools.paper_trader",
                args,
                label="Running to end…",
                est_seconds=60,
            )
            st.text(out)
        if os.path.exists(os.path.join(state_dir, "ledger.parquet")):
            led = pd.read_parquet(os.path.join(state_dir, "ledger.parquet"))
            st.line_chart(led.set_index("date")["equity"])
            st.dataframe(led.tail(10))
    with tab_sched:
        # Reuse scheduler controls – create overnight/weekly train; status
        st.caption("Quick scheduler controls (full options in Scheduler tab)")
        if st.button("Create Overnight Task (defaults)"):
            script = str((ROOT / "scripts" / "schedule-overnight.ps1").resolve())
            out = run_ps(
                script,
                [
                    "-TaskName",
                    "Engine-Overnight",
                    "-Start",
                    "09:25",
                    "-Config",
                    preset,
                    "-AlertEvery",
                    "5",
                    "-EntryTimes",
                    "09:35,15:55",
                    "-StartManager",
                ],
            )
            st.text(out)
        if st.button("Create Weekly Train Task (SUN 08:00)"):
            script = str((ROOT / "scripts" / "schedule-weekly-train.ps1").resolve())
            out = run_ps(
                script,
                [
                    "-TaskName",
                    "Engine-WeeklyTrain",
                    "-Days",
                    "SUN",
                    "-Start",
                    "08:00",
                    "-Features",
                    features,
                    "-Preset",
                    preset,
                ],
            )
            st.text(out)
if mode == "Scheduler":
    st.subheader("Schedule Overnight (Weekdays)")
    st.info(
        "Create or remove Windows Task Scheduler jobs for overnight alerts and weekly training. Use the status table to verify next runs."
    )
    sc1, sc2 = st.columns(2)
    with sc1:
        tname = st.text_input("Task name", value="Engine-Overnight")
        start = st.text_input("Start time (HH:MM)", value="09:25")
        cfg_path = st.text_input("Config (YAML)", value=preset)
        alert_every = st.number_input(
            "Alert every (mins)", value=5, min_value=1, step=1
        )
        entry_times = st.text_input("Entry times", value="09:35,15:55")
        start_manager = st.checkbox("Start RT position manager", value=True)
        parity_mode = st.checkbox("Parity mode (deterministic)", value=False)
        if st.button("Create Overnight Task"):
            script = str((ROOT / "scripts" / "schedule-overnight.ps1").resolve())
            args = [
                "-TaskName",
                tname,
                "-Start",
                start,
                "-Config",
                cfg_path,
                "-AlertEvery",
                str(int(alert_every)),
                "-EntryTimes",
                entry_times,
            ]
            if start_manager:
                args += ["-StartManager"]
            if parity_mode:
                args += ["-Parity"]
            with st.spinner("Creating task…"):
                out = run_ps(script, args)
            st.text(out)
    with sc2:
        st.subheader("Schedule Weekly Train")
        tname2 = st.text_input("Task name (weekly)", value="Engine-WeeklyTrain")
        days = st.text_input("Days (e.g., SUN)", value="SUN")
        start2 = st.text_input("Start time (HH:MM)", value="08:00")
        feats = st.text_input("Features parquet", value=features)
        pset = st.text_input("Preset (YAML)", value=preset)
        if st.button("Create Weekly Train Task"):
            script = str((ROOT / "scripts" / "schedule-weekly-train.ps1").resolve())
            args = [
                "-TaskName",
                tname2,
                "-Days",
                days,
                "-Start",
                start2,
                "-Features",
                feats,
                "-Preset",
                pset,
            ]
            with st.spinner("Creating weekly task…"):
                out = run_ps(script, args)
            st.text(out)
    st.divider()
    st.subheader("Remove Scheduled Tasks")
    del1, del2 = st.columns(2)
    with del1:
        del_name = st.text_input("Task to remove (overnight)", value="Engine-Overnight")
        if st.button("Remove Overnight Task"):
            cmdline = f'schtasks /Delete /TN `"{del_name}`" /F'
            with st.spinner("Removing task…"):
                st.text(run_ps_cmd(cmdline))
    with del2:
        del_name2 = st.text_input("Task to remove (weekly)", value="Engine-WeeklyTrain")
        if st.button("Remove Weekly Train Task"):
            cmdline = f'schtasks /Delete /TN `"{del_name2}`" /F'
            with st.spinner("Removing task…"):
                st.text(run_ps_cmd(cmdline))
    st.divider()
    st.subheader("Task Status (Engine-*)")
    tfilter = st.text_input(
        "Task filter (wildcard)",
        value="Engine-*",
        help="Use * as a wildcard (e.g., Engine-*)",
    )
    auto = st.checkbox(
        "Auto-refresh", value=False, help="Refresh this table automatically"
    )
    interval = st.number_input(
        "Refresh interval (sec)", value=15, min_value=5, max_value=300, step=5
    )
    if auto:
        # Simple page auto-refresh using meta tag
        st.markdown(
            f"<meta http-equiv='refresh' content='{int(interval)}'>",
            unsafe_allow_html=True,
        )
    if st.button("Refresh Task List") or auto:
        # Build a PowerShell command that outputs CSV we can parse easily
        pat = tfilter.replace("'", "''")
        ps = (
            f"Get-ScheduledTask | Where-Object {{$_.TaskName -like '{pat}'}} "
            "| ForEach-Object { $i = Get-ScheduledTaskInfo $_; "
            "[PSCustomObject]@{ TaskName=$_.TaskName; State=$_.State; NextRunTime=$i.NextRunTime; LastRunTime=$i.LastRunTime; LastTaskResult=$i.LastTaskResult } } "
            "| ConvertTo-Csv -NoTypeInformation"
        )
        out = run_ps_cmd(ps)
        try:
            df = pd.read_csv(io.StringIO(out))
            st.dataframe(df)
        except Exception:
            st.text(out or "No tasks found or insufficient permissions.")
    col_stop, col_force = st.columns(2)
    with col_stop:
        if st.button("Stop Matching Tasks"):
            pat = tfilter.replace("'", "''")
            cmdline = f"Get-ScheduledTask | Where-Object {{$_.TaskName -like '{pat}'}} | Stop-ScheduledTask"
            with st.spinner("Stopping tasks…"):
                st.text(run_ps_cmd(cmdline))
    with col_force:
        if st.button("Stop Overnight Processes (PID files)"):
            # Attempt graceful stop via bundled scripts, then force kill by PID files
            try:
                rt = str((ROOT / "scripts" / "stop-rt-alert.ps1").resolve())
                en = str((ROOT / "scripts" / "stop-entry.ps1").resolve())
                run_ps(rt, [])
                run_ps(en, [])
            except Exception:
                pass
            # Force kill by PIDs if present (Windows)
            msgs = []
            for pid_path in ["data/alerts/rt.pid", "data/entries/entry.pid"]:
                try:
                    if os.path.exists(pid_path):
                        pid = open(pid_path, "r", encoding="utf-8").read().strip()
                        if pid:
                            out = run_ps_cmd(f"taskkill /PID {pid} /T /F")
                            msgs.append(f"Killed PID {pid}:\n{out}")
                except Exception as e:
                    msgs.append(f"Failed killing using {pid_path}: {e}")
            st.text("\n\n".join(msgs) or "Stop signals sent.")

if mode == "Logs":
    st.subheader("Logs & Diagnostics")
    st.info(
        "Quickly inspect recent logs and diagnostics. Tail text logs or preview CSV/Parquet diagnostics. Use auto‑refresh while debugging."
    )
    # Quick picks for common files
    common = [
        "data/logs/rt-alert.out",
        "data/logs/rt-alert.err",
        "data/logs/entry-sched.out",
        "data/logs/entry-sched.err",
        "data/alerts/alert_diag.csv",
        "data/alerts/alert_log.parquet",
        "data/paper/entry_log.csv",
        "data/paper/ledger.parquet",
    ]
    common_exist = [p for p in common if os.path.exists(p)]
    quick = st.selectbox(
        "Quick pick",
        options=(common_exist + ["Custom…"]) if common_exist else ["Custom…"],
        index=0,
    )
    # Collect recent files
    log_dirs = ["data/logs", "data/alerts", "data/entries", "data/paper"]
    patterns = ["*.log", "*.out", "*.err", "*.txt", "*.csv", "*.parquet"]
    files: list[tuple[str, float]] = []
    for d in log_dirs:
        if os.path.isdir(d):
            for pat in patterns:
                for p in glob.glob(os.path.join(d, pat)):
                    try:
                        files.append((p, os.path.getmtime(p)))
                    except Exception:
                        continue
    files = sorted(set(files), key=lambda x: -x[1])
    file_list = [p for p, _ in files]
    if not file_list:
        st.info(
            "No log files found under data/logs, data/alerts, data/entries, or data/paper."
        )
    else:
        default_sel = file_list[0]
        chosen = quick if quick != "Custom…" and quick in file_list else default_sel
        sel = st.selectbox(
            "Select a log file",
            options=file_list,
            index=file_list.index(chosen),
            help="Most recent files appear first",
        )
        tail_lines = st.selectbox(
            "Tail last lines", options=[50, 100, 200, 500, 1000], index=2
        )
        auto = st.checkbox("Auto-refresh", value=False)
        interval = st.selectbox("Interval (sec)", options=[5, 10, 15, 30, 60], index=2)
        if auto:
            st.markdown(
                f"<meta http-equiv='refresh' content='{int(interval)}'>",
                unsafe_allow_html=True,
            )
        ext = os.path.splitext(sel)[1].lower()
        try:
            if ext == ".csv":
                df = pd.read_csv(sel)
                st.dataframe(df.tail(int(tail_lines)))
            elif ext == ".parquet":
                df = pd.read_parquet(sel)
                st.dataframe(df.tail(int(tail_lines)))
            else:
                try:
                    with open(sel, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                    text = "".join(lines[-int(tail_lines) :])
                except Exception:
                    text = "(failed to read file)"
                st.text(text)
        except Exception as e:
            st.error(f"Failed to load {sel}: {e}")
