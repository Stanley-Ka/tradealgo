from __future__ import annotations

"""Replay a trading day offline using the live entry + monitor stack.

Pipeline:
- Build (or reuse) a NASDAQ universe.
- Fetch recent daily bars (Yahoo) for the symbols and compute baseline features.
- Optionally fetch news and derive simple sentiment.
- Run `entry_loop` to open positions on the requested date using the live meta model.
- Step day-by-day with `position_monitor` to simulate exits.
- Produce a performance summary and persist artifacts under `data/backtests/replays`.

Intended for research/testing when markets are closed; helps build labelled trade data.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from ..infra.env import load_env_files
from ..infra.styles import resolve_style
from ..infra.yaml_config import load_yaml_config
from ..infra.log import get_logger
from ..features.baseline import compute_baseline_features
from ..data.daily_cache import get_daily_adjusted
from ..tools.build_universe_polygon import fetch_polygon_tickers
import random
from ..tools.entry_loop import main as entry_loop_main
from ..tools.position_monitor import main as position_monitor_main
from ..news.providers import fetch_news_multi
from ..news.sentiment import score_news
from ..data.polygon_client import PolygonClient


LOG = get_logger(__name__)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay a trading day offline with live entry/monitor logic"
    )
    p.add_argument("--date", required=True, help="Target trade date YYYY-MM-DD")
    p.add_argument(
        "--config",
        type=str,
        default="engine/presets/swing_aggressive.yaml",
        help="Preset YAML to load paths/risk",
    )
    p.add_argument(
        "--style",
        type=str,
        default="",
        help="Style alias (overrides --config if provided)",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=60,
        help="Trading days of history to fetch before target date",
    )
    p.add_argument(
        "--hold-days",
        type=int,
        default=-1,
        help="Maximum days to advance monitoring after entry (-1 = let engine decide / until data runs out)",
    )
    p.add_argument(
        "--max-symbols",
        type=int,
        default=120,
        help="Limit universe size for faster fetches",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Override top-K entries (fallback to config when 0)",
    )
    p.add_argument(
        "--provider",
        choices=["yahoo"],
        default="yahoo",
        help="Price provider for historical bars",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional output root (default data/backtests/replays/%%date%%/%%ts%%)",
    )
    p.add_argument(
        "--news-provider",
        choices=["polygon", "finnhub", "none"],
        default="polygon",
        help="News sentiment provider",
    )
    p.add_argument(
        "--news-window",
        type=int,
        default=3,
        help="Days of news history leading into target date",
    )
    p.add_argument(
        "--news-max-symbols",
        type=int,
        default=40,
        help="Limit news fetch symbols to reduce rate usage",
    )
    p.add_argument(
        "--skip-news",
        action="store_true",
        help="Skip news fetch/sentiment even if provider keys exist",
    )
    p.add_argument(
        "--entry-threshold",
        type=float,
        default=None,
        help="Override entry threshold (meta prob)",
    )
    p.add_argument(
        "--exit-threshold",
        type=float,
        default=0.45,
        help="Meta prob exit threshold for monitor",
    )
    p.add_argument(
        "--exit-consecutive",
        type=int,
        default=2,
        help="Consecutive days below threshold before exit",
    )
    # Optional overrides for ATR-based risk controls (default from config risk.*)
    p.add_argument(
        "--stop-atr-mult",
        type=float,
        default=None,
        help="Override stop ATR multiple for entries",
    )
    p.add_argument(
        "--take-profit-atr-mult",
        type=float,
        default=None,
        help="Override take-profit ATR multiple for monitor",
    )
    p.add_argument(
        "--trail-atr-mult",
        type=float,
        default=None,
        help="Override trailing stop ATR multiple for monitor (0 to disable)",
    )
    p.add_argument("--verbose", action="store_true", help="Increase logging detail")
    p.add_argument(
        "--download-workers",
        type=int,
        default=6,
        help="Concurrent Yahoo downloads (1..16)",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def _resolve_config(style: str, config: str) -> Path:
    if style:
        resolved = resolve_style(style)
        if not resolved:
            raise FileNotFoundError(f"Style '{style}' not registered in infra.styles")
        return Path(resolved).resolve()
    path = Path(config).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Preset config not found: {config}")
    return path


def _default_output_dir(base: Optional[str], target_date: pd.Timestamp) -> Path:
    root = Path(base).resolve() if base else Path("data/backtests/replays").resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = root / f"{target_date:%Y%m%d}" / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_universe_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip().upper() for ln in f if ln.strip() and not ln.startswith("#")]


def _maybe_fetch_universe(max_symbols: int, out_path: Path) -> list[str]:
    symbols: list[str] = []
    try:
        load_env_files()
        symbols = fetch_polygon_tickers(["CS"], ["NASDAQ"])
        if not symbols:
            raise RuntimeError("Polygon returned 0 symbols")
        LOG.info("[replay] fetched %d NASDAQ symbols from Polygon", len(symbols))
    except Exception as exc:
        LOG.warning(
            "[replay] Polygon universe fetch failed (%s); using local fallback", exc
        )
        fallback = Path("engine/data/universe/nasdaq100.example.txt")
        if not fallback.exists():
            raise FileNotFoundError(
                "Fallback universe file missing: engine/data/universe/nasdaq100.example.txt"
            )
        symbols = _read_universe_file(str(fallback))
        LOG.info("[replay] loaded %d symbols from fallback universe", len(symbols))
    if max_symbols > 0 and len(symbols) > max_symbols:
        rnd = random.Random(hash(out_path.name) & 0xFFFFFFFF)
        rnd.shuffle(symbols)
        symbols = symbols[:max_symbols]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sym in symbols:
            f.write(sym.upper() + "\n")
    return symbols


def _fetch_features(
    symbols: list[str], start: pd.Timestamp, end: pd.Timestamp, workers: int = 6
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    fetch_start = start - pd.Timedelta(days=45)
    fetch_end = end + pd.Timedelta(days=2)

    def _one(sym: str) -> Optional[pd.DataFrame]:
        try:
            # Prefer Polygon; fallback to Yahoo; with caching
            adj = get_daily_adjusted(
                sym, start=fetch_start, end=fetch_end, prefer="polygon"
            )
        except Exception as exc:
            LOG.debug("[replay] daily fetch failed for %s: %s", sym, exc)
            return None
        base = adj.copy()
        base["symbol"] = sym
        try:
            feat = compute_baseline_features(base)
        except Exception as exc:
            LOG.debug("[replay] baseline features failed for %s: %s", sym, exc)
            return None
        feat = feat[(feat["date"] >= start) & (feat["date"] <= end)].copy()
        if feat.empty:
            return None
        return feat

    workers = max(1, min(int(workers), 16))
    LOG.info(
        "[replay] starting Yahoo fetch with workers=%d symbols=%d",
        workers,
        len(symbols),
    )
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, sym): sym for sym in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                df = fut.result()
                if df is not None:
                    frames.append(df)
            except Exception as exc:  # pragma: no cover
                LOG.debug("[replay] fetch task failed for %s: %s", sym, exc)
            done += 1
            if done % 20 == 0 or done == len(symbols):
                LOG.info("[replay] fetched %d/%d symbols", done, len(symbols))

    if not frames:
        raise RuntimeError(
            "No features fetched; check network connectivity or provider limits"
        )
    panel = pd.concat(frames, axis=0, ignore_index=True)
    panel.sort_values(["date", "symbol"], inplace=True)
    panel.reset_index(drop=True, inplace=True)
    return panel


def _fetch_intraday_bars(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    out_dir: Path,
    interval: str = "minute",
    batch_days: int = 90,
) -> Optional[Path]:
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        LOG.warning("[replay] POLYGON_API_KEY missing; skipping intraday fidelity")
        return None
    if not symbols:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    client = PolygonClient()
    total_days = (end - start).days + 1
    for sym in symbols:
        LOG.info("[replay] intraday fetch start: %s (%s days)", sym, total_days)
        rows: list[dict] = []
        cur = start
        while cur <= end:
            LOG.debug("[replay] intraday %s %s", sym, cur.date())
            try:
                data = client.aggregates(
                    sym, 1, interval, cur.strftime("%Y-%m-%d"), cur.strftime("%Y-%m-%d")
                )
            except Exception as exc:
                LOG.warning(
                    "[replay] intraday fetch failed for %s on %s: %s",
                    sym,
                    cur.date(),
                    exc,
                )
                break
            results = data.get("results", [])
            for r in results:
                ts = pd.to_datetime(r.get("t", 0), unit="ms", utc=True).tz_convert(None)
                rows.append(
                    {
                        "ts": ts,
                        "open": float(r.get("o", 0.0)),
                        "high": float(r.get("h", 0.0)),
                        "low": float(r.get("l", 0.0)),
                        "close": float(r.get("c", 0.0)),
                        "volume": float(r.get("v", 0.0)),
                    }
                )
            cur += timedelta(days=1)
            time.sleep(0.25)
        if not rows:
            LOG.info("[replay] no intraday bars fetched for %s", sym)
            continue
        df = (
            pd.DataFrame(rows)
            .drop_duplicates("ts")
            .sort_values("ts")
            .reset_index(drop=True)
        )
        df.to_parquet(out_dir / f"{sym}.parquet", index=False)
        LOG.info("[replay] intraday bars stored: %s rows=%d", sym, len(df))
    return out_dir


def _build_news(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    provider: str,
    max_symbols: int,
    out_path: Path,
) -> Optional[Path]:
    if provider == "none":
        return None
    try:
        sample = symbols[:max_symbols] if max_symbols > 0 else symbols
        if not sample:
            return None
        items = fetch_news_multi(
            sample,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            provider=provider,
        )
    except Exception as exc:
        LOG.warning(
            "[replay] news fetch failed (%s); continuing without sentiment", exc
        )
        return None
    if not items:
        LOG.info("[replay] no news items returned for window")
        return None
    rows = []
    for it in items:
        try:
            avg, _, _ = score_news([it])
        except Exception:
            avg = 0.0
        rows.append(
            {
                "date": pd.to_datetime(it.date).normalize(),
                "symbol": str(it.symbol).upper(),
                "sentiment": float(np.clip(avg, -1.0, 1.0)),
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows).dropna()
    if df.empty:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    LOG.info("[replay] wrote news sentiment for %d rows -> %s", len(df), out_path)
    return out_path


def _ensure_meta_paths(
    cfg: dict, config_path: Path
) -> tuple[str, Optional[str], Optional[str]]:
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    meta_model = paths_cfg.get("meta_model")
    if not meta_model:
        raise RuntimeError(f"meta_model path missing in {config_path}")
    meta_model = str(Path(meta_model).expanduser())
    if not Path(meta_model).exists():
        raise FileNotFoundError(f"Meta model not found: {meta_model}")
    calib_cfg = (
        cfg.get("calibration", {}) if isinstance(cfg.get("calibration"), dict) else {}
    )
    cals = calib_cfg.get("calibrators_pkl")
    if cals:
        cals = str(Path(cals).expanduser())
        if not Path(cals).exists():
            LOG.warning(
                "[replay] calibrators file missing: %s (falling back to naive)", cals
            )
            cals = None
    meta_cal = calib_cfg.get("meta_calibrator_pkl")
    if meta_cal:
        meta_cal = str(Path(meta_cal).expanduser())
        if not Path(meta_cal).exists():
            LOG.warning("[replay] meta calibrator missing: %s (ignoring)", meta_cal)
            meta_cal = None
    return meta_model, cals, meta_cal


def _top_k_from_cfg(cfg: dict, override: int) -> int:
    if override and override > 0:
        return int(override)
    for path in (["entry", "top_k"], ["alert", "top_k"]):
        cur = cfg
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        if isinstance(cur, (int, float)) and int(cur) > 0:
            return int(cur)
    return 5


def _entry_threshold_from_cfg(cfg: dict, override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    entry_cfg = cfg.get("entry", {}) if isinstance(cfg.get("entry"), dict) else {}
    if isinstance(entry_cfg.get("entry_threshold"), (int, float)):
        return float(entry_cfg["entry_threshold"])
    alert_cfg = cfg.get("alert", {}) if isinstance(cfg.get("alert"), dict) else {}
    if isinstance(alert_cfg.get("signal_threshold"), (int, float)):
        return float(alert_cfg["signal_threshold"])
    return 0.0


def _confirmations_from_cfg(cfg: dict) -> int:
    entry_cfg = cfg.get("entry", {}) if isinstance(cfg.get("entry"), dict) else {}
    if isinstance(entry_cfg.get("confirmations"), (int, float)):
        return int(entry_cfg["confirmations"])
    return 1


def _risk_params(cfg: dict) -> dict:
    risk_cfg = cfg.get("risk", {}) if isinstance(cfg.get("risk"), dict) else {}
    return {
        "account_equity": float(risk_cfg.get("account_equity", 100000.0)),
        "risk_mode": str(risk_cfg.get("risk_mode", "auto")),
        "risk_pct": float(risk_cfg.get("risk_pct", 0.005)),
        "risk_min_pct": float(risk_cfg.get("min_risk_pct", 0.002)),
        "risk_max_pct": float(risk_cfg.get("max_risk_pct", 0.006)),
        "risk_curve": str(risk_cfg.get("risk_curve", "quadratic")),
        "risk_base_prob": float(risk_cfg.get("base_prob", 0.5)),
        "stop_atr_mult": float(risk_cfg.get("stop_atr_mult", 1.0)),
        "min_adv_usd": float(risk_cfg.get("min_adv_usd", 5_000_000))
        if risk_cfg.get("min_adv_usd") is not None
        else None,
        "max_atr_pct": float(risk_cfg.get("max_atr_pct", 0.12))
        if risk_cfg.get("max_atr_pct") is not None
        else 0.12,
        "max_name_weight": float(risk_cfg.get("max_name_weight", 0.10))
        if risk_cfg.get("max_name_weight") is not None
        else None,
        "max_position_notional": float(risk_cfg.get("max_position_notional"))
        if risk_cfg.get("max_position_notional") is not None
        else None,
        "take_profit_atr_mult": float(
            risk_cfg.get(
                "take_profit_atr_mult",
                risk_cfg.get("tp1_atr_mult", 1.5),
            )
        ),
        "trail_atr_mult": float(
            risk_cfg.get("trail_atr_mult", risk_cfg.get("trail_stop_mult", 0.0))
        ),
    }


def _price_filters(cfg: dict) -> tuple[Optional[float], Optional[float]]:
    alert_cfg = cfg.get("alert", {}) if isinstance(cfg.get("alert"), dict) else {}
    min_price = alert_cfg.get("min_price")
    max_price = alert_cfg.get("max_price")
    return (
        float(min_price) if isinstance(min_price, (int, float)) else None,
        float(max_price) if isinstance(max_price, (int, float)) else None,
    )


def _sector_params(cfg: dict) -> tuple[Optional[str], Optional[int]]:
    entry_cfg = cfg.get("entry", {}) if isinstance(cfg.get("entry"), dict) else {}
    map_path = entry_cfg.get("sector_map_csv")
    cap = entry_cfg.get("sector_cap")
    if map_path:
        map_path = str(Path(map_path).expanduser())
        if not Path(map_path).exists():
            LOG.warning("[replay] sector map missing: %s", map_path)
            map_path = None
    return (
        map_path,
        int(cap) if isinstance(cap, (int, float)) and int(cap) > 0 else None,
    )


def _run_entry(
    features_path: Path,
    universe_path: Path,
    cfg_path: Path,
    cfg: dict,
    risk: dict,
    meta_model: str,
    calibrators: Optional[str],
    meta_calibrator: Optional[str],
    news_path: Optional[Path],
    args: argparse.Namespace,
    run_dir: Path,
    target_date: pd.Timestamp,
) -> tuple[pd.DataFrame, Path, Path]:
    top_k = _top_k_from_cfg(cfg, args.top_k)
    entry_threshold = _entry_threshold_from_cfg(cfg, args.entry_threshold)
    confirmations = _confirmations_from_cfg(cfg)
    min_price, max_price = _price_filters(cfg)
    sector_map, sector_cap = _sector_params(cfg)

    positions_csv = run_dir / "positions.csv"
    decision_log = run_dir / "entry_log.csv"
    state_csv = run_dir / "entry_state.csv"

    entry_args: list[str] = [
        "--config",
        str(cfg_path),
        "--features",
        str(features_path),
        "--model-pkl",
        meta_model,
        "--universe-file",
        str(universe_path),
        "--positions-csv",
        str(positions_csv),
        "--decision-log-csv",
        str(decision_log),
        "--state-csv",
        str(state_csv),
        "--top-k",
        str(top_k),
        "--account-equity",
        f"{risk['account_equity']:.2f}",
        "--risk-mode",
        risk["risk_mode"],
        "--risk-pct",
        f"{risk['risk_pct']:.6f}",
        "--risk-min-pct",
        f"{risk['risk_min_pct']:.6f}",
        "--risk-max-pct",
        f"{risk['risk_max_pct']:.6f}",
        "--risk-curve",
        risk["risk_curve"],
        "--stop-atr-mult",
        f"{risk['stop_atr_mult']:.4f}",
        "--max-atr-pct",
        f"{risk['max_atr_pct']:.4f}",
        "--date",
        target_date.strftime("%Y-%m-%d"),
        "--discord-webhook",
        "",
    ]
    if calibrators:
        entry_args += ["--calibrators-pkl", calibrators]
    if meta_calibrator:
        entry_args += ["--meta-calibrator-pkl", meta_calibrator]
    if news_path is not None:
        entry_args += ["--news-sentiment", str(news_path)]
    if entry_threshold > 0:
        entry_args += ["--entry-threshold", f"{entry_threshold:.4f}"]
    if confirmations > 1:
        entry_args += ["--confirmations", str(confirmations)]
    if risk.get("risk_base_prob") is not None:
        entry_args += ["--risk-base-prob", f"{risk['risk_base_prob']:.4f}"]
    if risk.get("min_adv_usd") is not None:
        entry_args += ["--min-adv-usd", f"{risk['min_adv_usd']:.0f}"]
    if risk.get("max_name_weight") is not None:
        entry_args += ["--max-name-weight", f"{risk['max_name_weight']:.4f}"]
    if risk.get("max_position_notional") is not None:
        entry_args += [
            "--max-position-notional",
            f"{risk['max_position_notional']:.2f}",
        ]
    if min_price is not None:
        entry_args += ["--min-price", f"{min_price:.2f}"]
    if max_price is not None:
        entry_args += ["--max-price", f"{max_price:.2f}"]
    if sector_map and sector_cap:
        entry_args += ["--sector-map-csv", sector_map, "--sector-cap", str(sector_cap)]

    LOG.info(
        "[replay] running entry_loop with top_k=%s threshold=%.3f",
        top_k,
        entry_threshold,
    )
    entry_loop_main(entry_args)
    if not positions_csv.exists():
        LOG.info("[replay] entry_loop produced no trades (positions.csv missing)")
        return pd.DataFrame(), positions_csv, decision_log
    entries = pd.read_csv(positions_csv)
    return entries, positions_csv, decision_log


def _run_monitor(
    features_panel: pd.DataFrame,
    cfg_path: Path,
    meta_model: str,
    calibrators: Optional[str],
    meta_calibrator: Optional[str],
    news_path: Optional[Path],
    intraday_dir: Optional[Path],
    positions_csv: Path,
    exits_csv: Path,
    monitor_state: Path,
    target_date: pd.Timestamp,
    hold_days: int,
    exit_thresh: float,
    exit_consec: int,
    take_profit_mult: float,
    trail_mult: float,
    run_dir: Path,
) -> None:
    dates = sorted(d for d in features_panel["date"].unique() if d > target_date)
    if hold_days > 0:
        cutoff = target_date + pd.Timedelta(days=hold_days)
        dates = [d for d in dates if d <= cutoff]
    if not dates:
        LOG.info("[replay] no forward dates available for monitoring")
        return
    for day in dates:
        tmp_path = run_dir / f"features_through_{day:%Y%m%d}.parquet"
        subset = features_panel[features_panel["date"] <= day].copy()
        subset.to_parquet(tmp_path, index=False)
        mon_args = [
            "--features",
            str(tmp_path),
            "--positions-csv",
            str(positions_csv),
            "--exits-csv",
            str(exits_csv),
            "--model-pkl",
            meta_model,
            "--prob-exit-thresh",
            f"{exit_thresh:.4f}",
            "--prob-exit-consecutive",
            str(exit_consec),
            "--monitor-state-csv",
            str(monitor_state),
            "--discord-webhook",
            "",
        ]
        if calibrators:
            mon_args += ["--calibrators-pkl", calibrators]
        if meta_calibrator:
            mon_args += ["--meta-calibrator-pkl", meta_calibrator]
        if news_path is not None:
            mon_args += ["--news-sentiment", str(news_path)]
        if intraday_dir is not None:
            mon_args += ["--intraday-dir", str(intraday_dir)]
        if take_profit_mult is not None:
            mon_args += ["--take-profit-atr-mult", f"{take_profit_mult:.4f}"]
        if trail_mult is not None and trail_mult > 0:
            mon_args += ["--trail-atr-mult", f"{trail_mult:.4f}"]
        mon_args = ["--config", str(cfg_path)] + mon_args
        LOG.info("[replay] monitoring positions using data up to %s", day.date())
        position_monitor_main(mon_args)
        # Break if no positions remain
        if positions_csv.exists():
            remaining = pd.read_csv(positions_csv)
            if remaining.empty:
                LOG.info("[replay] all positions closed by %s", day.date())
                break


def _summarize_trades(
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    positions_csv: Path,
    features_panel: pd.DataFrame,
    target_date: pd.Timestamp,
    hold_days: int,
) -> pd.DataFrame:
    entries = entries.copy()
    entries["entry_date"] = pd.to_datetime(entries["entry_date"]).dt.normalize()
    exits = exits.copy()
    if not exits.empty:
        exits["exit_date"] = pd.to_datetime(exits["exit_date"]).dt.normalize()
    rows = []
    by_symbol_exit = exits.groupby("symbol") if not exits.empty else None
    for _, entry in entries.iterrows():
        sym = entry["symbol"]
        entry_px = float(entry["entry_price"])
        shares = float(entry.get("shares", 0) or 0)
        exit_row = None
        if by_symbol_exit is not None and sym in by_symbol_exit.groups:
            exit_row = (
                exits.loc[by_symbol_exit.groups[sym]].sort_values("exit_date").iloc[-1]
            )
        if exit_row is not None:
            exit_px = float(exit_row["exit_price"])
            exit_dt = pd.to_datetime(exit_row["exit_date"])  # normalized
            pnl_usd = float(exit_row["pnl"])
            reason = exit_row.get("exit_reason", "")
            meta = float(exit_row.get("meta_prob", np.nan))
        else:
            if hold_days > 0:
                horizon = target_date + pd.Timedelta(days=hold_days)
            else:
                horizon = pd.to_datetime(features_panel["date"].max())
            last_avail = features_panel[
                (features_panel["symbol"] == sym) & (features_panel["date"] <= horizon)
            ]
            if last_avail.empty:
                continue
            last = last_avail.sort_values("date").iloc[-1]
            exit_px = float(last["adj_close"])
            exit_dt = pd.to_datetime(last["date"])
            pnl_usd = (exit_px - entry_px) * shares
            reason = "open"
            meta = float(last.get("meta_prob", np.nan))
        pnl_pct = ((exit_px / entry_px) - 1.0) * 100.0 if entry_px else np.nan
        rows.append(
            {
                "symbol": sym,
                "entry_date": entry["entry_date"],
                "exit_date": exit_dt,
                "hold_days": (exit_dt - entry["entry_date"]).days,
                "entry_price": entry_px,
                "exit_price": exit_px,
                "shares": shares,
                "pnl_pct": pnl_pct,
                "pnl_usd": pnl_usd,
                "exit_reason": reason,
                "meta_prob_exit": meta,
            }
        )
    return pd.DataFrame(rows)


def _print_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        LOG.info("[replay] no trades generated")
        return
    total_closed = len(summary[summary["exit_reason"] != "open"])
    total_pnl = summary["pnl_usd"].sum()
    wins = summary[(summary["exit_reason"] != "open") & (summary["pnl_usd"] > 0)]
    win_rate = (len(wins) / total_closed * 100.0) if total_closed else 0.0
    LOG.info(
        "[replay] trades=%d closed=%d win-rate=%.1f%% pnl=$%.2f",
        len(summary),
        total_closed,
        win_rate,
        total_pnl,
    )
    display = summary.copy()
    display["entry_date"] = display["entry_date"].dt.strftime("%Y-%m-%d")
    display["exit_date"] = display["exit_date"].dt.strftime("%Y-%m-%d")
    display["pnl_pct"] = display["pnl_pct"].map(
        lambda v: f"{v:+.2f}%" if pd.notna(v) else "--"
    )
    display["pnl_usd"] = display["pnl_usd"].map(lambda v: f"{v:+.2f}")
    cols = [
        "symbol",
        "entry_date",
        "exit_date",
        "hold_days",
        "entry_price",
        "exit_price",
        "shares",
        "pnl_pct",
        "pnl_usd",
        "exit_reason",
    ]
    LOG.info("\n%s", display[cols].to_string(index=False))


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    load_env_files()
    LOG.setLevel("DEBUG" if args.verbose else "INFO")

    requested_date = pd.Timestamp(args.date).normalize()
    target_date = requested_date
    cfg_path = _resolve_config(args.style, args.config)
    cfg = load_yaml_config(str(cfg_path))
    risk = _risk_params(cfg)
    # Apply CLI overrides for ATR-based controls if provided
    try:
        if args.stop_atr_mult is not None:
            risk["stop_atr_mult"] = float(args.stop_atr_mult)
    except Exception:
        pass
    try:
        if args.take_profit_atr_mult is not None:
            risk["take_profit_atr_mult"] = float(args.take_profit_atr_mult)
    except Exception:
        pass
    try:
        if args.trail_atr_mult is not None:
            risk["trail_atr_mult"] = float(args.trail_atr_mult)
    except Exception:
        pass

    out_dir = _default_output_dir(args.output_dir, target_date)
    LOG.info("[replay] output directory -> %s", out_dir)

    universe_path = out_dir / "universe.txt"
    symbols = _maybe_fetch_universe(args.max_symbols, universe_path)

    lookback_start = target_date - pd.Timedelta(days=args.lookback_days)
    if args.hold_days and args.hold_days > 0:
        hold_end = target_date + pd.Timedelta(days=args.hold_days)
    else:
        hold_end = pd.Timestamp.today().normalize()
    LOG.info(
        "[replay] fetching features for %d symbols from %s to %s",
        len(symbols),
        lookback_start.date(),
        hold_end.date(),
    )
    features_panel = _fetch_features(
        symbols, lookback_start, hold_end, workers=int(args.download_workers)
    )
    features_panel.to_parquet(out_dir / "features_full.parquet", index=False)

    available_dates = sorted(features_panel["date"].unique())
    if not available_dates:
        LOG.warning("[replay] no feature dates available; aborting")
        return
    if requested_date not in available_dates:
        fallback = [d for d in available_dates if d <= requested_date]
        if not fallback:
            LOG.warning("[replay] no data on or before %s", requested_date.date())
            return
        target_date = pd.Timestamp(fallback[-1])
        LOG.info(
            "[replay] requested date %s not in panel; using %s",
            requested_date.date(),
            target_date.date(),
        )
    else:
        target_date = requested_date

    meta_model, calibrators, meta_calibrator = _ensure_meta_paths(cfg, cfg_path)

    news_path = None
    if not args.skip_news:
        news_provider = args.news_provider
        news_path = _build_news(
            symbols,
            target_date - pd.Timedelta(days=args.news_window),
            target_date,
            news_provider,
            args.news_max_symbols,
            out_dir / "news_sentiment.csv",
        )

    features_path = out_dir / "features_for_entry.parquet"
    features_panel.to_parquet(features_path, index=False)

    entries, positions_csv, _dec_log = _run_entry(
        features_path,
        universe_path,
        cfg_path,
        cfg,
        risk,
        meta_model,
        calibrators,
        meta_calibrator,
        news_path,
        args,
        out_dir,
        target_date,
    )

    if entries.empty:
        LOG.info("[replay] entry_loop produced no candidates")
        return

    monitor_end = pd.to_datetime(features_panel["date"].max())
    intraday_dir = None
    if risk.get("take_profit_atr_mult", 0.0) > 0:
        try:
            intraday_dir = _fetch_intraday_bars(
                entries["symbol"].astype(str).unique().tolist(),
                target_date,
                monitor_end,
                out_dir / "intraday",
                batch_days=90,
            )
        except Exception as exc:
            LOG.warning("[replay] intraday fetch encountered an error: %s", exc)
            intraday_dir = None

    exits_csv = out_dir / "exits.csv"
    monitor_state = out_dir / "monitor_state.csv"
    _run_monitor(
        features_panel,
        cfg_path,
        meta_model,
        calibrators,
        meta_calibrator,
        news_path,
        intraday_dir,
        positions_csv,
        exits_csv,
        monitor_state,
        target_date,
        args.hold_days,
        args.exit_threshold,
        args.exit_consecutive,
        risk.get("take_profit_atr_mult", 0.0),
        risk.get("trail_atr_mult", 0.0),
        out_dir,
    )

    exits = pd.read_csv(exits_csv) if exits_csv.exists() else pd.DataFrame()
    summary = _summarize_trades(
        entries, exits, positions_csv, features_panel, target_date, args.hold_days
    )
    summary_path = out_dir / "summary.csv"
    if not summary.empty:
        summary.to_csv(summary_path, index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2, default=str)
    )
    _print_summary(summary)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        LOG.error("[replay] failed: %s", exc)
        raise
