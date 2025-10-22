from __future__ import annotations

"""Intraday pipeline: build snapshot -> predict/entry from intraday features.

Uses the paid Polygon minute aggregates (15-min delayed on Starter plan) to
produce an intraday snapshot and select entries with adaptive risk and caps.

Typical usage:
  python -m engine.tools.intraday_pipeline --config engine/config.intraday.example.yaml

YAML keys (example in engine/config.intraday.example.yaml):
  bars:
    root: data/equities/polygon
    interval: 1m
    lookback_bars: 200
  paths:
    intraday_features: data/datasets/features_intraday_latest.parquet
    meta_model: data/models/meta_lr.pkl
    calibrators_pkl: data/models/spec_calibrators.pkl
  entry:
    universe_file: engine/data/universe/nasdaq100.example.txt
    top_k: 5
    entry_threshold: 0.50
    confirmations: 2
    sector_map_csv: data/mappings/symbol_sector.example.csv
    sector_cap: 1
    price_source: live
    live_provider: yahoo
    positions_csv: data/paper/positions.csv
  risk:
    account_equity: 20000
    risk_mode: auto
    min_risk_pct: 0.0035
    max_risk_pct: 0.0070
    risk_curve: sqrt
    base_prob: 0.52
    stop_atr_mult: 1.0
    max_name_weight: 0.10
    # max_position_notional: 2000
"""

import argparse
from typing import List, Optional

from .build_intraday_latest import main as build_snapshot
from .entry_loop import main as entry_main
from ..infra.yaml_config import load_yaml_config


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intraday snapshot -> entry pipeline")
    p.add_argument("--config", required=True, help="YAML with bars/paths/entry/risk")
    return p.parse_args(argv)


def _cfg_get(cfg: dict, path: List[str], default=None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_yaml_config(args.config)
    # 1) Build intraday snapshot
    bars_root = _cfg_get(cfg, ["bars", "root"], "data/equities/polygon")
    interval = _cfg_get(cfg, ["bars", "interval"], "1m")
    lookback = str(_cfg_get(cfg, ["bars", "lookback_bars"], 200))
    snap_out = _cfg_get(
        cfg,
        ["paths", "intraday_features"],
        "data/datasets/features_intraday_latest.parquet",
    )
    build_snapshot(
        [
            "--bars-root",
            bars_root,
            "--interval",
            interval,
            "--lookback-bars",
            lookback,
            "--out",
            snap_out,
        ]
    )
    # 2) Entry loop from snapshot
    uni = _cfg_get(cfg, ["entry", "watchlist_file"], "") or _cfg_get(
        cfg, ["entry", "universe_file"], ""
    )
    model = _cfg_get(cfg, ["paths", "meta_model"], "")
    if not (uni and model and snap_out):
        raise RuntimeError(
            "entry.universe_file, paths.meta_model, and paths.intraday_features required in YAML"
        )
    topk = str(_cfg_get(cfg, ["entry", "top_k"], 5))
    thr = str(_cfg_get(cfg, ["entry", "entry_threshold"], 0.0))
    conf = str(_cfg_get(cfg, ["entry", "confirmations"], 1))
    sector_map = _cfg_get(cfg, ["entry", "sector_map_csv"], "")
    sector_cap = _cfg_get(cfg, ["entry", "sector_cap"], 0)
    price_src = _cfg_get(cfg, ["entry", "price_source"], "feature")
    live_prov = _cfg_get(cfg, ["entry", "live_provider"], "yahoo")
    pos_csv = _cfg_get(cfg, ["entry", "positions_csv"], "data/paper/positions.csv")
    calibs = _cfg_get(cfg, ["paths", "calibrators_pkl"], "")
    oof = _cfg_get(cfg, ["paths", "oof"], "")
    # Risk
    r = cfg.get("risk", {}) if isinstance(cfg.get("risk", {}), dict) else {}
    acct = r.get("account_equity")
    risk_mode = r.get("risk_mode")
    rpct = r.get("risk_pct")
    rmin = r.get("min_risk_pct")
    rmax = r.get("max_risk_pct")
    rcurve = r.get("risk_curve")
    rbase = r.get("base_prob")
    stop_mult = r.get("stop_atr_mult")
    max_w = r.get("max_name_weight")
    max_notional = r.get("max_position_notional")

    eargs: List[str] = [
        "--intraday-features",
        snap_out,
        "--model-pkl",
        model,
        "--universe-file",
        uni,
        "--top-k",
        topk,
        "--confirmations",
        conf,
        "--price-source",
        price_src,
        "--live-provider",
        live_prov,
        "--positions-csv",
        pos_csv,
    ]
    if float(thr) > 0:
        eargs += ["--entry-threshold", thr]
    if sector_map:
        eargs += ["--sector-map-csv", sector_map]
    if int(sector_cap) > 0:
        eargs += ["--sector-cap", str(sector_cap)]
    if calibs:
        eargs += ["--calibrators-pkl", calibs]
    elif oof:
        eargs += ["--oof", oof]
    # Forward risk
    if acct is not None:
        eargs += ["--account-equity", str(acct)]
    if risk_mode:
        eargs += ["--risk-mode", str(risk_mode)]
    if rpct is not None:
        eargs += ["--risk-pct", str(rpct)]
    if rmin is not None:
        eargs += ["--risk-min-pct", str(rmin)]
    if rmax is not None:
        eargs += ["--risk-max-pct", str(rmax)]
    if rcurve:
        eargs += ["--risk-curve", str(rcurve)]
    if rbase is not None:
        eargs += ["--risk-base-prob", str(rbase)]
    if stop_mult is not None:
        eargs += ["--stop-atr-mult", str(stop_mult)]
    if max_w is not None:
        eargs += ["--max-name-weight", str(max_w)]
    if max_notional is not None:
        eargs += ["--max-position-notional", str(max_notional)]

    entry_main(eargs)


if __name__ == "__main__":
    main()
