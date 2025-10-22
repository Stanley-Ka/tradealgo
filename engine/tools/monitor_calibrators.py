from __future__ import annotations

"""Monitor calibration stability and optionally auto-fallback drifty calibrators.

Inputs:
- Decision log CSV from paper_trader (requires columns: date_decision, symbol, meta_prob, fret_1d_next)
- Optional features parquet + spec-config to recompute specialist scores for the window
- Calibrators pickle (per-specialist) to filter

Outputs:
- CSV report with Brier/AUC per specialist and meta
- Optionally writes a filtered calibrators pickle with drifty specs removed
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from ..features.specialists import compute_specialist_scores
from ..models.calib_utils import load_spec_calibrators, apply_calibrator, naive_prob_map
from ..infra.yaml_config import load_yaml_config


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monitor calibrator stability and auto-fallback on drift"
    )
    p.add_argument("--decision-log", required=True)
    p.add_argument(
        "--features",
        type=str,
        default="",
        help="Features parquet to recompute specialists (optional but recommended)",
    )
    p.add_argument(
        "--spec-config",
        type=str,
        default="",
        help="YAML with specialists params (optional)",
    )
    p.add_argument(
        "--calibrators-in",
        type=str,
        default="",
        help="Existing calibrators pickle to evaluate",
    )
    p.add_argument(
        "--calibrators-out",
        type=str,
        default="",
        help="Where to write filtered calibrators after fallback",
    )
    p.add_argument(
        "--report-csv", type=str, default="data/reports/calibration_monitor.csv"
    )
    p.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Rolling window (days) for decisions to evaluate",
    )
    p.add_argument(
        "--auc-floor",
        type=float,
        default=0.505,
        help="Disable calibrator if AUC falls below this",
    )
    p.add_argument(
        "--brier-ceil",
        type=float,
        default=0.26,
        help="Disable calibrator if Brier score exceeds this",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --calibrators-in with filtered output",
    )
    return p.parse_args(argv)


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    log = pd.read_csv(args.decision_log)
    if "date_decision" not in log.columns:
        raise RuntimeError("decision log missing date_decision column")
    log["date_decision"] = pd.to_datetime(log["date_decision"])  # already date-like
    log.sort_values("date_decision", inplace=True)
    # Window by last N days
    if len(log) == 0:
        print("[mon-cal] empty decision log")
        return
    last_day = log["date_decision"].max()
    start = last_day - pd.Timedelta(days=int(args.window_days))
    win = log[log["date_decision"] >= start].copy()
    if win.empty:
        print("[mon-cal] no rows in window; skipping")
        return
    # Meta metrics first
    y = (win.get("fret_1d_next", 0.0).astype(float) > 0).astype(int).values
    meta = {}
    if "meta_prob" in win.columns:
        p = win["meta_prob"].astype(float).values
        try:
            from sklearn.metrics import roc_auc_score  # type: ignore

            meta_auc = (
                float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
            )
        except Exception:
            meta_auc = float("nan")
        meta_brier = _brier(p, y)
        meta = {"spec": "meta", "auc": meta_auc, "brier": meta_brier}

    rows: List[dict] = []
    if meta:
        rows.append(meta)

    # If features provided, recompute specialists and evaluate per-spec calibrators
    disabled: List[str] = []
    if (
        args.features
        and os.path.exists(args.features)
        and args.calibrators_in
        and os.path.exists(args.calibrators_in)
    ):
        f = pd.read_parquet(args.features)
        f["date"] = pd.to_datetime(f["date"]).dt.normalize()
        f["symbol"] = f["symbol"].astype(str).str.upper()
        dates = sorted(win["date_decision"].unique())
        f_win = f[f["date"].isin(dates)].copy()
        cfg = load_yaml_config(args.spec_config) if args.spec_config else {}
        specs = compute_specialist_scores(
            f_win, news_sentiment=None, params=cfg.get("specialists", cfg)
        )
        cals = load_spec_calibrators(args.calibrators_in, None, kind="platt")
        # Build realized y per (date,symbol)
        yz = win[["date_decision", "symbol", "fret_1d_next"]].copy()
        yz.rename(columns={"date_decision": "date"}, inplace=True)
        yz["date"] = pd.to_datetime(yz["date"]).dt.normalize()
        yz["symbol"] = yz["symbol"].astype(str).str.upper()
        yz["y"] = (yz["fret_1d_next"].astype(float) > 0).astype(int)
        perf = {}
        for sc in [
            c
            for c in specs.columns
            if c.startswith("spec_") and not c.endswith("_prob")
        ]:
            sub = specs[["date", "symbol", sc]].merge(
                yz[["date", "symbol", "y"]], on=["date", "symbol"], how="inner"
            )
            if sub.empty:
                continue
            raw = sub[sc].astype(float).values
            yb = sub["y"].astype(int).values
            if sc in cals:
                p = apply_calibrator(cals[sc], raw)
            else:
                p = naive_prob_map(raw)
            try:
                from sklearn.metrics import roc_auc_score  # type: ignore

                auc = (
                    float(roc_auc_score(yb, p))
                    if len(np.unique(yb)) > 1
                    else float("nan")
                )
            except Exception:
                auc = float("nan")
            brier = _brier(p, yb)
            rows.append({"spec": sc, "auc": auc, "brier": brier})
            if (auc == auc and auc < float(args.auc_floor)) or (
                brier == brier and brier > float(args.brier_ceil)
            ):
                disabled.append(sc)

        # Write filtered calibrators if requested
        if args.calibrators_out:
            try:
                import pickle

                payload = {
                    "models": {k: v for k, v in cals.items() if k not in disabled},
                    "disabled": disabled,
                }
                with open(args.calibrators_out, "wb") as fpk:
                    pickle.dump(payload, fpk)
                if args.overwrite and os.path.abspath(
                    args.calibrators_out
                ) != os.path.abspath(args.calibrators_in):
                    import shutil

                    shutil.copyfile(args.calibrators_out, args.calibrators_in)
                print(
                    f"[mon-cal] filtered calibrators -> {args.calibrators_out}; disabled={disabled}"
                )
            except Exception as e:
                print(f"[mon-cal] warning: failed to write filtered calibrators: {e}")

    # Write report
    if rows:
        try:
            os.makedirs(os.path.dirname(args.report_csv), exist_ok=True)
            pd.DataFrame(rows).to_csv(args.report_csv, index=False)
            print(f"[mon-cal] report -> {args.report_csv}")
        except Exception as e:
            print(f"[mon-cal] warning: failed to write report: {e}")


if __name__ == "__main__":
    main()
