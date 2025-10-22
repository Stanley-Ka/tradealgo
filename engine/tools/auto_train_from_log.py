"""Auto-train calibrator and specialist weights from the paper entry log."""

from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import roc_auc_score, brier_score_loss  # type: ignore

from ..models.specialist_condition_weights import (
    WeightConfig,
    compute_condition_weights,
    save_weights,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-train from entry_log.csv")
    p.add_argument("--entry-log", default="data/paper/entry_log.csv")
    p.add_argument(
        "--skip-entry-log",
        action="store_true",
        help="Ignore the primary entry log even if it exists",
    )
    p.add_argument(
        "--log",
        dest="logs",
        action="append",
        default=[],
        help="Additional decision log (backtest, replay, etc.)",
    )
    p.add_argument("--min-rows", type=int, default=200)
    p.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicate date/symbol rows before training",
    )
    p.add_argument(
        "--lookahead-col",
        type=str,
        default="ret_3d",
        help="Outcome column to judge success",
    )
    p.add_argument("--success-threshold", type=float, default=0.0)
    p.add_argument(
        "--calibrator-out", type=str, default="data/models/meta_calibrator.auto.pkl"
    )
    p.add_argument(
        "--weights-out",
        type=str,
        default="data/models/specialist_condition_weights.json",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute diagnostics without writing outputs",
    )
    return p.parse_args()


def _prepare_log(path: str, dedupe: bool, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError("entry log is empty; collect more samples before training")
    required = {"meta_prob", "symbol"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"entry log missing required columns: {sorted(missing)}")
    df["meta_prob"] = pd.to_numeric(df["meta_prob"], errors="coerce").clip(
        1e-6, 1 - 1e-6
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["meta_prob"])
    if "source" not in df.columns:
        df["source"] = label
    if dedupe and {"date", "symbol"}.issubset(df.columns):
        df = df.sort_values(["date", "symbol"]).drop_duplicates(
            ["date", "symbol"], keep="last"
        )
    return df


def _fit_calibrator(
    df: pd.DataFrame, lookahead_col: str, threshold: float
) -> Dict[str, object]:
    col = lookahead_col
    if col not in df.columns:
        for alt in ("ret_1d_next", "fret_1d_next", "ret_1d", "fret_1d"):
            if alt in df.columns:
                print(
                    f"[auto-train] lookahead column '{lookahead_col}' missing; using '{alt}' instead"
                )
                col = alt
                break
        else:
            raise RuntimeError(
                f"entry log missing lookahead column '{lookahead_col}' and no fallback found"
            )
    y = pd.to_numeric(df[col], errors="coerce")
    y = np.where(y > threshold, 1, 0)
    if len(np.unique(y)) < 2:
        raise RuntimeError("Need both positive and negative outcomes to fit calibrator")
    x = df["meta_prob"].values.reshape(-1, 1)
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(x, y)
    prob = model.predict_proba(x)[:, 1]
    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
    try:
        brier = brier_score_loss(y, prob)
    except Exception:
        brier = float("nan")
    print(f"[auto-train] calibrator rows={len(df)} auc={auc:.4f} brier={brier:.6f}")
    return {
        "model": model,
        "kind": "platt",
        "trained_at": datetime.utcnow().isoformat(),
        "source": "auto_train_from_log",
        "metrics": {"auc": float(auc), "brier": float(brier)},
    }


def main() -> None:
    args = parse_args()
    frames: list[pd.DataFrame] = []
    if not args.skip_entry_log and args.entry_log:
        try:
            frames.append(_prepare_log(args.entry_log, args.dedupe, "live"))
            print(
                f"[auto-train] loaded live log -> {args.entry_log} rows={len(frames[-1])}"
            )
        except FileNotFoundError:
            print(f"[auto-train] live log missing -> {args.entry_log}")
    for idx, path in enumerate(args.logs or []):
        label = f"extra_{idx+1}"
        try:
            frm = _prepare_log(path, args.dedupe, label)
            frames.append(frm)
            print(f"[auto-train] loaded {label} -> {path} rows={len(frm)}")
        except FileNotFoundError:
            print(f"[auto-train] warning: log missing -> {path}")
    if not frames:
        raise RuntimeError(
            "No decision logs supplied; provide --entry-log or --log paths"
        )

    df = pd.concat(frames, ignore_index=True, sort=False)
    if len(df) < args.min_rows:
        raise RuntimeError(
            f"Need at least {args.min_rows} rows after merge; have {len(df)}"
        )

    cfg = WeightConfig(
        condition_col="condition_label",
        outcome_col=args.lookahead_col,
        success_threshold=args.success_threshold,
    )
    weights = compute_condition_weights(df, cfg)
    print(f"[auto-train] computed weights for {len(weights)} conditions")

    calibrator = _fit_calibrator(df, args.lookahead_col, args.success_threshold)

    if args.dry_run:
        print("[auto-train] dry-run complete; outputs not written")
        return

    os.makedirs(os.path.dirname(args.calibrator_out), exist_ok=True)
    with open(args.calibrator_out, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"[auto-train] saved calibrator -> {args.calibrator_out}")

    save_weights(weights, args.weights_out)
    print(f"[auto-train] saved weights -> {args.weights_out}")


if __name__ == "__main__":
    main()
