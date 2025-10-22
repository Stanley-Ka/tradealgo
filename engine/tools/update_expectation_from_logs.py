from __future__ import annotations

"""Update YAML config's expected return mapping from historical decisions.

Computes k_hat from decisions and features, then writes risk.expected_k (and optionally
base_prob) to the provided YAML config. Safe-guards: only updates when enough observations
and k_hat is within a sane range.

Usage:
  python -m engine.tools.update_expectation_from_logs \
    --config engine/config.research.yaml \
    --features data/datasets/features_daily_1D.parquet \
    --decisions data/paper/entry_log.csv \
    --base-prob 0.5 --lookahead 5
"""

import argparse
from typing import Optional

import numpy as np
import pandas as pd
import yaml


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Update YAML risk.expected_k from decision logs"
    )
    p.add_argument("--config", required=True, help="YAML to update in-place")
    p.add_argument(
        "--features", required=True, help="Parquet with date,symbol,fret_1d,atr_pct_14"
    )
    p.add_argument(
        "--decisions",
        required=True,
        help="CSV with date/symbol/meta_prob (alert or entry log)",
    )
    p.add_argument("--base-prob", type=float, default=0.5)
    p.add_argument("--lookahead", type=int, default=5)
    p.add_argument(
        "--min-observations", type=int, default=30, help="Min sample size to update"
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_cfg(path: str, cfg: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def compute_k_hat(
    features: pd.DataFrame, decisions: pd.DataFrame, base_prob: float, lookahead: int
) -> tuple[float, int]:
    f = features.copy()
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    f["symbol"] = f["symbol"].astype(str).str.upper()
    d = decisions.copy()
    d_date_col = (
        "date"
        if "date" in d.columns
        else ("date_decision" if "date_decision" in d.columns else None)
    )
    if d_date_col is None:
        raise RuntimeError("decisions CSV must include 'date' or 'date_decision'")
    d["date"] = pd.to_datetime(d[d_date_col]).dt.normalize()
    d["symbol"] = d["symbol"].astype(str).str.upper()
    if "meta_prob" not in d.columns:
        raise RuntimeError("decisions CSV must include meta_prob")

    f = f.sort_values(["symbol", "date"])  # required order
    grp = {sym: df.reset_index(drop=True) for sym, df in f.groupby("symbol")}
    xs = []
    ys = []
    for _, row in d.iterrows():
        sym = str(row["symbol"]).upper()
        mp = float(row["meta_prob"]) if pd.notna(row["meta_prob"]) else np.nan
        if sym not in grp or not np.isfinite(mp):
            continue
        gi = grp[sym]
        # find decision index (or nearest prior)
        idx = gi.index[gi["date"] == row["date"]]
        if idx.size == 0:
            prev_idx = gi.index[gi["date"] < row["date"]]
            if prev_idx.size == 0:
                continue
            start = int(prev_idx.max())
        else:
            start = int(idx.max())
        if start + 1 >= len(gi):
            continue
        atrp = (
            float(gi.loc[start, "atr_pct_14"]) if "atr_pct_14" in gi.columns else np.nan
        )
        seq = gi.loc[start + 1 : start + int(lookahead), "fret_1d"].astype(float).values
        if seq.size == 0:
            continue
        cum = float(np.prod(1.0 + np.nan_to_num(seq, nan=0.0)) - 1.0)
        x = max(0.0, mp - float(base_prob)) * (atrp if np.isfinite(atrp) else 0.0)
        xs.append(x)
        ys.append(cum)
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    n = int(mask.sum())
    if n < 5:
        return 2.5, n  # fallback default
    k = float((x[mask] * y[mask]).sum() / max(1e-12, (x[mask] * x[mask]).sum()))
    # clamp to sane range
    return float(max(0.1, min(k, 10.0))), n


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    feats = pd.read_parquet(args.features)
    if "fret_1d" not in feats.columns or "atr_pct_14" not in feats.columns:
        raise RuntimeError("features must include fret_1d and atr_pct_14")
    decisions = pd.read_csv(args.decisions)
    k_hat, n = compute_k_hat(
        feats, decisions, float(args.base_prob), int(args.lookahead)
    )
    print(f"[update-exp] observations={n} k_hat={k_hat:.3f}")
    if n < int(args.min_observations):
        print(
            f"[update-exp] not enough observations (<{args.min_observations}); skip update"
        )
        return
    cfg = _load_cfg(args.config)
    if "risk" not in cfg or not isinstance(cfg.get("risk"), dict):
        cfg["risk"] = {}
    cfg["risk"]["base_prob"] = float(args.base_prob)
    cfg["risk"]["expected_k"] = float(k_hat)
    if args.dry_run:
        print("[update-exp] dry-run; suggested changes:")
        print(
            yaml.safe_dump(
                {
                    "risk": {
                        "base_prob": float(args.base_prob),
                        "expected_k": float(k_hat),
                    }
                },
                sort_keys=False,
            )
        )
        return
    _save_cfg(args.config, cfg)
    print(
        f"[update-exp] updated {args.config} -> risk.base_prob={args.base_prob:.3f}, risk.expected_k={k_hat:.3f}"
    )


if __name__ == "__main__":
    main()
