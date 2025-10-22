"""Utilities to derive specialist weights from decision logs by market condition."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeightConfig:
    condition_col: str = "condition_label"
    outcome_col: str = "ret_3d"
    success_threshold: float = 0.0
    min_rows: int = 30
    min_condition_rows: int = 10
    spec_prefix: str = "spec_"
    spec_suffix: str = "_prob"


def _success_series(df: pd.DataFrame, outcome_col: str, threshold: float) -> pd.Series:
    if outcome_col in df.columns:
        vals = pd.to_numeric(df[outcome_col], errors="coerce")
        return (vals > threshold).astype(int)
    # Fallback to binary hit columns if outcome missing
    for candidate in ("hit_5d", "hit_3d", "hit_1d"):
        if candidate in df.columns:
            return pd.to_numeric(df[candidate], errors="coerce").fillna(0).astype(int)
    raise RuntimeError(
        f"Could not derive success label from outcome column '{outcome_col}'"
    )


def _detect_spec_cols(df: pd.DataFrame, cfg: WeightConfig) -> Tuple[str, ...]:
    cols = [
        c
        for c in df.columns
        if c.startswith(cfg.spec_prefix) and c.endswith(cfg.spec_suffix)
    ]
    return tuple(sorted(cols))


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    arr = np.array(list(weights.values()), dtype=float)
    if arr.size == 0:
        return weights
    avg = float(np.nanmean(arr))
    if not np.isfinite(avg) or avg == 0:
        return weights
    return {k: float(v / avg) for k, v in weights.items()}


def compute_condition_weights(
    df: pd.DataFrame, cfg: WeightConfig | None = None
) -> Dict[str, Dict[str, float]]:
    cfg = cfg or WeightConfig()
    if len(df) < cfg.min_rows:
        raise RuntimeError(
            f"Not enough rows to compute weights: {len(df)} < {cfg.min_rows}"
        )

    spec_cols = _detect_spec_cols(df, cfg)
    if not spec_cols:
        raise RuntimeError(
            "Decision log does not contain specialist probability columns"
        )

    if cfg.condition_col not in df.columns:
        print(
            f"[weights] warning: missing condition column '{cfg.condition_col}'; using single global bucket"
        )
        df[cfg.condition_col] = "__global__"

    data = df.copy()
    data[cfg.condition_col] = data[cfg.condition_col].astype(str).str.lower()
    data["success"] = _success_series(data, cfg.outcome_col, cfg.success_threshold)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[cfg.condition_col])

    if data.empty:
        raise RuntimeError("No usable rows after cleaning decision log")

    result: Dict[str, Dict[str, float]] = {}
    global_scores: Dict[str, float] = {}
    global_counts: Dict[str, float] = {}

    epsilon = 1e-6

    for condition, grp in data.groupby(cfg.condition_col):
        if len(grp) < cfg.min_condition_rows:
            continue
        weights: Dict[str, float] = {}
        for col in spec_cols:
            name = col.replace(cfg.spec_prefix, "").replace(cfg.spec_suffix, "")
            probs = pd.to_numeric(grp[col], errors="coerce").clip(lower=0.0, upper=1.0)
            wins = float((probs * grp["success"]).sum())
            losses = float((probs * (1 - grp["success"])).sum())
            score = (wins + epsilon) / (wins + losses + epsilon)
            weights[name] = score
            global_scores[name] = global_scores.get(name, 0.0) + wins
            global_counts[name] = global_counts.get(name, 0.0) + wins + losses
        result[condition] = _normalize(weights)

    if global_scores:
        global_weights = {
            name: (global_scores[name] + epsilon)
            / (global_counts.get(name, 0.0) + epsilon)
            for name in global_scores
        }
        result["__global__"] = _normalize(global_weights)

    if not result:
        raise RuntimeError(
            "No condition met the minimum sample threshold for weight computation"
        )

    return result


def save_weights(weights: Dict[str, Dict[str, float]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2, sort_keys=True)
