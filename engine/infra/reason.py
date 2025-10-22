from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from .market_state import state_as_dict


_WEIGHTS_PATH = os.environ.get(
    "SPECIALIST_WEIGHTS_PATH",
    "data/models/specialist_condition_weights.json",
)


def _load_weights() -> Dict[str, Dict[str, float]]:
    path = _WEIGHTS_PATH
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
            if isinstance(data, dict):
                return {
                    str(cond): {str(k): float(v) for k, v in (weights or {}).items()}
                    for cond, weights in data.items()
                }
    except Exception:
        return {}
    return {}


@lru_cache(maxsize=1)
def _weight_cache() -> Dict[str, Dict[str, float]]:
    return _load_weights()


def _weights_for_condition(condition: str) -> Dict[str, float]:
    weights = _weight_cache()
    cond = weights.get(condition) or {}
    baseline = weights.get("__global__", {})
    merged = dict(baseline)
    merged.update(cond)
    return merged


def consensus_for_symbol(specs_df: pd.DataFrame, symbol: str) -> str:
    """Build a short consensus string from specialist probabilities for a symbol.

    Looks for columns named like 'spec_XXX_prob', takes top-3 by value.
    Returns a comma-separated string like: 'technicals 0.62, patterns 0.57, sequence 0.55'.
    """
    try:
        cols = [
            c for c in specs_df.columns if c.startswith("spec_") and c.endswith("_prob")
        ]
        mask = specs_df["symbol"].astype(str).str.upper() == str(symbol).upper()
        row = specs_df.loc[mask]
        if row.empty:
            return ""
        row = row.iloc[0]
        state = state_as_dict(row)
        weight_map = _weights_for_condition(state.get("condition_label", ""))
        pairs = []
        for c, v in row.items():
            name = c.replace("spec_", "").replace("_prob", "")
            try:
                prob = float(v)
            except Exception:
                continue
            weight = float(weight_map.get(name, 1.0))
            try:
                pairs.append((name, prob, prob * weight, weight))
            except Exception:
                continue
        pairs.sort(key=lambda x: x[2], reverse=True)
        out = []
        for name, prob, score, weight in pairs[:3]:
            if abs(weight - 1.0) <= 1e-3:
                out.append(f"{name} {prob:.2f}")
            else:
                out.append(f"{name} {prob:.2f}×{weight:.2f}")
        return ", ".join(out)
    except Exception:
        return ""


def expected_return_and_horizon(
    meta_prob: float,
    atr_pct: float,
    base_prob: Optional[float] = 0.5,
    k_scale: Optional[float] = None,
    cap_mult: float = 2.0,
    cut1: float = 0.55,
    cut2: float = 0.60,
) -> Tuple[float, str]:
    """Heuristic expected return (%) and time horizon based on conviction and ATR.

    - expected_ret_pct = clamp( 100 * k * max(0, meta_prob - base_prob) * atr_pct, 0, 2 * 100 * atr_pct )
    - horizon: 1–2d if strong (>=0.60), 2–3d if medium (>=0.55), else 3–5d
    """
    try:
        mp = float(meta_prob)
        atrp = float(atr_pct) if np.isfinite(atr_pct) else 0.01
        bp = float(base_prob if base_prob is not None else 0.5)
        edge = max(0.0, mp - bp)
        k = float(k_scale) if k_scale is not None else 2.5
        exp = 100.0 * k * edge * atrp
        exp = float(max(0.0, min(exp, cap_mult * 100.0 * atrp)))
    except Exception:
        exp = float("nan")
    try:
        horizon = "3–5 days" if mp < cut1 else ("2–3 days" if mp < cut2 else "1–2 days")
    except Exception:
        horizon = "2–4 days"
    return exp, horizon
