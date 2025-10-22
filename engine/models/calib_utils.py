from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def fit_per_specialist_calibrators_from_oof(
    oof: pd.DataFrame, kind: str = "platt"
) -> Dict[str, object]:
    """Fit a calibrator per specialist from OOF raw->y_true pairs.

    Expects columns like spec_foo_raw and a column y_true.
    Returns {"spec_foo": model, ...}
    """
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.isotonic import IsotonicRegression  # type: ignore

    models: Dict[str, object] = {}
    raw_cols = [c for c in oof.columns if c.endswith("_raw")]
    for sc in raw_cols:
        base = sc[:-4]
        x = oof[sc].astype(float).values
        y = oof["y_true"].astype(int).values
        mask = np.isfinite(x)
        x = x[mask]
        y = y[mask]
        if x.size == 0 or len(np.unique(y)) <= 1:
            # Insufficient data or single-class label — skip to avoid training errors.
            continue
        try:
            if kind == "platt":
                lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
                lr.fit(x.reshape(-1, 1), y)
                models[base] = lr
            else:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(x, y)
                models[base] = iso
        except Exception:
            # Training can fail on degenerate datasets; fall back to skipping this specialist.
            continue
    return models


def apply_calibrator(model: object, x: np.ndarray) -> np.ndarray:
    import numpy as _np

    try:
        return model.predict_proba(x.reshape(-1, 1))[:, 1]
    except Exception:
        try:
            return _np.asarray(model.predict(x), dtype=float)
        except Exception:
            return _np.asarray(x, dtype=float)


def naive_prob_map(x: np.ndarray) -> np.ndarray:
    # Map [-1,1] -> [0,1]
    return np.clip((np.asarray(x, dtype=float) + 1.0) * 0.5, 0.0, 1.0)


def load_spec_calibrators(
    calibrators_pkl: str | None = None, oof_path: str | None = None, kind: str = "platt"
) -> Dict[str, object]:
    """Load per-specialist calibrators from a pickle or fit from OOF.

    Returns a mapping {spec_col_name: calibrator} (e.g., {"spec_technical": model}).
    """
    models: Dict[str, object] = {}
    if calibrators_pkl:
        try:
            import pickle

            with open(calibrators_pkl, "rb") as f:
                payload = pickle.load(f)
            models = payload.get("models", {})
            return models
        except Exception:
            return {}
    if oof_path:
        try:
            oof = pd.read_parquet(oof_path)
            return fit_per_specialist_calibrators_from_oof(oof, kind)
        except Exception:
            return {}
    return {}


def apply_meta_calibrator(
    calibrator_or_path: object | str | None,
    probs: np.ndarray,
    regime_vals: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a meta-level calibrator to a probability vector.

    Supports:
      - raw sklearn calibrator (predict_proba/transform)
      - path to pickle containing {model}
      - dict payload with either {model} or {by_regime: {feature, bins, models}}
    If by_regime is present and regime_vals provided, choose calibrator per regime bin.
    """
    if calibrator_or_path is None:
        return probs
    mdl = None
    if isinstance(calibrator_or_path, str):
        try:
            import pickle

            with open(calibrator_or_path, "rb") as f:
                payload = pickle.load(f)
            # Allow per-regime payload via file
            if isinstance(payload, dict) and "by_regime" in payload:
                return apply_meta_calibrator(payload, probs, regime_vals)
            mdl = payload.get("model", payload)
        except Exception:
            return probs
    elif isinstance(calibrator_or_path, dict):
        payload = calibrator_or_path
        # Per-regime calibrator
        if (
            "by_regime" in payload
            and isinstance(payload["by_regime"], dict)
            and regime_vals is not None
        ):
            br = payload["by_regime"]
            import numpy as _np

            bins = _np.asarray(br.get("bins", []), dtype=float)
            models = br.get("models", [])
            if (
                bins.size >= 2
                and isinstance(models, (list, tuple))
                and len(models) == bins.size - 1
            ):
                out = _np.asarray(probs, dtype=float).copy()
                rv = _np.asarray(regime_vals, dtype=float)
                idx = _np.digitize(rv, bins, right=False) - 1
                idx = _np.clip(idx, 0, len(models) - 1)
                for i in range(len(models)):
                    mask = idx == i
                    if not _np.any(mask):
                        continue
                    mdl_i = models[i]
                    try:
                        if hasattr(mdl_i, "predict_proba"):
                            out[mask] = mdl_i.predict_proba(out[mask].reshape(-1, 1))[
                                :, 1
                            ]
                        elif hasattr(mdl_i, "transform"):
                            out[mask] = mdl_i.transform(out[mask])
                    except Exception:
                        pass
                return out
        mdl = payload.get("model", payload)
    else:
        mdl = calibrator_or_path
    try:
        if hasattr(mdl, "predict_proba"):
            return mdl.predict_proba(probs.reshape(-1, 1))[:, 1]
        if hasattr(mdl, "transform"):
            return mdl.transform(probs)
    except Exception:
        return probs
    return probs
