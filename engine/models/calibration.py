from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from sklearn.isotonic import IsotonicRegression  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore


Calibrator = Callable[[np.ndarray], np.ndarray]


@dataclass
class FittedCalibrator:
    kind: str
    apply: Calibrator


def fit_platt(scores: Iterable[float], y: Iterable[int]) -> FittedCalibrator:
    x = np.asarray(scores, dtype=float).ravel()
    yy = np.asarray(y, dtype=int).ravel()
    mask = np.isfinite(x)
    x_fit = x[mask]
    y_fit = yy[mask]
    # Fallback if insufficient data or single class
    if x_fit.size < 10 or len(np.unique(y_fit)) < 2:

        def _apply_naive(s: np.ndarray) -> np.ndarray:
            s2 = np.asarray(s, dtype=float)
            return np.clip((s2 + 1.0) * 0.5, 0.0, 1.0)

        return FittedCalibrator(kind="platt", apply=_apply_naive)
    lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
    lr.fit(x_fit.reshape(-1, 1), y_fit)

    def _apply(s: np.ndarray) -> np.ndarray:
        return lr.predict_proba(np.asarray(s, dtype=float).reshape(-1, 1))[:, 1]

    return FittedCalibrator(kind="platt", apply=_apply)


def fit_isotonic(scores: Iterable[float], y: Iterable[int]) -> FittedCalibrator:
    x = np.asarray(scores, dtype=float).ravel()
    yy = np.asarray(y, dtype=int).ravel()
    mask = np.isfinite(x)
    x_fit = x[mask]
    y_fit = yy[mask]
    # Fallback if insufficient data or single class
    if x_fit.size < 10 or len(np.unique(y_fit)) < 2:

        def _apply_naive(s: np.ndarray) -> np.ndarray:
            s2 = np.asarray(s, dtype=float)
            return np.clip((s2 + 1.0) * 0.5, 0.0, 1.0)

        return FittedCalibrator(kind="isotonic", apply=_apply_naive)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x_fit, y_fit)

    def _apply(s: np.ndarray) -> np.ndarray:
        return iso.transform(np.asarray(s, dtype=float))

    return FittedCalibrator(kind="isotonic", apply=_apply)
