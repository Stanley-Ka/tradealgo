from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore


Calibrator = Callable[[np.ndarray], np.ndarray]


@dataclass
class FittedCalibrator:
    kind: str
    apply: Calibrator


def fit_platt(scores: Iterable[float], y: Iterable[int]) -> FittedCalibrator:
    X = np.asarray(scores, dtype=float).reshape(-1, 1)
    yy = np.asarray(y, dtype=int).ravel()
    # Strong regularization to avoid overfitting small folds
    lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
    lr.fit(X, yy)

    def _apply(s: np.ndarray) -> np.ndarray:
        return lr.predict_proba(s.reshape(-1, 1))[:, 1]

    return FittedCalibrator(kind="platt", apply=_apply)


def fit_isotonic(scores: Iterable[float], y: Iterable[int]) -> FittedCalibrator:
    x = np.asarray(scores, dtype=float)
    yy = np.asarray(y, dtype=int).ravel()
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x, yy)

    def _apply(s: np.ndarray) -> np.ndarray:
        return iso.transform(np.asarray(s, dtype=float))

    return FittedCalibrator(kind="isotonic", apply=_apply)

