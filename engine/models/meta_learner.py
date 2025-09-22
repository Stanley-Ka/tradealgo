"""Meta-learner stub that combines calibrated probabilities and context features.

Replace with a trained logistic/XGBoost/stacking model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass
class MetaLearner:
    """Placeholder meta-learner combining five probabilities and context.

    For now: simple clipped linear blend to enable wiring tests.
    """

    weights: Mapping[str, float] | None = None

    def predict_proba(self, p: Mapping[str, float], context: Mapping[str, float] | None = None) -> float:
        _ = context
        keys = ["pattern", "technical", "sequence", "nlp", "alt"]
        if self.weights is None:
            w = {k: 1.0 for k in keys}
        else:
            w = {k: float(self.weights.get(k, 0.0)) for k in keys}
        num = sum(w[k] * float(p.get(k, 0.5)) for k in keys)
        den = sum(w.values()) or 1.0
        y = num / den
        return max(0.0, min(1.0, y))

