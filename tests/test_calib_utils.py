from __future__ import annotations

import numpy as np
import pandas as pd

from engine.models.calib_utils import fit_per_specialist_calibrators_from_oof


def test_fit_per_specialist_skips_degenerate_single_class():
    df = pd.DataFrame(
        {
            "spec_good_raw": [-0.5, -0.2, 0.2, 0.6],
            # Only finite rows for positive class -> should be skipped gracefully.
            "spec_bad_raw": [np.nan, np.nan, 0.1, 0.3],
            "y_true": [0, 0, 1, 1],
        }
    )

    models = fit_per_specialist_calibrators_from_oof(df, kind="platt")

    assert "spec_good" in models  # fitted on mixed classes
    assert "spec_bad" not in models  # skipped instead of raising
