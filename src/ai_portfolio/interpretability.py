"""Model interpretation helpers for fitted scikit-learn pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def extract_feature_effects(model: Pipeline) -> pd.DataFrame | None:
    """Return transformed-feature coefficients/importances when available."""
    preprocessor = model.named_steps.get("preprocessor")
    estimator = model.named_steps.get("model")
    if preprocessor is None or estimator is None:
        return None

    feature_names = preprocessor.get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_)
        effect_type = "feature_importance"
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_)
        if coefficients.ndim != 2 or coefficients.shape[0] != 1:
            return None
        values = coefficients[0]
        effect_type = "coefficient"
    else:
        return None

    if len(feature_names) != len(values):
        return None

    frame = pd.DataFrame({"feature": feature_names, "effect": values, "absolute_effect": np.abs(values), "effect_type": effect_type})
    return frame.sort_values("absolute_effect", ascending=False).reset_index(drop=True)
