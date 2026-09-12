import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ai_portfolio.interpretability import extract_feature_effects


def test_extract_feature_effects_for_linear_pipeline():
    X = pd.DataFrame({"tenure": [1, 2, 20, 24], "contract": ["Month", "Month", "Year", "Year"]})
    y = [1, 1, 0, 0]
    preprocessor = ColumnTransformer([("num", StandardScaler(), ["tenure"]), ("cat", OneHotEncoder(handle_unknown="ignore"), ["contract"])])
    model = Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression())])
    model.fit(X, y)
    effects = extract_feature_effects(model)
    assert effects is not None
    assert set(effects.columns) == {"feature", "effect", "absolute_effect", "effect_type"}
    assert (effects["effect_type"] == "coefficient").all()
