import pandas as pd

from ai_portfolio.data import build_preprocessor, load_telco_dataset


def test_load_telco_dataset_and_preprocessor(tmp_path):
    frame = pd.DataFrame(
        {
            "customerID": ["A", "B", "C"],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "Yes"],
            "tenure": [1, 24, 12],
            "MonthlyCharges": [29.85, 80.0, 55.0],
            "TotalCharges": ["29.85", " ", "660.0"],
            "Churn": ["No", "Yes", "No"],
        }
    )
    path = tmp_path / "telco.csv"
    frame.to_csv(path, index=False)

    bundle = load_telco_dataset(path)

    assert list(bundle.y) == [0, 1, 0]
    assert "customerID" not in bundle.X.columns
    assert "TotalCharges" in bundle.numeric_features
    assert "gender" in bundle.categorical_features

    transformer = build_preprocessor(bundle.numeric_features, bundle.categorical_features)
    transformed = transformer.fit_transform(bundle.X)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] >= 4
