# Data setup

The flagship project uses the **Telco Customer Churn** dataset from the original coursework.

Expected local filename for the portfolio training command:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The copy used in the coursework contains **7,043 customer records and 21 columns** with `Churn` as the binary target. It includes demographic/service indicators, account tenure and contract information, internet/support services, `MonthlyCharges` and `TotalCharges`.

## Why large datasets are not duplicated here

This public portfolio intentionally keeps large raw/processed dataset copies out of version control. That keeps the repository lightweight, avoids republishing duplicate third-party data and makes the boundary between code and data explicit.

Place an authorised copy of the raw Telco CSV in this directory before running the experiment. The machine-readable **coursework result CSVs** that are useful for verification are preserved under [`../coursework/coursework-2/outputs/`](../coursework/coursework-2/outputs/).

## Portfolio preprocessing

The engineering version:

- drops the customer identifier from model features;
- converts `TotalCharges` to numeric;
- maps the target to binary values;
- performs imputation, scaling and one-hot encoding inside scikit-learn pipelines so transformations are fitted from training data rather than the full dataset.

See [`../src/ai_portfolio/data.py`](../src/ai_portfolio/data.py) for the implementation.
