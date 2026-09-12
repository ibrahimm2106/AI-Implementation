# Dataset setup

The flagship project uses the Telco Customer Churn dataset used in the original coursework.

Expected local filename:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The raw file contains **7,043 customer records and 21 columns** in the copy used for the project. The target is `Churn` (`Yes`/`No`).

Key fields include:

- demographic/service indicators such as `gender`, `SeniorCitizen`, `Partner`, and `Dependents`
- account tenure and contract information
- internet, security, support, and streaming services
- `MonthlyCharges` and `TotalCharges`
- the binary `Churn` target

## Why the data file is not committed

The repository keeps datasets out of version control so the source remains lightweight and the code/data boundary is clear. Put your own authorised copy of the CSV in this directory before running the training script.

The preprocessing code converts `TotalCharges` to numeric, removes the customer identifier, maps the target to 0/1, and performs train-fold-only imputation/scaling/encoding inside scikit-learn pipelines.
