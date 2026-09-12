# Coursework 1 Report — Portfolio Text Edition

> Based on the submitted coursework report. Personal student identifiers and local file paths are omitted from this public edition.

## Project objective

The project prepares the IBM-style Telco Customer Churn dataset for supervised machine learning. The target, `Churn`, indicates whether a customer left the service. The original dataset contains **7,043 records and 21 attributes** covering demographics, services, contracts and billing.

## Data preparation

The submitted workflow inspected shape, data types, missingness and duplicates; converted `TotalCharges` to numeric; used median imputation; removed the non-predictive `customerID`; encoded Yes/No fields; one-hot encoded multi-class categoricals; standardised `tenure`, `MonthlyCharges` and `TotalCharges`; and exported the processed dataset.

## Data understanding

The churn target is imbalanced, with roughly 26.5% of customers in the churn class. The coursework therefore identified class imbalance as an important modelling concern for the second coursework. Visual exploration also highlighted customer tenure, charges and contract/service features as useful candidates for later prediction.

## Reflection

The work demonstrates why data quality, consistent encoding and scale-aware preprocessing matter before model development. The later portfolio refactor moves learned transformations inside scikit-learn pipelines during modelling to reduce leakage risk.

## Continuation

Coursework 2 uses the prepared data to compare supervised classification models, tune hyperparameters and evaluate performance using metrics suited to imbalanced classification.
