import pandas as pd
from sklearn.preprocessing import StandardScaler

# Original exercise: preview, assess missingness, impute, scale, encode and export.
df = pd.read_excel("HousingData.xlsx")
print(df.head())
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include="number").columns
categorical_cols = df.select_dtypes(include="object").columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
for col in categorical_cols:
    df[col] = df[col].fillna("Missing")

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df.to_csv("HousingData_processed.csv", index=False)
