# Telco Customer Churn - Data Preprocessing Script
# Coursework 1 - CMP-X316-0

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Original notebook used a local Windows path. Set DATA_PATH to your local CSV.
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(DATA_PATH)

print("Initial Shape:", df.shape)
print(df.info())
print("Missing Values:\n", df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())

df.columns = df.columns.str.strip()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df.drop_duplicates(inplace=True)
df.drop(["customerID"], axis=1, inplace=True)

binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

df = pd.get_dummies(df, drop_first=True)
scaler = StandardScaler()
numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
df[numerical_features] = scaler.fit_transform(df[numerical_features])

df.describe().to_csv("telco_summary_statistics.csv")

plt.figure(figsize=(8, 5))
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.tight_layout()
plt.savefig("churn_distribution.png")
plt.close()

plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

df.to_csv("telco_processed.csv", index=False)
print("Preprocessing complete.")
