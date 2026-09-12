import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

df = pd.read_csv("PastHires.csv")
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = LabelEncoder().fit_transform(df[column])

X = df.drop("Hired", axis=1)
y = df["Hired"]
model = DecisionTreeClassifier(random_state=42).fit(X, y)

sample_1 = [[10, 1, 4, 0, 0, 0]]
sample_2 = [[1, 0, 4, 0, 0, 0]]
print("Sample 1 Prediction:", "Hired" if model.predict(sample_1)[0] == 1 else "Not Hired")
print("Sample 2 Prediction:", "Hired" if model.predict(sample_2)[0] == 1 else "Not Hired")
print(export_text(model, feature_names=list(X.columns)))
