import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Breast_Cancer.csv")

# Drop ID and unnamed column (if any)
df.drop(columns=[col for col in df.columns if "id" in col.lower() or "Unnamed" in col], inplace=True)

# Encode diagnosis: M = 1, B = 0
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Normalize all numeric columns
features = df.drop(columns=["diagnosis"])
scaler = StandardScaler()
df[features.columns] = scaler.fit_transform(features)

# Save
df.to_csv("cancer_processed.csv", index=False)
