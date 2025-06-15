import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("Penguin.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical columns
df["sex"] = LabelEncoder().fit_transform(df["sex"])
df["island"] = LabelEncoder().fit_transform(df["island"])
df["species"] = LabelEncoder().fit_transform(df["species"])  # target

# Normalize numeric columns
features = df.drop(columns=["species"])
scaler = StandardScaler()
df[features.columns] = scaler.fit_transform(features)

# Save
df = pd.concat([df[features.columns], df["species"]], axis=1)
df.to_csv("penguin_processed.csv", index=False)
