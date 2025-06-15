import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Iris.csv")

# Drop ID
df.drop(columns=["Id"], inplace=True)

# Encode species
species_map = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}
df["Species"] = df["Species"].map(species_map)

# Optional: normalize features (helpful for KNN)
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Save
df.to_csv("iris_processed.csv", index=False)
