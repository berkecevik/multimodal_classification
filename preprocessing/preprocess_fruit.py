import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load original dataset
df = pd.read_csv("fruits_weight_sphercity.csv")

# Encode color (Red=80, Orange=20)
color_map = {"Red": 80, "Orange": 20}
df["Color"] = df["Color"].map(color_map)

# Encode label (apple=0, orange=1)
label_map = {"apple": 0, "orange": 1}
df["Label"] = df["labels"].map(label_map)

# Drop old label column
df = df.drop(columns=["labels"])

# Standardize features
features = ["Weight", "Sphericity", "Color"]
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Save to a new CSV
df.to_csv("fruit_processed.csv", index=False)
