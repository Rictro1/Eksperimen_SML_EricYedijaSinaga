
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESSING_DIR = os.path.join(BASE_DIR, "preprocessing")

RAW_PATH = os.path.join(BASE_DIR, "winequality_raw.csv")
OUTPUT_PATH = os.path.join(PREPROCESSING_DIR, "winequality_preprocessing.csv")

df = pd.read_csv(RAW_PATH)

if "Id" in df.columns:
    df = df.drop(columns=["Id"])

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

df = df.drop_duplicates().reset_index(drop=True)

df["quality_label"] = np.where(df["quality"] >= 7, 1, 0)

eps = 1e-6

df["alcohol_density_ratio"] = df["alcohol"] / (df["density"] + eps)
df["acid_ratio"] = df["fixed acidity"] / (df["volatile acidity"] + eps)
df["sulfur_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + eps)
df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]
df["sweetness_acidity_ratio"] = df["residual sugar"] / (df["total_acidity"] + eps)

X = df.drop(columns=["quality", "quality_label"])
y = df["quality_label"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

X_clipped = X.copy()

for col in numeric_features:
    Q1 = X_clipped[col].quantile(0.25)
    Q3 = X_clipped[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    X_clipped[col] = X_clipped[col].clip(
        lower=lower_bound,
        upper=upper_bound
    )

scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X_clipped)

X_scaled = pd.DataFrame(
    X_scaled_array,
    columns=X_clipped.columns
)

winequality_preprocessing = X_scaled.copy()
winequality_preprocessing["quality_label"] = y.reset_index(drop=True)

winequality_preprocessing.to_csv(OUTPUT_PATH, index=False)

print("Preprocessing selesai.")
print("File tersimpan di:", OUTPUT_PATH)
