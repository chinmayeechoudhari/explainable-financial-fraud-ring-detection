import pandas as pd
import numpy as np

# =========================
# Load Processed Dataset
# =========================

file_path = "data/processed/processed_transactions.csv"

print("Loading processed dataset...")

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# =========================
# Transaction Amount Features
# =========================

print("\nCreating transaction amount features...")

# Difference between amount received and amount paid
df["Amount Difference"] = (
    df["Amount Received"] - df["Amount Paid"]
)

# Ratio between amount received and amount paid
df["Amount Ratio"] = (
    df["Amount Received"] /
    df["Amount Paid"]
)

print("Transaction amount features created successfully!")

print("\nSample:")
print(
    df[
        [
            "Amount Received",
            "Amount Paid",
            "Amount Difference",
            "Amount Ratio"
        ]
    ].head()
)

# =========================
# Bank Relationship Feature
# =========================

print("\nCreating bank relationship feature...")

# Check whether sender and receiver belong to the same bank
df["Same Bank Transaction"] = (
    df["From Bank"] == df["To Bank"]
).astype(int)

print("Bank relationship feature created successfully!")

print("\nSample:")
print(
    df[
        [
            "From Bank",
            "To Bank",
            "Same Bank Transaction"
        ]
    ].head()
)

# =========================
# Cross-Bank Transaction Feature
# =========================

print("\nCreating cross-bank transaction feature...")

df["Cross Bank Transaction"] = (
    df["From Bank"] != df["To Bank"]
).astype(int)

print("Cross-bank transaction feature created successfully!")

# =========================
# Transaction Time Category
# =========================

print("\nCreating transaction time category...")

df["Transaction Time Category"] = pd.cut(
    df["Hour"],
    bins=[-1, 5, 11, 17, 23],
    labels=[0, 1, 2, 3]
).astype(int)

print("Transaction time category created successfully!")
# =========================
# Weekend Transaction Feature
# =========================

print("\nCreating weekend transaction feature...")

df["Is Weekend"] = (
    df["DayOfWeek"] >= 5
).astype(int)

print("Weekend transaction feature created successfully!")

# =========================
# Log Amount Features
# =========================

print("\nCreating log amount features...")

df["Log Amount Received"] = np.log1p(df["Amount Received"])
df["Log Amount Paid"] = np.log1p(df["Amount Paid"])

print("Log amount features created successfully!")

# =========================
# Currency Relationship Feature
# =========================

print("\nCreating currency match feature...")

df["Same Currency"] = (
    df["Receiving Currency"] ==
    df["Payment Currency"]
).astype(int)

print("Currency match feature created successfully!")

# =========================
# Feature Engineering Summary
# =========================

print("\nFeature engineering summary...")

new_features = [
    "Amount Difference",
    "Amount Ratio",
    "Same Bank Transaction",
    "Cross Bank Transaction",
    "Transaction Time Category",
    "Is Weekend",
    "Log Amount Received",
    "Log Amount Paid",
    "Same Currency"
]

print("\nNew features created:")
print(new_features)

print("\nDataset shape after feature engineering:")
print(df.shape)

print("\nSample of new features:")
print(df[new_features].head())

# =========================
# Validate Engineered Features
# =========================

print("\nValidating engineered features...")

print("\nMissing values in new features:")
print(df[new_features].isnull().sum())

print("\nInfinite values in new features:")
print(np.isinf(df[new_features].select_dtypes(include=np.number)).sum())

print("\nFeature validation completed successfully!")

# =========================
# Feature Distribution Check
# =========================

print("\nFeature distributions:")

print("\nSame Bank Transaction:")
print(df["Same Bank Transaction"].value_counts())

print("\nTransaction Time Category:")
print(df["Transaction Time Category"].value_counts().sort_index())

print("\nIs Weekend:")
print(df["Is Weekend"].value_counts())

print("\nSame Currency:")
print(df["Same Currency"].value_counts())

# =========================
# Save Feature-Engineered Dataset
# =========================

output_path = "data/processed/feature_engineered_transactions.csv"

print("\nSaving feature-engineered dataset...")

df.to_csv(output_path, index=False)

print("Feature-engineered dataset saved successfully!")
print("Saved to:", output_path)