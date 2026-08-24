import pandas as pd

# =========================
# Load Raw Dataset
# =========================

file_path = "data/raw/LI-Small_Trans.csv"

print("Loading dataset...")

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)


# =========================
# Remove Exact Duplicates
# =========================

duplicate_count = df.duplicated().sum()

print("\nDuplicate rows found:", duplicate_count)

df = df.drop_duplicates()

print("Duplicates removed successfully!")
print("Shape after removing duplicates:", df.shape)


# =========================
# Process Timestamp
# =========================

print("\nProcessing timestamp...")

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%Y/%m/%d %H:%M"
)

# Extract useful time-based features
df["Year"] = df["Timestamp"].dt.year
df["Month"] = df["Timestamp"].dt.month
df["Day"] = df["Timestamp"].dt.day
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek

print("Timestamp processed successfully!")

print("\nSample timestamp features:")
print(
    df[
        ["Timestamp", "Year", "Month", "Day", "Hour", "DayOfWeek"]
    ].head()
)


# =========================
# Inspect Categorical Columns
# =========================

print("\nCategorical column unique counts:")

categorical_columns = [
    "Receiving Currency",
    "Payment Currency",
    "Payment Format"
]

for column in categorical_columns:
    print(f"{column}: {df[column].nunique()} unique values")


# =========================
# Encode Categorical Columns
# =========================

print("\nEncoding categorical columns...")

for column in categorical_columns:
    df[column] = df[column].astype("category").cat.codes

print("Categorical columns encoded successfully!")

print("\nEncoded sample:")
print(
    df[
        [
            "Receiving Currency",
            "Payment Currency",
            "Payment Format"
        ]
    ].head()
)


# =========================
# Rename Account Columns
# =========================

print("\nRenaming account columns...")

df = df.rename(
    columns={
        "Account": "From Account",
        "Account.1": "To Account"
    }
)

print("Account columns renamed successfully!")

print("\nCurrent columns:")
print(df.columns.tolist())
# =========================
# Numerical Data Validation
# =========================

print("\nChecking numerical columns...")

numerical_columns = [
    "Amount Received",
    "Amount Paid"
]

for column in numerical_columns:
    print(f"\n{column} statistics:")
    print(df[column].describe())

    negative_values = (df[column] < 0).sum()
    zero_values = (df[column] == 0).sum()

    print(f"Negative values: {negative_values}")
    print(f"Zero values: {zero_values}")

    # =========================
# Account ID Validation
# =========================

print("\nChecking account columns...")

account_columns = [
    "From Account",
    "To Account"
]

for column in account_columns:
    missing_values = df[column].isnull().sum()
    unique_accounts = df[column].nunique()

    print(f"\n{column}:")
    print(f"Missing values: {missing_values}")
    print(f"Unique accounts: {unique_accounts}")

print("\nAccount ID validation completed successfully!")

# =========================
# Target Column Validation
# =========================

print("\nValidating target column...")

print("\nUnique values in Is Laundering:")
print(df["Is Laundering"].unique())

invalid_target_values = (
    ~df["Is Laundering"].isin([0, 1])
).sum()

print(f"\nInvalid target values: {invalid_target_values}")

print("\nTarget class distribution:")
print(df["Is Laundering"].value_counts())

print("\nTarget class distribution percentage:")
print(
    (
        df["Is Laundering"]
        .value_counts(normalize=True)
        * 100
    ).round(4)
)

print("\nTarget column validation completed successfully!")

# =========================
# Final Dataset Validation
# =========================

print("\nFinal dataset validation...")

print("\nFinal shape:", df.shape)

print("\nTotal missing values:")
print(df.isnull().sum().sum())

print("\nRemaining duplicate rows:")
print(df.duplicated().sum())

print("\nFinal columns:")
print(df.columns.tolist())

print("\nFinal dataset validation completed successfully!")

# =========================
# Save Processed Dataset
# =========================

output_path = "data/processed/processed_transactions.csv"

print("\nSaving processed dataset...")

df.to_csv(
    output_path,
    index=False
)

print("Processed dataset saved successfully!")
print(f"Saved to: {output_path}")