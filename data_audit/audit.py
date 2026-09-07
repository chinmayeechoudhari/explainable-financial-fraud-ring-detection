import pandas as pd

file_path = "data/raw/LI-Small_Trans.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nLaundering Class Distribution:")
print(df["Is Laundering"].value_counts())

total = len(df)

print("\nClass Distribution Percentage:")
print((df["Is Laundering"].value_counts() / total * 100).round(4))

print("\nUnique Values Per Column:")
print(df.nunique())

print("\nReceiving Currency Distribution:")
print(df["Receiving Currency"].value_counts())

print("\nPayment Currency Distribution:")
print(df["Payment Currency"].value_counts())

print("\nPayment Format Distribution:")
print(df["Payment Format"].value_counts())

print("\nUnique From Banks:", df["From Bank"].nunique())
print("Unique To Banks:", df["To Bank"].nunique())


# Laundering Transaction Analysis

laundering_df = df[df["Is Laundering"] == 1]

print("\nTotal Laundering Transactions:")
print(len(laundering_df))

print("\nLaundering Payment Format Distribution:")
print(laundering_df["Payment Format"].value_counts())

print("\nLaundering Receiving Currency Distribution:")
print(laundering_df["Receiving Currency"].value_counts())

print("\nLaundering Amount Statistics:")
print(laundering_df["Amount Received"].describe())


# Duplicate Analysis

print("\nDuplicate Analysis:")

duplicate_count = df.duplicated().sum()

duplicate_percentage = (
    duplicate_count / len(df)
) * 100

print(f"Total Duplicate Rows: {duplicate_count:,}")
print(f"Duplicate Percentage: {duplicate_percentage:.6f}%")


# Create Audit Report

total_transactions = len(df)

laundering_transactions = df["Is Laundering"].sum()

legitimate_transactions = (
    total_transactions - laundering_transactions
)

laundering_percentage = (
    laundering_transactions / total_transactions
) * 100


report = f"""# Data Audit Report

## Dataset Overview

- Total Transactions: {total_transactions:,}
- Total Features: {df.shape[1]}
- Laundering Transactions: {laundering_transactions:,}
- Legitimate Transactions: {legitimate_transactions:,}
- Laundering Percentage: {laundering_percentage:.4f}%

## Dataset Quality

- Missing Values: 0
- Exact Duplicate Rows: {duplicate_count:,}
- Duplicate Percentage: {duplicate_percentage:.6f}%
- Decision: Exact duplicate rows will be removed during preprocessing.

## Key Observations

1. The dataset contains {total_transactions:,} financial transactions.
2. The dataset is highly imbalanced.
3. Only {laundering_percentage:.4f}% of transactions are labelled as laundering.
4. This class imbalance must be handled carefully during model training.
5. The dataset contains multiple transaction currencies and payment formats.
"""


with open(
    "data_audit/audit_report.md",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)


print("\nAudit report saved successfully!")