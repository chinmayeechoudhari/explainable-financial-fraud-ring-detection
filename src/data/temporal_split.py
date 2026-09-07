import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "data/processed/temporal_transactions.csv"
OUTPUT_DIR = "data/processed/splits"

TRAIN_END = pd.Timestamp("2022-09-09 00:00:00")
VALIDATION_END = pd.Timestamp("2022-09-10 00:00:00")
TEST_END = pd.Timestamp("2022-09-11 00:00:00")


# ============================================================
# LOAD DATA
# ============================================================

print("Loading feature-engineered transaction dataset...")

df = pd.read_csv(INPUT_PATH)

print(f"Dataset loaded successfully!")
print(f"Original shape: {df.shape}")


# ============================================================
# TIMESTAMP PROCESSING
# ============================================================

print("\nProcessing timestamps...")

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="mixed",
    errors="coerce"
)

invalid_timestamps = df["Timestamp"].isna().sum()

if invalid_timestamps > 0:
    raise ValueError(
        f"Found {invalid_timestamps} invalid timestamps."
    )

print("Timestamp validation: PASSED")


# ============================================================
# CHRONOLOGICAL SORTING
# ============================================================

print("\nSorting transactions chronologically...")

df = df.sort_values(
    "Timestamp",
    kind="mergesort"
).reset_index(drop=True)

if not df["Timestamp"].is_monotonic_increasing:
    raise ValueError(
        "Transactions are not chronologically ordered."
    )

print("Chronological ordering: PASSED")


# ============================================================
# TEMPORAL SPLIT
# ============================================================

print("\nCreating chronological splits...")

train = df[df["Timestamp"] < TRAIN_END].copy()

validation = df[
    (df["Timestamp"] >= TRAIN_END) &
    (df["Timestamp"] < VALIDATION_END)
].copy()

test = df[
    (df["Timestamp"] >= VALIDATION_END) &
    (df["Timestamp"] < TEST_END)
].copy()

future = df[
    df["Timestamp"] >= TEST_END
].copy()


# ============================================================
# SPLIT SUMMARY
# ============================================================

def print_split_summary(name, data):
    positives = int(data["Is Laundering"].sum())
    total = len(data)

    rate = (
        positives / total * 100
        if total > 0
        else 0
    )

    print(
        f"\n{name}"
        f"\n  Transactions: {total:,}"
        f"\n  Positive transactions: {positives:,}"
        f"\n  Positive rate: {rate:.4f}%"
        f"\n  Earliest: {data['Timestamp'].min()}"
        f"\n  Latest: {data['Timestamp'].max()}"
    )


print_split_summary("TRAIN", train)
print_split_summary("VALIDATION", validation)
print_split_summary("TEST", test)
print_split_summary("FUTURE GENERALIZATION", future)


# ============================================================
# VALIDATE SPLIT ORDER
# ============================================================

print("\nValidating temporal separation...")

if len(train) == 0:
    raise ValueError("Training split is empty.")

if len(validation) == 0:
    raise ValueError("Validation split is empty.")

if len(test) == 0:
    raise ValueError("Test split is empty.")

if len(future) == 0:
    raise ValueError("Future split is empty.")


if train["Timestamp"].max() >= validation["Timestamp"].min():
    raise ValueError(
        "Temporal overlap detected between train and validation."
    )

if validation["Timestamp"].max() >= test["Timestamp"].min():
    raise ValueError(
        "Temporal overlap detected between validation and test."
    )

if test["Timestamp"].max() >= future["Timestamp"].min():
    raise ValueError(
        "Temporal overlap detected between test and future."
    )

print("Temporal separation: PASSED")


# ============================================================
# VALIDATE TOTAL ROW COUNT
# ============================================================

print("\nValidating row preservation...")

total_split_rows = (
    len(train)
    + len(validation)
    + len(test)
    + len(future)
)

if total_split_rows != len(df):
    raise ValueError(
        "Split row counts do not match original dataset."
    )

print("Row preservation: PASSED")
print(f"Original rows: {len(df):,}")
print(f"Split rows:    {total_split_rows:,}")


# ============================================================
# VALIDATE POSITIVE LABEL PRESERVATION
# ============================================================

print("\nValidating target preservation...")

original_positive_count = int(
    df["Is Laundering"].sum()
)

split_positive_count = (
    int(train["Is Laundering"].sum())
    + int(validation["Is Laundering"].sum())
    + int(test["Is Laundering"].sum())
    + int(future["Is Laundering"].sum())
)

if split_positive_count != original_positive_count:
    raise ValueError(
        "Positive label counts do not match original dataset."
    )

print("Target preservation: PASSED")
print(f"Original positives: {original_positive_count:,}")
print(f"Split positives:    {split_positive_count:,}")


# ============================================================
# SAVE SPLITS
# ============================================================

print("\nSaving temporal splits...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_path = os.path.join(
    OUTPUT_DIR,
    "train.csv"
)

validation_path = os.path.join(
    OUTPUT_DIR,
    "validation.csv"
)

test_path = os.path.join(
    OUTPUT_DIR,
    "test.csv"
)

future_path = os.path.join(
    OUTPUT_DIR,
    "future.csv"
)

train.to_csv(train_path, index=False)
validation.to_csv(validation_path, index=False)
test.to_csv(test_path, index=False)
future.to_csv(future_path, index=False)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\nTemporal split completed successfully!")

print(f"Train:       {train_path}")
print(f"Validation:  {validation_path}")
print(f"Test:        {test_path}")
print(f"Future:      {future_path}")