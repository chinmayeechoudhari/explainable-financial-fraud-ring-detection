import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SPLIT_DIR = "data/processed/splits"

TRAIN_PATH = os.path.join(SPLIT_DIR, "train.csv")
VALIDATION_PATH = os.path.join(SPLIT_DIR, "validation.csv")
TEST_PATH = os.path.join(SPLIT_DIR, "test.csv")
FUTURE_PATH = os.path.join(SPLIT_DIR, "future.csv")


# Columns that must never be used as model features
FORBIDDEN_COLUMNS = {
    "laundering_count",
}


# Columns required for every transaction split
REQUIRED_COLUMNS = {
    "Timestamp",
    "From Bank",
    "From Account",
    "To Bank",
    "To Account",
    "Is Laundering",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_split(path, name):
    print(f"\nLoading {name}...")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing split file: {path}"
        )

    df = pd.read_csv(path)

    print(f"{name} shape: {df.shape}")

    return df


def parse_timestamps(series):
    """
    Parse the dataset's mixed timestamp formats.

    The CSV contains both:
        YYYY-MM-DD
        YYYY-MM-DD HH:MM:SS

    format='mixed' allows pandas to correctly parse both.
    """
    return pd.to_datetime(
        series,
        format="mixed",
        errors="coerce"
    )


def validate_required_columns(df, name):
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            f"{sorted(missing)}"
        )

    print(f"{name} required columns: PASSED")


def validate_forbidden_columns(df, name):
    present = FORBIDDEN_COLUMNS.intersection(
        df.columns
    )

    if present:
        raise ValueError(
            f"{name} contains forbidden "
            f"target-derived columns: {sorted(present)}"
        )

    print(f"{name} forbidden-column check: PASSED")


def validate_timestamps(df, name):
    timestamps = parse_timestamps(
        df["Timestamp"]
    )

    invalid = int(timestamps.isna().sum())

    if invalid > 0:
        bad_values = df.loc[
            timestamps.isna(),
            "Timestamp"
        ].head(10).tolist()

        raise ValueError(
            f"{name} contains {invalid} invalid "
            f"timestamps. Examples: {bad_values}"
        )

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            f"{name} is not chronologically sorted."
        )

    print(f"{name} timestamp validation: PASSED")


def validate_labels(df, name):
    unique_labels = set(
        df["Is Laundering"].dropna().unique()
    )

    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"{name} contains unexpected labels: "
            f"{unique_labels}"
        )

    missing_labels = int(
        df["Is Laundering"].isna().sum()
    )

    if missing_labels > 0:
        raise ValueError(
            f"{name} contains {missing_labels} "
            f"missing labels."
        )

    print(f"{name} label validation: PASSED")


def validate_numeric_values(df, name):
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    nan_count = int(
        df[numeric_columns].isna().sum().sum()
    )

    inf_count = int(
        np.isinf(
            df[numeric_columns].to_numpy()
        ).sum()
    )

    if nan_count > 0:
        raise ValueError(
            f"{name} contains {nan_count} NaN values "
            f"in numeric columns."
        )

    if inf_count > 0:
        raise ValueError(
            f"{name} contains {inf_count} infinite "
            f"values in numeric columns."
        )

    print(
        f"{name} numeric NaN/Inf check: PASSED"
    )


# ============================================================
# LOAD ALL SPLITS
# ============================================================

train = load_split(
    TRAIN_PATH,
    "TRAIN"
)

validation = load_split(
    VALIDATION_PATH,
    "VALIDATION"
)

test = load_split(
    TEST_PATH,
    "TEST"
)

future = load_split(
    FUTURE_PATH,
    "FUTURE"
)


splits = {
    "TRAIN": train,
    "VALIDATION": validation,
    "TEST": test,
    "FUTURE": future,
}


# ============================================================
# INDIVIDUAL SPLIT VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("INDIVIDUAL SPLIT VALIDATION")
print("=" * 60)

for name, df in splits.items():

    validate_required_columns(
        df,
        name
    )

    validate_forbidden_columns(
        df,
        name
    )

    validate_timestamps(
        df,
        name
    )

    validate_labels(
        df,
        name
    )

    validate_numeric_values(
        df,
        name
    )


# ============================================================
# TEMPORAL BOUNDARY VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("TEMPORAL BOUNDARY VALIDATION")
print("=" * 60)

train_timestamps = parse_timestamps(
    train["Timestamp"]
)

validation_timestamps = parse_timestamps(
    validation["Timestamp"]
)

test_timestamps = parse_timestamps(
    test["Timestamp"]
)

future_timestamps = parse_timestamps(
    future["Timestamp"]
)


train_max = train_timestamps.max()
validation_min = validation_timestamps.min()

validation_max = validation_timestamps.max()
test_min = test_timestamps.min()

test_max = test_timestamps.max()
future_min = future_timestamps.min()


print(f"TRAIN latest:       {train_max}")
print(f"VALIDATION earliest: {validation_min}")
print(f"VALIDATION latest:   {validation_max}")
print(f"TEST earliest:       {test_min}")
print(f"TEST latest:         {test_max}")
print(f"FUTURE earliest:     {future_min}")


if train_max >= validation_min:
    raise ValueError(
        "Temporal overlap detected between "
        "TRAIN and VALIDATION."
    )


if validation_max >= test_min:
    raise ValueError(
        "Temporal overlap detected between "
        "VALIDATION and TEST."
    )


if test_max >= future_min:
    raise ValueError(
        "Temporal overlap detected between "
        "TEST and FUTURE."
    )


print(
    "\nTrain < Validation < Test < Future: PASSED"
)


# ============================================================
# EXACT TIMESTAMP OVERLAP CHECK
# ============================================================

print("\nChecking exact timestamp overlap...")

train_times = set(
    train_timestamps.unique()
)

validation_times = set(
    validation_timestamps.unique()
)

test_times = set(
    test_timestamps.unique()
)

future_times = set(
    future_timestamps.unique()
)


overlap_train_validation = (
    train_times & validation_times
)

overlap_validation_test = (
    validation_times & test_times
)

overlap_test_future = (
    test_times & future_times
)


if overlap_train_validation:
    raise ValueError(
        "Exact timestamps overlap between "
        "TRAIN and VALIDATION."
    )


if overlap_validation_test:
    raise ValueError(
        "Exact timestamps overlap between "
        "VALIDATION and TEST."
    )


if overlap_test_future:
    raise ValueError(
        "Exact timestamps overlap between "
        "TEST and FUTURE."
    )


print(
    "Exact timestamp overlap: PASSED"
)


# ============================================================
# DUPLICATE TRANSACTION CHECK
# ============================================================

print("\nChecking duplicate transaction rows...")

identity_columns = [
    "Timestamp",
    "From Bank",
    "From Account",
    "To Bank",
    "To Account",
    "Amount Received",
    "Amount Paid",
    "Is Laundering",
]


for name, df in splits.items():

    duplicate_count = int(
        df.duplicated(
            subset=identity_columns
        ).sum()
    )

    print(
        f"{name} duplicate transaction rows: "
        f"{duplicate_count:,}"
    )

    if duplicate_count > 0:
        print(
            f"WARNING: {name} contains duplicate "
            f"transaction identity rows."
        )


# ============================================================
# ACCOUNT OVERLAP ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("ACCOUNT OVERLAP ANALYSIS")
print("=" * 60)


def get_accounts(df):
    return set(
        df["From Account"]
    ).union(
        set(df["To Account"])
    )


train_accounts = get_accounts(train)
validation_accounts = get_accounts(validation)
test_accounts = get_accounts(test)
future_accounts = get_accounts(future)


print(
    f"Unique TRAIN accounts: "
    f"{len(train_accounts):,}"
)

print(
    f"Unique VALIDATION accounts: "
    f"{len(validation_accounts):,}"
)

print(
    f"Unique TEST accounts: "
    f"{len(test_accounts):,}"
)

print(
    f"Unique FUTURE accounts: "
    f"{len(future_accounts):,}"
)


train_validation_overlap = (
    train_accounts &
    validation_accounts
)

train_test_overlap = (
    train_accounts &
    test_accounts
)

train_future_overlap = (
    train_accounts &
    future_accounts
)


print(
    f"\nTRAIN <-> VALIDATION account overlap: "
    f"{len(train_validation_overlap):,}"
)

print(
    f"TRAIN <-> TEST account overlap: "
    f"{len(train_test_overlap):,}"
)

print(
    f"TRAIN <-> FUTURE account overlap: "
    f"{len(train_future_overlap):,}"
)


print(
    "\nAccount overlap is reported, "
    "not treated as leakage."
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)


for name, df in splits.items():

    positives = int(
        df["Is Laundering"].sum()
    )

    total = len(df)

    rate = (
        positives / total * 100
        if total > 0
        else 0
    )

    print(
        f"{name}: "
        f"{positives:,} positives / "
        f"{total:,} transactions "
        f"({rate:.4f}%)"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("LEAKAGE VALIDATION COMPLETE")
print("=" * 60)

print(
    "Target-derived forbidden columns: PASSED"
)

print(
    "Timestamp validity: PASSED"
)

print(
    "Chronological ordering: PASSED"
)

print(
    "Temporal split boundaries: PASSED"
)

print(
    "Exact timestamp separation: PASSED"
)

print(
    "Numeric NaN/Inf validation: PASSED"
)

print(
    "Label validation: PASSED"
)

print(
    "Account overlap analysis: COMPLETED"
)

print(
    "\nNo direct target-derived feature leakage "
    "was detected in the split files."
)

print(
    "IMPORTANT: Temporal neighborhood leakage "
    "will be validated separately during TGAT "
    "construction."
)