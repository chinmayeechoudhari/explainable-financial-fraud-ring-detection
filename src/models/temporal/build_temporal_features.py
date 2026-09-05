from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.temporal.temporal_state import TemporalStateStore


REQUIRED_COLUMNS = [
    "Timestamp",
    "From Account",
    "To Account",
    "Amount Received",
    "Amount Paid",
    "Is Laundering",
]


def build_temporal_features(
    input_path: str,
    max_rows: int | None = None,
) -> pd.DataFrame:

    path = Path(input_path)

    df = pd.read_csv(
        path,
        nrows=max_rows,
        low_memory=False,
    )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
        format="mixed",
    )

    if df["Timestamp"].isna().any():
        raise ValueError("Invalid or missing timestamps detected.")

    df = df.sort_values(
        ["Timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)

    store = TemporalStateStore()

    output_rows = []

    column_index = {
        column: df.columns.get_loc(column)
        for column in REQUIRED_COLUMNS
    }

    # Process one exact timestamp at a time.
    for timestamp, group in df.groupby("Timestamp", sort=True):

        timestamp_seconds = int(timestamp.timestamp())

        current_rows = []

        # ---------------------------------------------------------------
        # STEP 1:
        # Read account states BEFORE this timestamp.
        # ---------------------------------------------------------------
        for values in group.itertuples(index=False, name=None):

            sender = str(values[column_index["From Account"]])
            receiver = str(values[column_index["To Account"]])

            amount_received = float(
                values[column_index["Amount Received"]]
            )

            amount_paid = float(
                values[column_index["Amount Paid"]]
            )

            label = int(
                values[column_index["Is Laundering"]]
            )

            sender_state = store.get(
                sender
            ).as_vector(timestamp_seconds)

            receiver_state = store.get(
                receiver
            ).as_vector(timestamp_seconds)

            current_rows.append(
                {
                    "Timestamp": timestamp,
                    "From Account": sender,
                    "To Account": receiver,

                    "sender_in_count": sender_state[0],
                    "sender_out_count": sender_state[1],
                    "sender_total_count": sender_state[2],
                    "sender_in_amount": sender_state[3],
                    "sender_out_amount": sender_state[4],
                    "sender_avg_in_amount": sender_state[5],
                    "sender_avg_out_amount": sender_state[6],
                    "sender_time_since_last": sender_state[7],

                    "receiver_in_count": receiver_state[0],
                    "receiver_out_count": receiver_state[1],
                    "receiver_total_count": receiver_state[2],
                    "receiver_in_amount": receiver_state[3],
                    "receiver_out_amount": receiver_state[4],
                    "receiver_avg_in_amount": receiver_state[5],
                    "receiver_avg_out_amount": receiver_state[6],
                    "receiver_time_since_last": receiver_state[7],

                    "Amount Received": amount_received,
                    "Amount Paid": amount_paid,

                    "Is Laundering": label,
                }
            )

        # ---------------------------------------------------------------
        # STEP 2:
        # Store features for the whole timestamp group.
        #
        # IMPORTANT:
        # No account state has been updated yet.
        # ---------------------------------------------------------------
        output_rows.extend(current_rows)

        # ---------------------------------------------------------------
        # STEP 3:
        # NOW update account states using this timestamp.
        # ---------------------------------------------------------------
        for values in group.itertuples(index=False, name=None):

            sender = str(values[column_index["From Account"]])
            receiver = str(values[column_index["To Account"]])

            amount_paid = float(
                values[column_index["Amount Paid"]]
            )

            amount_received = float(
                values[column_index["Amount Received"]]
            )

            store.update_transaction(
                sender=sender,
                receiver=receiver,
                amount_paid=amount_paid,
                amount_received=amount_received,
                timestamp=timestamp_seconds,
            )

    return pd.DataFrame(output_rows)


def validate_output(result: pd.DataFrame) -> None:

    if result.empty:
        raise AssertionError(
            "Temporal feature output is empty."
        )

    feature_columns = [
        column
        for column in result.columns
        if column.startswith("sender_")
        or column.startswith("receiver_")
    ]

    if len(feature_columns) != 16:
        raise AssertionError(
            f"Expected 16 temporal state features, "
            f"found {len(feature_columns)}."
        )

    numeric = result[
        feature_columns
    ].to_numpy(dtype=np.float64)

    if np.isnan(numeric).any():
        raise AssertionError(
            "Temporal features contain NaN values."
        )

    if np.isinf(numeric).any():
        raise AssertionError(
            "Temporal features contain infinite values."
        )

    if not result["Timestamp"].is_monotonic_increasing:
        raise AssertionError(
            "Output timestamps are not chronological."
        )

    if "Is Laundering" not in result.columns:
        raise AssertionError(
            "Target column is missing."
        )

    if not set(
        result["Is Laundering"].unique()
    ).issubset({0, 1}):
        raise AssertionError(
            "Target contains values other than 0/1."
        )

    print(f"Rows processed: {len(result):,}")
    print(
        f"Temporal feature count: "
        f"{len(feature_columns)}"
    )
    print(
        f"Unique timestamps: "
        f"{result['Timestamp'].nunique():,}"
    )

    unique_accounts = (
        set(result["From Account"])
        | set(result["To Account"])
    )

    print(
        f"Unique accounts: "
        f"{len(unique_accounts):,}"
    )

    print("NaN check: PASSED")
    print("Inf check: PASSED")
    print("Chronological ordering: PASSED")
    print(
        "Target excluded from state construction: PASSED"
    )
    print(
        "TEMPORAL FEATURE BUILDER SMOKE TEST: PASSED"
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/processed/temporal_transactions.csv",
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=10000,
        help=(
            "Number of chronological rows "
            "to process for smoke testing."
        ),
    )

    args = parser.parse_args()

    print("Loading temporal transaction data...")
    print(f"Input: {args.input}")
    print(
        f"Smoke-test rows: "
        f"{args.rows:,}"
    )

    result = build_temporal_features(
        input_path=args.input,
        max_rows=args.rows,
    )

    validate_output(result)


if __name__ == "__main__":
    main()