import argparse
import csv
import json
import os

import numpy as np
import pandas as pd

from src.models.temporal.temporal_state import TemporalStateStore


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

CURRENT_TRANSACTION_FEATURES = [
    "Amount Received",
    "Amount Paid",
    "Amount Difference",
    "Amount Ratio",
    "Same Bank Transaction",
    "Cross Bank Transaction",
    "Transaction Time Category",
    "Is Weekend",
    "Log Amount Received",
    "Log Amount Paid",
    "Same Currency",
]


TEMPORAL_FEATURE_COLUMNS = [
    "sender_in_count",
    "sender_out_count",
    "sender_total_count",
    "sender_in_amount",
    "sender_out_amount",
    "sender_avg_in_amount",
    "sender_avg_out_amount",
    "sender_time_since_last",
    "receiver_in_count",
    "receiver_out_count",
    "receiver_total_count",
    "receiver_in_amount",
    "receiver_out_amount",
    "receiver_avg_in_amount",
    "receiver_avg_out_amount",
    "receiver_time_since_last",
]


INPUT_COLUMNS = [
    "Timestamp",
    "From Account",
    "To Account",
    *CURRENT_TRANSACTION_FEATURES,
    "Is Laundering",
]


OUTPUT_COLUMNS = [
    "From Account",
    "To Account",
    "Timestamp",
    *CURRENT_TRANSACTION_FEATURES,
    *TEMPORAL_FEATURE_COLUMNS,
    "Is Laundering",
]


CHUNK_SIZE = 100_000


# Official duplicate-cleaned modeling split sizes.
OFFICIAL_SPLIT_ROWS = {
    "train": 5_749_364,
    "validation": 891_571,
    "test": 282_877,
    "future": 223,
}


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value):
    """
    Convert a value to finite float.

    Any NaN or infinity is replaced by 0.0.
    """

    value = float(value)

    if not np.isfinite(value):
        return 0.0

    return value


def account_vector(state, timestamp):
    """
    Convert an AccountState into the 8 temporal features
    for one account at the current timestamp.

    The state represents information strictly before
    the current timestamp.
    """

    return [
        safe_float(state.in_count),
        safe_float(state.out_count),
        safe_float(state.total_count),
        safe_float(state.in_amount),
        safe_float(state.out_amount),
        safe_float(state.avg_in_amount),
        safe_float(state.avg_out_amount),
        safe_float(
            state.time_since_last(timestamp)
        ),
    ]


# ============================================================
# FEATURE CONSTRUCTION
# ============================================================

def build_feature_row(
    row,
    state_store,
    timestamp,
):
    """
    Build one transaction-level temporal feature row.

    IMPORTANT:
    The account states are read BEFORE the current
    timestamp is added to the state store.

    Therefore, the generated temporal features only
    contain information from timestamps strictly earlier
    than the current timestamp.
    """

    sender = str(row["From Account"])
    receiver = str(row["To Account"])

    sender_state = state_store.get(sender)
    receiver_state = state_store.get(receiver)

    sender_features = account_vector(
        sender_state,
        timestamp,
    )

    receiver_features = account_vector(
        receiver_state,
        timestamp,
    )

    current_features = [
        safe_float(row[column])
        for column in CURRENT_TRANSACTION_FEATURES
    ]

    return [
        sender,
        receiver,
        row["Timestamp"],
        *current_features,
        *sender_features,
        *receiver_features,
        int(row["Is Laundering"]),
    ]


def update_state(
    row,
    state_store,
    timestamp,
):
    """
    Update the temporal state using one transaction.

    The target column is NOT used here.
    """

    sender = str(row["From Account"])
    receiver = str(row["To Account"])

    state_store.update_transaction(
        sender=sender,
        receiver=receiver,
        amount_paid=safe_float(
            row["Amount Paid"]
        ),
        amount_received=safe_float(
            row["Amount Received"]
        ),
        timestamp=timestamp,
    )


# ============================================================
# TIMESTAMP GROUP PROCESSING
# ============================================================

def process_timestamp_group(
    group,
    state_store,
    writer,
):
    """
    Process one complete timestamp group.

    Temporal rule:

        Features at T
              ↓
        use state from < T
              ↓
        write all features
              ↓
        update state using T

    Therefore transactions occurring at exactly the same
    timestamp cannot influence one another's temporal
    features.
    """

    if group.empty:
        return 0

    timestamp = int(
        group.iloc[0]["_timestamp"]
    )

    output_rows = []

    # --------------------------------------------------------
    # STEP 1
    # Calculate features for ALL transactions at T
    # using the pre-T state.
    # --------------------------------------------------------

    for row in group.to_dict("records"):

        output_rows.append(
            build_feature_row(
                row=row,
                state_store=state_store,
                timestamp=timestamp,
            )
        )

    # --------------------------------------------------------
    # STEP 2
    # Write all rows.
    # --------------------------------------------------------

    for output_row in output_rows:
        writer.writerow(output_row)

    # --------------------------------------------------------
    # STEP 3
    # Only now update the temporal state.
    # --------------------------------------------------------

    for row in group.to_dict("records"):

        update_state(
            row=row,
            state_store=state_store,
            timestamp=timestamp,
        )

    return len(output_rows)


# ============================================================
# SPLIT PROCESSING
# ============================================================

def process_split(
    input_path,
    output_path,
    state_store,
    chunk_size=CHUNK_SIZE,
):
    """
    Stream one modeling split.

    The input is read in chunks so that the complete
    5.7M-row training file does not need to be loaded
    into memory.

    Timestamp groups are preserved across chunk boundaries.
    """

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    total_rows = 0
    timestamp_groups = 0

    pending_group = None
    pending_timestamp = None

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.writer(
            output_file
        )

        writer.writerow(
            OUTPUT_COLUMNS
        )

        # ----------------------------------------------------
        # Read input in chunks.
        # ----------------------------------------------------

        for chunk in pd.read_csv(
            input_path,
            usecols=INPUT_COLUMNS,
            chunksize=chunk_size,
            low_memory=False,
        ):

            # ------------------------------------------------
            # Parse timestamps.
            # ------------------------------------------------

            chunk["Timestamp"] = pd.to_datetime(
                chunk["Timestamp"],
                format="mixed",
                errors="coerce",
            )

            if chunk["Timestamp"].isna().any():
                raise ValueError(
                    f"Invalid timestamp detected in "
                    f"{input_path}"
                )

            # ------------------------------------------------
            # Sort within the chunk.
            # ------------------------------------------------

            chunk = chunk.sort_values(
                "Timestamp",
                kind="stable",
            )

            # ------------------------------------------------
            # Internal Unix timestamp.
            #
            # This is only used for temporal state calculations.
            # The original Timestamp is preserved in output.
            # ------------------------------------------------

            chunk["_timestamp"] = (
                chunk["Timestamp"].astype("int64")
                // 10**9
            )

            # ------------------------------------------------
            # Group by exact timestamp.
            # ------------------------------------------------

            for timestamp, group in chunk.groupby(
                "_timestamp",
                sort=False,
            ):

                timestamp = int(timestamp)

                # --------------------------------------------
                # First timestamp group.
                # --------------------------------------------

                if pending_group is None:

                    pending_group = group.copy()
                    pending_timestamp = timestamp

                    continue

                # --------------------------------------------
                # Same timestamp continues into this chunk.
                # Keep accumulating it.
                # --------------------------------------------

                if timestamp == pending_timestamp:

                    pending_group = pd.concat(
                        [
                            pending_group,
                            group,
                        ],
                        ignore_index=True,
                    )

                    continue

                # --------------------------------------------
                # Timestamp changed.
                #
                # Therefore the previous timestamp group
                # is now complete and safe to process.
                # --------------------------------------------

                written = process_timestamp_group(
                    pending_group,
                    state_store,
                    writer,
                )

                total_rows += written
                timestamp_groups += 1

                pending_group = group.copy()
                pending_timestamp = timestamp

            print(
                f"  Processed rows: {total_rows:,} | "
                f"Timestamp groups: {timestamp_groups:,} | "
                f"Account states: {len(state_store):,}",
                end="\r",
                flush=True,
            )

        # ----------------------------------------------------
        # Process the final timestamp group.
        # ----------------------------------------------------

        if pending_group is not None:

            written = process_timestamp_group(
                pending_group,
                state_store,
                writer,
            )

            total_rows += written
            timestamp_groups += 1

    print()

    return (
        total_rows,
        timestamp_groups,
    )


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def validate_output(
    output_path,
    expected_rows,
    split_name,
):
    """
    Validate the generated temporal dataset.

    expected_rows is the actual number of rows processed
    for the split. This allows both:
        - official full datasets
        - small smoke-test datasets
    """

    print(
        f"\nVALIDATING {split_name.upper()}"
    )

    df = pd.read_csv(
        output_path,
        low_memory=False,
    )

    # --------------------------------------------------------
    # ROW COUNT
    # --------------------------------------------------------

    if len(df) != expected_rows:

        raise AssertionError(
            f"Output row count mismatch: "
            f"expected {expected_rows:,}, "
            f"got {len(df):,}"
        )

    # --------------------------------------------------------
    # EXPECTED FEATURE COUNT
    # --------------------------------------------------------

    expected_feature_count = (
        len(CURRENT_TRANSACTION_FEATURES)
        + len(TEMPORAL_FEATURE_COLUMNS)
    )

    expected_output_column_count = len(
        OUTPUT_COLUMNS
    )

    if len(df.columns) != expected_output_column_count:

        raise AssertionError(
            f"Column count mismatch: "
            f"expected "
            f"{expected_output_column_count}, "
            f"got {len(df.columns)}"
        )

    actual_feature_count = (
        len(df.columns) - 4
    )

    if actual_feature_count != expected_feature_count:

        raise AssertionError(
            f"Feature count mismatch: "
            f"expected {expected_feature_count}, "
            f"got {actual_feature_count}"
        )

    # --------------------------------------------------------
    # COLUMN CHECK
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in OUTPUT_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise AssertionError(
            f"Missing output columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # NUMERIC VALIDATION
    # --------------------------------------------------------

    numeric_columns = [
        column
        for column in df.columns
        if column not in [
            "From Account",
            "To Account",
            "Timestamp",
        ]
    ]

    numeric_values = df[
        numeric_columns
    ].to_numpy(
        dtype=np.float64
    )

    if np.isnan(
        numeric_values
    ).any():

        raise AssertionError(
            "NaN detected"
        )

    if np.isinf(
        numeric_values
    ).any():

        raise AssertionError(
            "Inf detected"
        )

    # --------------------------------------------------------
    # TARGET-DERIVED FEATURE CHECK
    # --------------------------------------------------------

    if "laundering_count" in df.columns:

        raise AssertionError(
            "Forbidden target-derived feature detected"
        )

    # --------------------------------------------------------
    # LABEL VALIDATION
    # --------------------------------------------------------

    labels = df[
        "Is Laundering"
    ]

    if labels.isna().any():

        raise AssertionError(
            "Missing labels detected"
        )

    if not labels.isin(
        [0, 1]
    ).all():

        raise AssertionError(
            "Labels must contain only 0/1"
        )

    # --------------------------------------------------------
    # TIMESTAMP VALIDATION
    # --------------------------------------------------------

    timestamps = pd.to_datetime(
        df["Timestamp"],
        format="mixed",
        errors="coerce",
    )

    if timestamps.isna().any():

        raise AssertionError(
            "Invalid output timestamp"
        )

    if not timestamps.is_monotonic_increasing:

        raise AssertionError(
            "Output is not chronologically ordered"
        )

    # --------------------------------------------------------
    # VALIDATION SUMMARY
    # --------------------------------------------------------

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Current transaction features: "
        f"{len(CURRENT_TRANSACTION_FEATURES)}"
    )

    print(
        f"Temporal state features: "
        f"{len(TEMPORAL_FEATURE_COLUMNS)}"
    )

    print(
        f"Total model features: "
        f"{expected_feature_count}"
    )

    print(
        f"Output columns: "
        f"{len(df.columns)}"
    )

    print(
        f"Positive labels: "
        f"{int(labels.sum()):,}"
    )

    print(
        f"Negative labels: "
        f"{int((labels == 0).sum()):,}"
    )

    print(
        "NaN check: PASSED"
    )

    print(
        "Inf check: PASSED"
    )

    print(
        "Chronological ordering: PASSED"
    )

    print(
        "Target-derived feature exclusion: PASSED"
    )

    print(
        "Label validation: PASSED"
    )

    return {
        "rows": len(df),
        "positive_labels": int(
            labels.sum()
        ),
        "negative_labels": int(
            (labels == 0).sum()
        ),
        "feature_count": expected_feature_count,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe temporal GNN datasets."
        )
    )

    parser.add_argument(
        "--input-dir",
        default="data/processed/modeling_splits",
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed/temporal_gnn",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
    )

    parser.add_argument(
        "--validate",
        action="store_true",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=[
            "train",
            "validation",
            "test",
            "future",
        ],
        choices=[
            "train",
            "validation",
            "test",
            "future",
        ],
    )

    args = parser.parse_args()

    print("=" * 70)
    print(
        "TEMPORAL GNN DATASET BUILDER"
    )
    print("=" * 70)

    print(
        "\nTemporal rule:"
    )

    print(
        "Features at timestamp T use only state "
        "from timestamps < T."
    )

    print(
        "All transactions at the same timestamp "
        "are processed before state updates."
    )

    print(
        f"\nChunk size: {args.chunk_size:,}"
    )

    print(
        f"Splits: {', '.join(args.splits)}"
    )

    # --------------------------------------------------------
    # One state store is intentionally maintained across:
    #
    # train → validation → test → future
    #
    # This represents online temporal inference.
    # --------------------------------------------------------

    state_store = TemporalStateStore()

    split_metadata = {}

    for split_name in args.splits:

        input_path = os.path.join(
            args.input_dir,
            f"{split_name}.csv",
        )

        output_path = os.path.join(
            args.output_dir,
            f"{split_name}.csv",
        )

        if not os.path.exists(
            input_path
        ):

            raise FileNotFoundError(
                f"Input file not found: "
                f"{input_path}"
            )

        print(
            "\n" + "-" * 70
        )

        print(
            f"PROCESSING {split_name.upper()}"
        )

        print(
            "-" * 70
        )

        rows_written, timestamp_groups = (
            process_split(
                input_path=input_path,
                output_path=output_path,
                state_store=state_store,
                chunk_size=args.chunk_size,
            )
        )

        # ----------------------------------------------------
        # Verify input/output row preservation.
        #
        # This works for both the 100k smoke test and the
        # official duplicate-cleaned modeling splits.
        # ----------------------------------------------------

        input_row_count = 0

        with open(
            input_path,
            "rb",
        ) as input_file:

            # Header is excluded.
            input_row_count = (
                sum(
                    1
                    for _ in input_file
                )
                - 1
            )

        if rows_written != input_row_count:

            raise AssertionError(
                f"{split_name}: input/output row mismatch. "
                f"Input rows: {input_row_count:,}, "
                f"Output rows: {rows_written:,}"
            )

        print(
            f"Rows written: "
            f"{rows_written:,}"
        )

        print(
            f"Timestamp groups: "
            f"{timestamp_groups:,}"
        )

        print(
            f"Account states: "
            f"{len(state_store):,}"
        )

        # ----------------------------------------------------
        # Check official split size when processing the
        # finalized modeling split.
        # ----------------------------------------------------

        if (
            input_row_count
            == OFFICIAL_SPLIT_ROWS[split_name]
        ):

            print(
                f"Official {split_name} row count: "
                f"{OFFICIAL_SPLIT_ROWS[split_name]:,} "
                f"- PASSED"
            )

        else:

            print(
                f"Custom/smoke split detected: "
                f"{input_row_count:,} rows"
            )

        # ----------------------------------------------------
        # Validate generated output.
        # ----------------------------------------------------

        if args.validate:

            metadata = validate_output(
                output_path=output_path,
                expected_rows=rows_written,
                split_name=split_name,
            )

            metadata[
                "timestamp_groups"
            ] = timestamp_groups

            split_metadata[
                split_name
            ] = metadata

    # ========================================================
    # SAVE METADATA
    # ========================================================

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    metadata_path = os.path.join(
        args.output_dir,
        "metadata.json",
    )

    metadata = {
        "dataset": (
            "IBM AML LI-Small_Trans"
        ),

        "temporal_rule": (
            "Features at timestamp T use only "
            "state from strictly earlier timestamps < T."
        ),

        "same_timestamp_rule": (
            "All rows at timestamp T receive features "
            "from the same pre-T state; state updates "
            "occur after the complete timestamp group."
        ),

        "target_column": (
            "Is Laundering"
        ),

        "target_used_for_features": False,

        "current_transaction_feature_count": len(
            CURRENT_TRANSACTION_FEATURES
        ),

        "temporal_state_feature_count": len(
            TEMPORAL_FEATURE_COLUMNS
        ),

        "total_model_feature_count": (
            len(CURRENT_TRANSACTION_FEATURES)
            + len(TEMPORAL_FEATURE_COLUMNS)
        ),

        "chunk_size": args.chunk_size,

        "processed_splits": args.splits,

        "splits": split_metadata,
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TEMPORAL GNN DATASET BUILD COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Output directory: "
        f"{args.output_dir}"
    )

    print(
        f"Final account states: "
        f"{len(state_store):,}"
    )

    print(
        f"Metadata: "
        f"{metadata_path}"
    )


if __name__ == "__main__":
    main()