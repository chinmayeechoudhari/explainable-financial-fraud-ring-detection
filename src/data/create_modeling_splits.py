import os
import pandas as pd


INPUT_DIR = "data/processed/splits"
OUTPUT_DIR = "data/processed/modeling_splits"

IDENTITY_COLUMNS = [
    "Timestamp",
    "From Bank",
    "From Account",
    "To Bank",
    "To Account",
    "Amount Received",
    "Amount Paid",
    "Is Laundering",
]

SPLITS = ["train", "validation", "test", "future"]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("CREATING CLEAN MODELING SPLITS")
    print("=" * 70)

    total_before = 0
    total_after = 0
    total_removed = 0

    for split in SPLITS:
        input_path = os.path.join(INPUT_DIR, f"{split}.csv")
        output_path = os.path.join(OUTPUT_DIR, f"{split}.csv")

        print(f"\n[{split.upper()}]")

        df = pd.read_csv(input_path)

        before = len(df)

        # Remove only completely identical transaction records
        duplicate_mask = df.duplicated(
            subset=IDENTITY_COLUMNS,
            keep="first"
        )

        duplicate_count = duplicate_mask.sum()

        df_clean = df.loc[~duplicate_mask].copy()

        after = len(df_clean)

        # Safety check
        remaining_duplicates = df_clean.duplicated(
            subset=IDENTITY_COLUMNS,
            keep=False
        ).sum()

        if remaining_duplicates != 0:
            raise RuntimeError(
                f"{split}: duplicates still remain after deduplication."
            )

        # Preserve chronological order
        timestamp = pd.to_datetime(
            df_clean["Timestamp"],
            format="mixed",
            errors="coerce"
        )

        if timestamp.isna().any():
            raise RuntimeError(
                f"{split}: invalid timestamps found after cleanup."
            )

        df_clean = (
            df_clean.assign(_parsed_timestamp=timestamp)
            .sort_values("_parsed_timestamp", kind="mergesort")
            .drop(columns="_parsed_timestamp")
            .reset_index(drop=True)
        )

        df_clean.to_csv(output_path, index=False)

        print(f"Rows before:              {before:,}")
        print(f"Duplicate rows removed:   {duplicate_count:,}")
        print(f"Rows after:               {after:,}")
        print(f"Remaining duplicates:     {remaining_duplicates:,}")
        print(f"Saved to:                 {output_path}")

        total_before += before
        total_after += after
        total_removed += duplicate_count

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"Total rows before:         {total_before:,}")
    print(f"Total duplicate rows:      {total_removed:,}")
    print(f"Total rows after:          {total_after:,}")

    print("\nExpected:")
    print("  Train duplicate groups:       5")
    print("  Validation duplicate groups:  1")
    print("  Test duplicate groups:        0")
    print("  Future duplicate groups:      0")
    print("  Test removed:         0")
    print("  Future removed:       0")

    if total_removed != 6:
        raise RuntimeError(
            f"Expected 6 duplicate rows to be removed, "
            f"but found {total_removed}."
        )

    print("\nSTATUS: PASSED")
    print("Clean modeling splits created successfully.")


if __name__ == "__main__":
    main()