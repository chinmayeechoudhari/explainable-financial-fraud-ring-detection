import os
import pandas as pd

print("Loading feature-engineered dataset...")

input_path = "data/processed/feature_engineered_transactions.csv"
df = pd.read_csv(input_path)

print("Dataset loaded successfully!")
print(f"Original dataset shape: {df.shape}")

# Convert Timestamp to datetime
print("\nConverting timestamps...")

df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

invalid_timestamps = df["Timestamp"].isna().sum()

print(f"Invalid timestamps: {invalid_timestamps}")

if invalid_timestamps > 0:
    raise ValueError(f"Found {invalid_timestamps} invalid timestamps.")

print("Timestamp conversion completed successfully!")

# Sort transactions chronologically
print("\nSorting transactions chronologically...")

df = df.sort_values("Timestamp").reset_index(drop=True)

print("Transactions sorted successfully!")
print(f"Oldest transaction: {df['Timestamp'].min()}")
print(f"Newest transaction: {df['Timestamp'].max()}")

# Create numerical temporal feature
print("\nCreating numerical temporal feature...")

reference_time = df["Timestamp"].min()

df["Temporal Seconds"] = (
    df["Timestamp"] - reference_time
).dt.total_seconds()

print("Temporal feature created successfully!")

# Validate chronological ordering
print("\nValidating chronological order...")

if df["Timestamp"].is_monotonic_increasing:
    print("Chronological ordering: PASSED")
else:
    raise ValueError("Chronological ordering validation FAILED.")

# Validate temporal values
print("\nValidating temporal values...")

if (df["Temporal Seconds"] < 0).sum() == 0:
    print("Temporal value validation: PASSED")
else:
    raise ValueError("Negative temporal values found.")

# Save output
print("\nSaving temporal transaction dataset...")

os.makedirs("data/processed", exist_ok=True)

output_path = "data/processed/temporal_transactions.csv"

df.to_csv(output_path, index=False)

print("Temporal dataset saved successfully!")
print(f"Saved to: {output_path}")
print(f"\nFinal dataset shape: {df.shape}")

print("\nSample:")
print(df[["Timestamp", "Temporal Seconds"]].head())