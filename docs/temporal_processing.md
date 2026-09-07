# Temporal Processing

## Overview

This module prepares transaction data for temporal graph analysis and future TGAT modeling.

The feature-engineered transaction dataset is processed using the `Timestamp` column to ensure that all transactions are arranged in chronological order.

## Input Dataset

The input dataset is:

`data/processed/feature_engineered_transactions.csv`

The dataset contains 6,924,041 transactions and 25 columns.

## Temporal Processing Steps

### 1. Timestamp Conversion

The `Timestamp` column is converted to a datetime format using Pandas.

Invalid timestamps are checked during processing.

Result:

- Invalid timestamps: 0
- Timestamp conversion: Successful

### 2. Chronological Sorting

All transactions are sorted in ascending chronological order based on the `Timestamp` column.

The transaction period ranges from:

- Oldest transaction: 2022-09-01 00:00:00
- Newest transaction: 2022-09-17 15:28:00

Chronological ordering validation passed successfully.

### 3. Temporal Feature Creation

A numerical temporal feature called `Temporal Seconds` is created.

The value represents the number of seconds elapsed since the earliest transaction in the dataset.

The earliest transaction therefore has:

`Temporal Seconds = 0`

This temporal representation can be used as an input for temporal graph analysis and future TGAT modeling.

### 4. Validation

The processed dataset was validated to ensure:

- No invalid timestamps were present.
- Transactions were sorted chronologically.
- No negative temporal values were generated.

All validation checks passed successfully.

## Output Dataset

The processed dataset is saved as:

`data/processed/temporal_transactions.csv`

The final dataset contains:

- Rows: 6,924,041
- Columns: 26

The additional column created during processing is:

`Temporal Seconds`

## Source Code

The temporal processing implementation is located at:

`src/data/temporal_processing.py`