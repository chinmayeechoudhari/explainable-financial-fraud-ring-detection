# Data Preprocessing Report

## Overview

The raw financial transaction dataset was preprocessed to prepare it for further feature engineering, graph construction, and machine learning tasks.

## Dataset Cleaning

- Original dataset shape: 6,924,049 rows and 11 columns
- Exact duplicate rows found: 8
- Exact duplicate rows removed: 8
- Final dataset shape: 6,924,041 rows and 16 columns
- Missing values: 0
- Remaining duplicate rows: 0

## Timestamp Processing

The `Timestamp` column was converted from string format to datetime format.

The following time-based features were extracted:

- Year
- Month
- Day
- Hour
- DayOfWeek

## Categorical Data Processing

The following categorical columns were encoded into numerical values:

- Receiving Currency
- Payment Currency
- Payment Format

## Account Column Processing

The account columns were renamed for better readability:

- `Account` → `From Account`
- `Account.1` → `To Account`

The account identifiers were retained because they will be useful for graph construction and fraud ring analysis.

## Numerical Data Validation

The following numerical columns were validated:

- Amount Received
- Amount Paid

No negative values or zero values were found.

## Target Column Validation

The target column `Is Laundering` was validated.

- Valid classes: 0 and 1
- Invalid values: 0
- Laundering transactions: 3,565
- Legitimate transactions: 6,920,476

The dataset remains highly imbalanced, with approximately 0.0515% of transactions labelled as laundering.

## Processed Dataset

The final processed dataset contains:

- 6,924,041 transactions
- 16 columns
- No missing values
- No duplicate rows

The processed dataset is saved as:

`data/processed/processed_transactions.csv`