# Feature Engineering

## Objective

The objective of feature engineering is to create additional meaningful features from the preprocessed transaction dataset that can help identify patterns associated with financial fraud and money laundering.

## Input Dataset

The feature engineering pipeline uses:

`data/processed/processed_transactions.csv`

The input dataset contains **6,924,041 transactions** and **16 columns**.

## Engineered Features

### 1. Amount Difference

Calculated as:

Amount Received - Amount Paid

This feature captures the difference between the received and paid transaction amounts.

### 2. Amount Ratio

Calculated as:

Amount Received / Amount Paid

This feature represents the relationship between the received and paid amounts.

### 3. Same Bank Transaction

- `1` indicates that the sender and receiver belong to the same bank.
- `0` indicates that they belong to different banks.

### 4. Cross Bank Transaction

- `1` indicates a transaction between different banks.
- `0` indicates a transaction within the same bank.

### 5. Transaction Time Category

The transaction hour is grouped into four categories:

| Category | Time |
|---|---|
| 0 | 00:00–05:00 |
| 1 | 06:00–11:00 |
| 2 | 12:00–17:00 |
| 3 | 18:00–23:00 |

### 6. Is Weekend

- `1` indicates Saturday or Sunday.
- `0` indicates Monday through Friday.

### 7. Log Amount Received

A logarithmic transformation using `log(1 + Amount Received)` to reduce the effect of extremely large transaction values.

### 8. Log Amount Paid

A logarithmic transformation using `log(1 + Amount Paid)` to reduce the effect of extremely large transaction values.

### 9. Same Currency

- `1` indicates that the receiving and payment currencies are the same.
- `0` indicates different currencies.

## Validation

The engineered features were validated for:

- Missing values
- Infinite values
- Feature distributions

No missing or infinite values were found in the engineered features.

## Output Dataset

The final feature-engineered dataset contains:

- **6,924,041 transactions**
- **25 columns**
- **9 newly engineered features**

Output file:

`data/processed/feature_engineered_transactions.csv`

## Pipeline

Raw Dataset
↓
Data Audit
↓
Data Preprocessing
↓
Feature Engineering
↓
Feature-Engineered Dataset