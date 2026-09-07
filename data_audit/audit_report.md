# Data Audit Report

## Dataset Overview

- Total Transactions: 6,924,049
- Total Features: 11
- Laundering Transactions: 3,565
- Legitimate Transactions: 6,920,484
- Laundering Percentage: 0.0515%

## Dataset Quality

- Missing Values: 0
- Exact Duplicate Rows: 8
- Duplicate Percentage: 0.000116%
- Decision: Exact duplicate rows will be removed during preprocessing.

## Key Observations

1. The dataset contains 6,924,049 financial transactions.
2. The dataset is highly imbalanced.
3. Only 0.0515% of transactions are labelled as laundering.
4. This class imbalance must be handled carefully during model training.
5. The dataset contains multiple transaction currencies and payment formats.
