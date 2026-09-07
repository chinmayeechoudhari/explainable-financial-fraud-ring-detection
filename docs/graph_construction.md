# Graph Construction

## Overview

The transaction dataset was transformed into a directed graph to represent financial relationships between accounts.

## Graph Representation

- **Nodes:** Unique bank-account combinations.
- **Edges:** Aggregated transactions from a sender account to a receiver account.
- **Graph Type:** Directed graph (`DiGraph`).

Each node is represented using:

Bank_ID + Account_ID

Example:

11_ACCOUNT123

Each directed edge represents:

Sender Account → Receiver Account

## Edge Attributes

The following attributes were stored for each account relationship:

- Transaction Count
- Total Amount Paid
- Average Amount Paid
- Laundering Count

Multiple transactions between the same sender and receiver were aggregated into a single directed edge.

## Graph Statistics

- Total Transactions: 6,924,041
- Total Nodes: 705,907
- Total Edges: 1,384,862
- Graph Density: 2.7791509537270087e-06
- Directed: True

## Validation

The constructed graph was validated by confirming:

- The graph contains account nodes.
- The graph contains transaction edges.
- The graph is directed.
- Node and edge counts were successfully generated.

## Output

The constructed graph was saved as:

`data/graphs/transaction_graph.gpickle`