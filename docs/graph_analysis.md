# Graph Analysis

## Overview

This module performs structural analysis of the financial transaction graph created during the graph construction stage.

The transaction network is represented as a directed graph where:

- Nodes represent unique bank accounts.
- Directed edges represent transaction relationships between sender and receiver accounts.

The analysis identifies highly connected and influential accounts using graph-based metrics.

---

## Graph Statistics

The constructed transaction graph contains:

- Number of nodes: 705,907
- Number of edges: 1,384,862
- Graph type: Directed
- Graph density: 2.7791509537270087e-06
- Average degree: 3.9236386662832357

---

## Degree Analysis

For every account, the following metrics are calculated:

### In-Degree

The number of unique accounts that send transactions to a particular account.

### Out-Degree

The number of unique accounts that receive transactions from a particular account.

### Total Degree

The total connectivity of an account.

Total Degree = In-Degree + Out-Degree

Accounts with unusually high degree values may represent highly active entities and can be investigated further during fraud analysis.

---

## Connected Accounts

The graph analysis identified:

- Isolated accounts: 0
- Weakly connected components: 160,033
- Largest weakly connected component: 504,391 accounts

The presence of a large connected component indicates that a significant portion of the transaction network belongs to a major interconnected structure.

---

## PageRank Analysis

PageRank is used to identify influential accounts within the transaction network.

Unlike simple degree analysis, PageRank considers both the number and importance of connected accounts.

The top accounts identified through PageRank can be used for further fraud investigation and suspicious account analysis.

---

## Output

The analysis generates the following dataset:

`data/processed/graph_analysis_results.csv`

The output contains the following columns:

| Column | Description |
|---|---|
| Account | Unique account identifier |
| In Degree | Number of unique incoming account connections |
| Out Degree | Number of unique outgoing account connections |
| Total Degree | Total number of account connections |
| PageRank | Influence score within the transaction network |

Final dataset size:

- Rows: 705,907
- Columns: 5

---

## Usage

Run the graph analysis script from the project root:

```bash
python src/graph/analyze_graph.py