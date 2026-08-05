<div align="center">

# Explainable Temporal Graph Attention Network for Financial Fraud Ring Detection

### Detecting Coordinated Financial Fraud Rings using Temporal Graph Neural Networks and Explainable AI

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch-Geometric-orange.svg)]()
[![Graph Neural Networks](https://img.shields.io/badge/Graph-Neural%20Networks-green.svg)]()
[![Research](https://img.shields.io/badge/Research-IEEE%20Inspired-blueviolet.svg)]()
[![License](https://img.shields.io/badge/License-MIT-success.svg)]()

*A research-driven Graph AI framework for detecting organized financial fraud rings through temporal transaction network analysis and explainable graph learning.*

</div>

---

# Abstract

Financial institutions traditionally rely on tabular machine learning algorithms to detect fraudulent transactions. While these approaches perform well in identifying isolated fraudulent activities, they are inherently limited in detecting **coordinated fraud rings**, where multiple accounts collaborate to conceal illicit financial activities through layered and interconnected transaction networks.

This project proposes an **Explainable Temporal Graph Attention Network (TGAT)** capable of modeling financial transactions as a dynamic graph. Rather than evaluating transactions independently, the model learns structural relationships between accounts, temporal transaction behavior, and community-level interactions to identify suspicious collaborative networks.

To improve transparency and support financial investigations, the proposed framework incorporates **Explainable Artificial Intelligence (XAI)** techniques that provide interpretable explanations for every detected fraud ring.

---

# Problem Statement

Existing fraud detection systems primarily analyze transactions independently using tabular machine learning models, making them ineffective in identifying coordinated fraud rings that involve multiple interconnected financial accounts and evolve over time.

---

# Motivation

Organized financial crimes rarely occur through a single fraudulent transaction.

Modern fraudsters create networks of accounts that:

- Transfer money repeatedly among themselves.
- Perform circular fund movements.
- Use intermediary (mule) accounts.
- Split large transactions into smaller transfers.
- Operate across multiple financial institutions.
- Continuously evolve their transaction behavior to evade detection.

Traditional machine learning models ignore these relationships because they treat every transaction as an independent observation.

Graph Neural Networks provide a natural solution by learning both the **structure** and **evolution** of financial transaction networks.

---

# Research Objectives

The primary objectives of this research are:

- Detect coordinated financial fraud rings instead of isolated fraudulent transactions.
- Represent financial transaction history as a dynamic graph.
- Learn temporal transaction behavior using Temporal Graph Attention Networks.
- Identify suspicious account communities.
- Generate interpretable explanations for every fraud prediction.
- Provide visual investigation support through graph-based visualization.
- Compare graph learning approaches with traditional machine learning techniques.

---

# Proposed Methodology

The proposed framework consists of six major stages.

```
IBM AML Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Financial Graph Construction
        │
        ▼
Temporal Feature Engineering
        │
        ▼
Temporal Graph Attention Network
        │
        ▼
Fraud Ring Classification
        │
        ▼
Explainable AI Module
        │
        ▼
Interactive Investigation Dashboard
```

---

# Dataset

This project utilizes the **IBM Anti-Money Laundering (AML) Transaction Dataset**, which contains synthetic financial transaction data designed for anti-money laundering and fraud detection research.

### Dataset Attributes

| Feature | Description |
|----------|-------------|
| Timestamp | Transaction time |
| From Bank | Source financial institution |
| From Account | Sender account |
| To Bank | Destination financial institution |
| To Account | Receiver account |
| Amount | Transaction amount |
| Payment Format | Payment method |
| Is Laundering | Ground truth label |

Unlike conventional fraud datasets, the IBM AML dataset naturally represents a financial transaction network, making it highly suitable for graph learning.

---

# Graph Representation

Instead of representing the data as independent rows, the complete transaction history is transformed into a graph.

## Nodes

Every financial account becomes a node.

```
Account A001

Account B425

Account C731
```

---

## Edges

Every transaction becomes a directed edge.

```
A001 ─────────▶ B425

B425 ─────────▶ C731

C731 ─────────▶ A001
```

---

## Edge Attributes

Each edge stores:

- Transaction Amount
- Timestamp
- Payment Format
- Source Bank
- Destination Bank

---

## Node Features

Each account is represented using graph and statistical features including:

- Total Incoming Transactions
- Total Outgoing Transactions
- Total Amount Sent
- Total Amount Received
- Average Transaction Amount
- Account Degree
- Degree Centrality
- Betweenness Centrality
- PageRank Score
- Clustering Coefficient

---

# Model Architecture

The proposed architecture consists of three major components.

## Stage 1 — Graph Construction

- Data Cleaning
- Feature Engineering
- Dynamic Graph Generation

---

## Stage 2 — Graph Learning

Temporal Graph Attention Network (TGAT)

The model learns

- Transaction relationships
- Neighborhood information
- Temporal evolution
- Community structure
- Attention weights

---

## Stage 3 — Explainability

Graph Explainability techniques identify:

- Important accounts
- Suspicious transaction paths
- Influential neighbors
- High-risk communities
- Attention scores
- Fraud propagation patterns

---

# Project Workflow

```
Historical Transactions
            │
            ▼
Data Cleaning & Validation
            │
            ▼
Feature Engineering
            │
            ▼
Graph Construction
            │
            ▼
Temporal Graph Generation
            │
            ▼
Temporal Graph Attention Network
            │
            ▼
Node Embedding Generation
            │
            ▼
Fraud Ring Detection
            │
            ▼
Explainability Module
            │
            ▼
Visualization Dashboard
            │
            ▼
Investigation Report
```

---

# Technology Stack

| Layer | Technology |
|--------|------------|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Graph Learning | PyTorch Geometric |
| Graph Analytics | NetworkX |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Graphviz |
| API | FastAPI |
| Database | PostgreSQL |
| Dashboard | React.js |
| Deployment | Docker |

---

# Repository Structure

```
financial-fraud-ring-detection/

│

├── data/
│   ├── raw/
│   ├── processed/
│   ├── graphs/
│   └── features/
│
├── notebooks/
│
├── src/
│   ├── preprocessing/
│   ├── graph_builder/
│   ├── feature_engineering/
│   ├── models/
│   │      ├── tgat.py
│   │      ├── gat.py
│   │      ├── graphsage.py
│   │      └── baselines.py
│   │
│   ├── explainability/
│   │      ├── gnnexplainer.py
│   │      ├── pgexplainer.py
│   │      └── graph_visualizer.py
│   │
│   ├── dashboard/
│   │
│   └── utils/
│
├── experiments/
│
├── configs/
│
├── outputs/
│
├── tests/
│
├── requirements.txt
│
├── LICENSE
│
└── README.md
```

---

# Model Inputs

The model receives:

- Financial transaction graph
- Node feature matrix
- Edge feature matrix
- Temporal information
- Graph topology

---

# Model Outputs

The proposed system predicts:

- Fraud Ring Probability
- Suspicious Accounts
- High-risk Communities
- Important Transaction Paths
- Confidence Score
- Graph-based Explanation

---

# Explainable AI

Unlike conventional fraud detection systems that only output a binary prediction, this framework explains every decision.

Example:

```
Fraud Ring Detected

Confidence Score:
96.7%

Reason:

✓ Circular Money Movement

✓ Repeated Cross-Bank Transactions

✓ Shared Neighbor Accounts

✓ High Temporal Activity

✓ Community-Level Anomaly
```

This enables investigators to understand **why** a fraud ring has been detected.

---

# Evaluation Metrics

## Classification Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## Graph Metrics

- Precision@K
- Recall@K
- Mean Average Precision

---

## Explainability Metrics

- Fidelity
- Sparsity
- Stability

---

# Research Contributions

This work aims to contribute the following:

- Dynamic financial transaction graph construction
- Temporal graph learning for fraud ring detection
- Explainable graph neural network framework
- Community-based fraud investigation
- Comparative analysis with traditional machine learning models
- Interactive graph visualization for financial investigators

---

# Future Enhancements

- Streaming graph neural networks
- Online graph learning
- Real-time transaction monitoring
- Cross-bank collaborative fraud detection
- Federated graph learning
- Blockchain transaction analysis
- Large Language Model assisted fraud investigation
- Knowledge Graph integration

---

# Development Roadmap

| Task | Status |
|------|--------|
| Literature Survey | ✅ Completed |
| Dataset Collection | ⏳ In Progress |
| Data Preprocessing | ⏳ Pending |
| Graph Construction | ⏳ Pending |
| Feature Engineering | ⏳ Pending |
| TGAT Implementation | ⏳ Pending |
| Explainability Module | ⏳ Pending |
| Dashboard Development | ⏳ Pending |
| Model Evaluation | ⏳ Pending |
| Research Paper | ⏳ Pending |

---

# References

1. IBM Anti-Money Laundering Dataset
2. Temporal Graph Attention Networks (TGAT)
3. Graph Neural Networks (GNN)
4. Graph Attention Networks (GAT)
5. PyTorch Geometric
6. GNNExplainer
7. PGExplainer

---

# Disclaimer

This project is intended solely for **academic research and educational purposes**. The IBM AML dataset used in this repository is a **synthetic dataset** released for research and does not contain any real customer financial information.

---

<div align="center">

**Financial Fraud Ring Detection using Explainable Graph Intelligence**

*Built for Research • Explainable AI • Graph Machine Learning • Financial Security*

</div>