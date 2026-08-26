import pandas as pd
import networkx as nx
import pickle

print("Loading feature-engineered dataset...")

file_path = "data/processed/feature_engineered_transactions.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nCreating transaction graph...")

G = nx.DiGraph()

print("Graph initialized successfully!")

print("\nPreparing account nodes...")

# Create unique identifiers for sender and receiver accounts
df["From Node"] = (
    df["From Bank"].astype(str)
    + "_"
    + df["From Account"].astype(str)
)

df["To Node"] = (
    df["To Bank"].astype(str)
    + "_"
    + df["To Account"].astype(str)
)

print("Account node identifiers created successfully!")

print("Unique sender accounts:", df["From Node"].nunique())
print("Unique receiver accounts:", df["To Node"].nunique())

print("\nAdding account nodes to the graph...")

all_nodes = pd.concat([
    df["From Node"],
    df["To Node"]
]).unique()

G.add_nodes_from(all_nodes)

print("Nodes added successfully!")
print("Total nodes:", G.number_of_nodes())

print("\nAggregating transactions between accounts...")

edge_data = df.groupby(
    ["From Node", "To Node"]
).agg(
    Transaction_Count=("Amount Paid", "size"),
    Total_Amount_Paid=("Amount Paid", "sum"),
    Average_Amount_Paid=("Amount Paid", "mean"),
    Laundering_Count=("Is Laundering", "sum")
).reset_index()

print("Transaction aggregation completed!")
print("Unique account relationships:", len(edge_data))

print("\nAdding transaction relationships to the graph...")

for _, row in edge_data.iterrows():

    G.add_edge(
        row["From Node"],
        row["To Node"],
        transaction_count=row["Transaction_Count"],
        total_amount=row["Total_Amount_Paid"],
        average_amount=row["Average_Amount_Paid"],
        laundering_count=row["Laundering_Count"]
    )

print("Edges added successfully!")

print("Total nodes:", G.number_of_nodes())
print("Total edges:", G.number_of_edges())

# =========================
# Graph Statistics
# =========================

print("\nCalculating graph statistics...")

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

print("Graph density:", nx.density(G))

print("Is directed:", G.is_directed())

# =========================
# Graph Validation
# =========================

print("\nValidating graph...")

if G.number_of_nodes() > 0:
    print("Graph contains nodes: PASSED")
else:
    print("Graph contains nodes: FAILED")

if G.number_of_edges() > 0:
    print("Graph contains edges: PASSED")
else:
    print("Graph contains edges: FAILED")

print("Graph validation completed successfully!")

# =========================
# Save Graph
# =========================

print("\nSaving transaction graph...")

output_path = "data/graphs/transaction_graph.gpickle"

with open(output_path, "wb") as file:
    pickle.dump(G, file)

print("Graph saved successfully!")
print("Saved to:", output_path)