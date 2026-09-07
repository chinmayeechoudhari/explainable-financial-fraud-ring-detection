import pickle
import networkx as nx
import pandas as pd


# --------------------------------------------------
# PART 1: LOAD THE TRANSACTION GRAPH
# --------------------------------------------------

print("Loading transaction graph...")

graph_path = "data/graphs/transaction_graph.gpickle"

with open(graph_path, "rb") as file:
    G = pickle.load(file)

print("Graph loaded successfully!")
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())


# --------------------------------------------------
# PART 2: BASIC GRAPH ANALYSIS
# --------------------------------------------------

print("\nCalculating basic graph statistics...")

print("Graph is directed:", G.is_directed())
print("Graph density:", nx.density(G))

if G.number_of_nodes() > 0:
    print("Average degree:", sum(dict(G.degree()).values()) / G.number_of_nodes())


# --------------------------------------------------
# PART 3: IN-DEGREE AND OUT-DEGREE ANALYSIS
# --------------------------------------------------

print("\nAnalyzing account transaction degrees...")

in_degrees = dict(G.in_degree())
out_degrees = dict(G.out_degree())

print("In-degree analysis completed!")
print("Out-degree analysis completed!")

print("\nSample account degrees:")

sample_nodes = list(G.nodes())[:5]

for node in sample_nodes:
    print(
        f"Account: {node} | "
        f"In-degree: {in_degrees[node]} | "
        f"Out-degree: {out_degrees[node]}"
    )

    # --------------------------------------------------
# PART 4: IDENTIFY HIGHLY CONNECTED ACCOUNTS
# --------------------------------------------------

print("\nIdentifying highly connected accounts...")

total_degrees = dict(G.degree())

top_connected_accounts = sorted(
    total_degrees.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

print("\nTop 10 highly connected accounts:")

for account, degree in top_connected_accounts:
    print(f"Account: {account} | Total Degree: {degree}")


# --------------------------------------------------
# PART 5: IDENTIFY TOP RECEIVER AND SENDER ACCOUNTS
# --------------------------------------------------

print("\nTop 10 receiver accounts...")

top_receivers = sorted(
    in_degrees.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

for account, degree in top_receivers:
    print(f"Account: {account} | In-degree: {degree}")


print("\nTop 10 sender accounts...")

top_senders = sorted(
    out_degrees.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

for account, degree in top_senders:
    print(f"Account: {account} | Out-degree: {degree}")


# --------------------------------------------------
# PART 6: CREATE ANALYSIS SUMMARY DATAFRAME
# --------------------------------------------------

print("\nCreating account analysis summary...")

analysis_df = pd.DataFrame({
    "Account": list(G.nodes()),
    "In Degree": [in_degrees[node] for node in G.nodes()],
    "Out Degree": [out_degrees[node] for node in G.nodes()],
    "Total Degree": [total_degrees[node] for node in G.nodes()]
})

print("Account analysis summary created successfully!")

print("\nSample:")
print(analysis_df.head())

# --------------------------------------------------
# PART 7: IDENTIFY ISOLATED ACCOUNTS
# --------------------------------------------------

print("\nIdentifying isolated accounts...")

isolated_accounts = list(nx.isolates(G))

print("Number of isolated accounts:", len(isolated_accounts))


# --------------------------------------------------
# PART 8: FIND CONNECTED COMPONENTS
# --------------------------------------------------

print("\nAnalyzing weakly connected components...")

components = list(nx.weakly_connected_components(G))

component_sizes = sorted(
    [len(component) for component in components],
    reverse=True
)

print("Total weakly connected components:", len(components))

print("Largest component size:", component_sizes[0])

print("\nTop 10 component sizes:")
print(component_sizes[:10])

# --------------------------------------------------
# PART 9: PAGERANK ANALYSIS
# --------------------------------------------------

print("\nCalculating PageRank scores...")

pagerank_scores = nx.pagerank(
    G,
    alpha=0.85,
    max_iter=100
)

print("PageRank calculation completed successfully!")


# --------------------------------------------------
# PART 10: TOP PAGERANK ACCOUNTS
# --------------------------------------------------

top_pagerank_accounts = sorted(
    pagerank_scores.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

print("\nTop 10 influential accounts based on PageRank:")

for account, score in top_pagerank_accounts:
    print(f"Account: {account} | PageRank Score: {score:.8f}")

    # --------------------------------------------------
# PART 11: ADD PAGERANK TO ANALYSIS DATAFRAME
# --------------------------------------------------

print("\nAdding PageRank scores to account analysis...")

analysis_df["PageRank"] = [
    pagerank_scores[node]
    for node in analysis_df["Account"]
]

print("PageRank scores added successfully!")


# --------------------------------------------------
# PART 12: SAVE GRAPH ANALYSIS RESULTS
# --------------------------------------------------

print("\nSaving graph analysis results...")

output_path = "data/processed/graph_analysis_results.csv"

analysis_df.to_csv(output_path, index=False)

print("Graph analysis results saved successfully!")
print("Saved to:", output_path)

print("\nFinal dataset shape:", analysis_df.shape)

print("\nFinal sample:")
print(analysis_df.head())