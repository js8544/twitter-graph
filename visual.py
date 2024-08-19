import json
import networkx as nx
from pyvis.network import Network
import pandas as pd

# Load the data
try:
    with open('output.jsonl', 'r', encoding='utf-8') as f:
        posts = [json.loads(line) for line in f]
    print("Loaded posts successfully")
except Exception as e:
    print(f"Error loading posts: {e}")

try:
    with open('similar_pairs.jsonl', 'r', encoding='utf-8') as f:
        similarities = [json.loads(line) for line in f]
    print("Loaded similarities successfully")
except Exception as e:
    print(f"Error loading similarities: {e}")

# Create a DataFrame for posts
try:
    posts_df = pd.DataFrame(posts)
    posts_dict = posts_df.set_index('number').T.to_dict()
    print("Created DataFrame for posts")
except Exception as e:
    print(f"Error creating DataFrame: {e}")

# Initialize the graph
G = nx.Graph()

# Add nodes
for post in posts:
    try:
        G.add_node(post['number'],
                   title=post['quote'],
                   label=post['number'],
                   full_text=post['original_text'])
    except Exception as e:
        print(f"Error adding node {post['number']}: {e}")

# Add edges
for similarity in similarities:
    try:
        source = similarity['source']
        target = similarity['target']
        value = similarity.get('value', 0)
        G.add_edge(source,
                   target,
                   weight=value,
                   title=f'Similarity: {value:.2f}')
    except Exception as e:
        print(f"Error adding edge from {source} to {target}: {e}")

# Filter to get the top nodes with the highest degrees
top_num = min(2000, len(G.nodes))
top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_num]
top_nodes_set = {node[0] for node in top_nodes}

# Add neighbors of the top 50 nodes to the set
neighbors = set()
for node in top_nodes_set:
    neighbors.update(G.neighbors(node))

# Union the top 50 nodes and their neighbors
filtered_nodes = top_nodes_set.union(neighbors)

# Filter only nodes with at least one edge
filtered_nodes = [node for node in filtered_nodes if G.degree[node] > 0]

# Create a subgraph with the filtered nodes
subG = G.subgraph(filtered_nodes)

# Initialize Pyvis network
net = Network(notebook=False,
              height="750px",
              width="100%",
              bgcolor="#222222",
              font_color="white")

# Load the Networkx subgraph
net.from_nx(subG)

# Customize the nodes and edges
for node in net.nodes:
    try:
        degree = G.degree[node['id']]
        node['title'] = posts_dict[node['id']]['original_text']
        node['value'] = degree  # Size of nodes proportional to degree
    except Exception as e:
        print(f"Error customizing node {node['id']}: {e}")

for edge in net.edges:
    try:
        edge['width'] = edge.get(
            'weight', 0) * 10  # Edge thickness proportional to similarity
    except Exception as e:
        print(
            f"Error customizing edge from {edge['from']} to {edge['to']}: {e}")

# Enable physics for better layout
net.show_buttons(filter_=['physics'])

# Save and display the graph
net.show('twitter_similarity_graph.html', notebook=False)
