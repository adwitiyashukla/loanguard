"""Graph / ring-fraud features.

Builds an applicant similarity graph where edges connect applicants
sharing key identifiers (zip, employer, income bucket). The
community-level fraud rate and degree centrality of each node are
strong fraud signals — they catch organised rings that any single
feature would miss.

For very large datasets the graph would be built with GraphTool or
Neo4j; networkx is plenty for up to ~500k applications.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx

from ..utils.logging import get_logger

log = get_logger(__name__)


def build_graph_features(
    df: pd.DataFrame,
    link_keys: tuple[str, ...] = ("zip_code", "emp_title", "income_bucket"),
    min_link_keys: int = 2,
    max_component_size: int = 5_000,
) -> pd.DataFrame:
    """Attach graph-derived features to each row.

    Added columns:
      - graph_component_size  : size of the connected component
      - graph_degree          : node degree
      - graph_default_rate    : (only set during training) fraud rate of component
    """
    df = df.copy().reset_index(drop=True)

    log.info(
        f"Building applicant graph from {len(df):,} nodes, "
        f"link_keys={link_keys}, min_link_keys={min_link_keys}"
    )

    G = nx.Graph()
    G.add_nodes_from(df.index.tolist())

    # For each link key, group rows and connect pairs that share it.
    # We avoid all-pairs O(N^2) by capping per-group connections.
    for key in link_keys:
        if key not in df.columns:
            continue
        groups = df.groupby(key, observed=True).indices
        for value, idxs in groups.items():
            if pd.isna(value) or len(idxs) < 2 or len(idxs) > 200:
                # ignore huge groups (e.g. NaN zip) — they're too generic
                continue
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a, b = int(idxs[i]), int(idxs[j])
                    if G.has_edge(a, b):
                        G[a][b]["weight"] = G[a][b].get("weight", 1) + 1
                    else:
                        G.add_edge(a, b, weight=1)

    # Keep only edges with weight >= min_link_keys (i.e. share >= K keys)
    weak_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < min_link_keys]
    G.remove_edges_from(weak_edges)

    log.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges():,} edges")

    # Compute per-node features
    component_sizes = np.ones(len(df), dtype=int)
    degrees = np.zeros(len(df), dtype=int)

    for comp in nx.connected_components(G):
        size = len(comp)
        if size > max_component_size:
            continue
        for node in comp:
            component_sizes[node] = size

    for node, deg in G.degree():
        degrees[node] = deg

    df["graph_component_size"] = component_sizes
    df["graph_degree"] = degrees
    df["graph_isolated"] = (degrees == 0).astype(int)

    return df
