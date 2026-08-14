from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
from ..utils.logging import get_logger
log = get_logger(__name__)

def build_graph_features(df: pd.DataFrame, link_keys: tuple[str, ...]=('zip_code', 'emp_title', 'income_bucket'), min_link_keys: int=2, max_component_size: int=5000) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    log.info(f'Building applicant graph from {len(df):,} nodes, link_keys={link_keys}, min_link_keys={min_link_keys}')
    G = nx.Graph()
    G.add_nodes_from(df.index.tolist())
    for key in link_keys:
        if key not in df.columns:
            continue
        groups = df.groupby(key, observed=True).indices
        for (value, idxs) in groups.items():
            if pd.isna(value) or len(idxs) < 2 or len(idxs) > 200:
                continue
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    (a, b) = (int(idxs[i]), int(idxs[j]))
                    if G.has_edge(a, b):
                        G[a][b]['weight'] = G[a][b].get('weight', 1) + 1
                    else:
                        G.add_edge(a, b, weight=1)
    weak_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] < min_link_keys]
    G.remove_edges_from(weak_edges)
    log.info(f'Graph: {G.number_of_nodes()} nodes, {G.number_of_edges():,} edges')
    component_sizes = np.ones(len(df), dtype=int)
    degrees = np.zeros(len(df), dtype=int)
    for comp in nx.connected_components(G):
        size = len(comp)
        if size > max_component_size:
            continue
        for node in comp:
            component_sizes[node] = size
    for (node, deg) in G.degree():
        degrees[node] = deg
    df['graph_component_size'] = component_sizes
    df['graph_degree'] = degrees
    df['graph_isolated'] = (degrees == 0).astype(int)
    return df
