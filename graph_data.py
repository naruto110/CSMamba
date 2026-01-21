"""Graph and structural feature utilities."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List

import networkx as nx
import numpy as np


def base_structural_features(G: nx.Graph, node) -> np.ndarray:
    deg = G.degree(node)
    max_deg = max(1, max(dict(G.degree()).values()))
    clustering = nx.clustering(G, node)
    neighbors = list(G.neighbors(node))
    if neighbors:
        nbr_deg = [G.degree(n) for n in neighbors]
        avg_nbr_deg = float(np.mean(nbr_deg))
        std_nbr_deg = float(np.std(nbr_deg))
    else:
        avg_nbr_deg = 0.0
        std_nbr_deg = 0.0

    egonet = nx.ego_graph(G, node, radius=1)
    ego_edges = egonet.number_of_edges()
    ego_nodes = egonet.number_of_nodes()
    ego_density = 0.0
    if ego_nodes > 1:
        ego_density = 2.0 * ego_edges / (ego_nodes * (ego_nodes - 1))

    return np.array(
        [
            deg,
            deg / max_deg,
            clustering,
            avg_nbr_deg / max_deg,
            std_nbr_deg / max_deg,
            ego_density,
        ],
        dtype=np.float32,
    )


def ego_subgraph_features(G: nx.Graph, node) -> np.ndarray:
    egonet = nx.ego_graph(G, node, radius=1)
    degrees = [deg for _, deg in egonet.degree()]
    size = egonet.number_of_nodes()
    edges = egonet.number_of_edges()
    density = 0.0
    if size > 1:
        density = 2.0 * edges / (size * (size - 1))

    triangles = sum(nx.triangles(egonet).values()) / 3.0
    max_triangles = size * (size - 1) * (size - 2) / 6 if size >= 3 else 1
    tri_ratio = triangles / max_triangles

    deg_mean = float(np.mean(degrees)) if degrees else 0.0
    deg_std = float(np.std(degrees)) if degrees else 0.0
    deg_max = float(np.max(degrees)) if degrees else 0.0

    return np.array(
        [size, edges, density, tri_ratio, deg_mean, deg_std, deg_max],
        dtype=np.float32,
    )


def structural_token(G: nx.Graph, node) -> np.ndarray:
    base = base_structural_features(G, node)
    ego = ego_subgraph_features(G, node)
    return np.concatenate([base, ego], dtype=np.float32)


def token_dim(G: nx.Graph) -> int:
    sample_node = next(iter(G.nodes()))
    return structural_token(G, sample_node).shape[0]


def jaccard_neighbor_similarity(G: nx.Graph, u, v) -> float:
    u_n = set(G.neighbors(u))
    v_n = set(G.neighbors(v))
    inter = len(u_n & v_n)
    union = len(u_n | v_n)
    if union == 0:
        return 0.0
    return inter / union


def shortest_path_matrix(G: nx.Graph, nodes: Iterable) -> Dict:
    """Compute shortest path lengths for a list of nodes."""
    lengths = {}
    for n in nodes:
        lengths[n] = dict(nx.single_source_shortest_path_length(G, n))
    return lengths


def compute_hop_dists(G: nx.Graph, node_index: List, max_hop: int = 5):
    """
    Compute truncated shortest-path distances for nodes in node_index.
    Returns a float numpy array dist[i, j] (max_hop+1 when beyond cutoff).
    """
    idx = {n: i for i, n in enumerate(node_index)}
    n = len(node_index)
    dist = np.full((n, n), fill_value=max_hop + 1, dtype=np.float32)
    np.fill_diagonal(dist, 0.0)
    for i, u in enumerate(node_index):
        lengths = nx.single_source_shortest_path_length(G, u, cutoff=max_hop)
        for v, d in lengths.items():
            j = idx.get(v)
            if j is not None:
                dist[i, j] = float(d)
    return dist


def boundary_softness(G: nx.Graph, node_index: List) -> np.ndarray:
    """
    Heuristic boundary/bridge softness score in [0,1]:
    higher means more ambiguous (less dense / lower clustering).
    """
    scores = []
    max_deg = max(1, max(dict(G.degree()).values()))
    for n in node_index:
        clustering = nx.clustering(G, n)
        ego = nx.ego_graph(G, n, radius=1)
        ego_edges = ego.number_of_edges()
        ego_nodes = ego.number_of_nodes()
        ego_density = 0.0
        if ego_nodes > 1:
            ego_density = 2.0 * ego_edges / (ego_nodes * (ego_nodes - 1))
        # Boundary score rises when clustering / ego density are low.
        score = (1.0 - clustering) * (1.0 - ego_density)
        score = float(np.clip(score, 0.0, 1.0))
        scores.append(score)
    return np.array(scores, dtype=np.float32)


def compute_rw_cooc_for_anchor(anchor_id, walks: List[List[int]]):
    """
    Flatten walks and count node occurrences; return normalized frequencies.
    """
    counter = {}
    total = 0
    for w in walks:
        for n in w:
            counter[n] = counter.get(n, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items() if k != anchor_id}


def compute_local_jaccard_for_anchor(anchor_id, candidates, neighbors: Dict):
    """
    Compute Jaccard similarity between anchor's neighbors and candidates' neighbors.
    """
    anchor_nei = set(neighbors.get(anchor_id, []))
    scores = {}
    for c in candidates:
        nei_c = set(neighbors.get(c, []))
        inter = len(anchor_nei & nei_c)
        union = len(anchor_nei | nei_c)
        scores[c] = inter / union if union > 0 else 0.0
    return scores


def get_structural_pos_neg(
    anchor_id,
    rw_scores: Dict,
    jaccard_scores: Dict,
    pos_top_ratio: float = 0.1,
    neg_bottom_ratio: float = 0.5,
    jaccard_low_thresh: float = 0.1,
):
    """
    Use RW co-occurrence + local Jaccard to pick structural pos/neg.
    """
    if not rw_scores:
        return [], []
    items = sorted(rw_scores.items(), key=lambda x: x[1], reverse=True)
    n = len(items)
    pos_cut = max(1, int(n * pos_top_ratio))
    neg_cut = max(1, int(n * neg_bottom_ratio))
    pos_candidates = [i[0] for i in items[:pos_cut]]
    neg_candidates = [i[0] for i in items[-neg_cut:]]
    pos_nodes = [p for p in pos_candidates if jaccard_scores.get(p, 0.0) >= jaccard_low_thresh and p != anchor_id]
    neg_nodes = [n for n in neg_candidates if jaccard_scores.get(n, 0.0) < jaccard_low_thresh and n != anchor_id]
    return pos_nodes, neg_nodes
