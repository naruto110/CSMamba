"""Structural community quality metrics and sweep cut search."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import networkx as nx


def conductance(G: nx.Graph, subset: Iterable) -> float:
    S = set(subset)
    if not S or len(S) == len(G):
        return 1.0
    cut_edges = 0
    vol_S = 0
    for u in S:
        deg = G.degree(u)
        vol_S += deg
        for v in G.neighbors(u):
            if v not in S:
                cut_edges += 1
    vol_out = 2 * G.number_of_edges() - vol_S
    denom = min(vol_S, vol_out)
    if denom == 0:
        return 1.0
    return cut_edges / denom


def internal_density(G: nx.Graph, subset: Iterable) -> float:
    S = set(subset)
    if len(S) < 2:
        return 0.0
    subG = G.subgraph(S)
    m = subG.number_of_edges()
    n = subG.number_of_nodes()
    return 2 * m / (n * (n - 1))


def sweep_cut_best(
    G: nx.Graph,
    ranking: List,
    min_size: int = 3,
    max_size: int | None = None,
    density_weight: float = 1.0,
) -> Tuple[List, List[dict]]:
    best_score = float("inf")
    best_set: List = []
    history: List[dict] = []
    max_size = max_size or len(ranking)

    for k in range(min_size, max_size + 1):
        subset = ranking[:k]
        cond = conductance(G, subset)
        dens = internal_density(G, subset)
        score = cond + density_weight * (1.0 - dens)
        history.append(
            {"k": k, "conductance": cond, "density": dens, "score": score, "density_weight": density_weight}
        )
        if score < best_score:
            best_score = score
            best_set = list(subset)
    return best_set, history


def community_pr_re_f1(pred: Iterable, node_to_comm: dict, query) -> dict:
    """
    Compare predicted community to ground-truth community of the query node.
    Returns precision/recall/F1 and IoU.
    """
    if query not in node_to_comm:
        return {}
    gt_comm = node_to_comm[query]
    gt_nodes = {n for n, c in node_to_comm.items() if c == gt_comm}
    pred_set = set(pred)
    inter = pred_set & gt_nodes
    prec = len(inter) / (len(pred_set) + 1e-8)
    rec = len(inter) / (len(gt_nodes) + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    iou = len(inter) / (len(pred_set | gt_nodes) + 1e-8)
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou, "gt_size": len(gt_nodes)}
 