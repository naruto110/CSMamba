"""Query-time community search using trained embeddings (unsupervised ranking).

Two-stage design (embedding-guided candidate + connectivity-preserving peeling).
Ground truth labels are ONLY used for evaluation, never for search/ranking.
"""

from __future__ import annotations

import argparse
from collections import deque
from typing import List, Set, Optional

import numpy as np
import torch

from config import SearchConfig
from demo_dataset import load_demo_graph
from graph_data import token_dim
from models import MambaCommunityModel
from rw_tokenizer import RandomWalkTokenizer
from utils import get_device, log
import networkx as nx


def _load_custom_npz(path: str):
    data = np.load(path, allow_pickle=True)
    nodes = list(data["nodes"])
    labels = data["labels"] if "labels" in data else None
    edges = data["edges"] if "edges" in data else None
    tokens = data["tokens"] if "tokens" in data else None
    if edges is None:
        raise ValueError("Custom npz must contain 'edges' array for search.")
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges.tolist())
    node_to_comm = {}
    if labels is not None and len(labels) == len(nodes):
        node_to_comm = {n: int(lbl) for n, lbl in zip(nodes, labels)}
    return G, node_to_comm, nodes, tokens


def evaluate_query(
    query_id,
    pred_nodes: list,
    node_to_comm: dict,
) -> dict:
    """
    Evaluation ONLY: uses ground-truth labels to compute precision/recall/F1/IoU.
    MUST NOT influence search logic.
    """
    if query_id not in node_to_comm:
        return {}
    gt_set = {n for n, c in node_to_comm.items() if c == node_to_comm[query_id]}
    pred_set = set(pred_nodes)
    inter = len(pred_set & gt_set)
    prec = inter / (len(pred_set) + 1e-9)
    rec = inter / (len(gt_set) + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    iou = inter / (len(pred_set | gt_set) + 1e-9)
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "iou": iou,
        "gt_size": len(gt_set),
    }


def load_model(checkpoint_path: str, token_dim_size: int, search_cfg: SearchConfig, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    d = ckpt.get("config", {})
    # Backfill walk params if not provided
    if search_cfg.walk_lengths is None and ckpt.get("walk_lengths"):
        search_cfg.walk_lengths = ckpt.get("walk_lengths")
    if search_cfg.walk_length is None and ckpt.get("walk_length"):
        search_cfg.walk_length = ckpt.get("walk_length")
    # Backfill tokenizer bias params
    if hasattr(search_cfg, "restart_p") and d.get("restart_p") is not None:
        search_cfg.restart_p = d.get("restart_p")
    if hasattr(search_cfg, "jaccard_bias") and d.get("jaccard_bias") is not None:
        search_cfg.jaccard_bias = d.get("jaccard_bias")
    model = MambaCommunityModel(
        token_dim_size,
        d.get("token_hidden_dim", 64),
        d.get("d_model", 96),
        d.get("d_state", 16),
        d.get("d_conv", 4),
        dropout=d.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_embeddings(path: str, nodes: list) -> torch.Tensor:
    if path.endswith(".pkl"):
        import pickle
        with open(path, "rb") as f:
            pkg = pickle.load(f)
        emb = pkg.get("emb")
        file_nodes = pkg.get("nodes")
        if emb is None:
            raise ValueError("pkl does not contain emb")
    else:
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            emb = data["emb"]
            file_nodes = data["nodes"] if "nodes" in data else None
        else:
            emb = data
            file_nodes = None
    if file_nodes is not None:
        file_nodes = list(file_nodes)
        idx = {n: i for i, n in enumerate(file_nodes)}
        order = [idx[n] for n in nodes]
        emb = emb[order]
    return torch.from_numpy(emb)

def load_tokens(path: str, nodes: list) -> torch.Tensor:
    if path.endswith(".pkl"):
        import pickle
        with open(path, "rb") as f:
            pkg = pickle.load(f)
        tokens = pkg.get("tokens")
        file_nodes = pkg.get("nodes")
        if tokens is None:
            raise ValueError("pkl does not contain tokens")
    else:
        data = np.load(path, allow_pickle=True)
        tokens = data["tokens"]
        file_nodes = data["nodes"] if "nodes" in data else None
    if file_nodes is not None:
        file_nodes = list(file_nodes)
        idx = {n: i for i, n in enumerate(file_nodes)}
        order = [idx[n] for n in nodes]
        tokens = tokens[order]
    return torch.from_numpy(tokens).float()


def _component_containing_q(q: int, nodes_subset: Set[int], adj: List[Set[int]]) -> Set[int]:
    """Return the connected component containing q within nodes_subset."""
    if q not in nodes_subset:
        return set()
    visited = set([q])
    dq = deque([q])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v in nodes_subset and v not in visited:
                visited.add(v)
                dq.append(v)
    return visited


def candidate_connected_subgraph(
    q: int,
    emb: np.ndarray,
    adj: List[Set[int]],
    k_e_init: int = 200,
    k_e_max: int = 2000,
    k_e_step: int = 200,
    k_min: int = 30,
) -> Set[int]:
    """
    Embedding-guided candidate selection: grow by top-k similarity, keep CC containing q.
    Returns a connected set containing q.
    """
    sims = emb @ emb[q]
    sims[q] = -np.inf  # exclude self from ranking
    order = np.argsort(-sims)

    total_nodes = len(order) + 1
    k_e = min(k_e_init, total_nodes - 1)
    best_cc: Set[int] = set([q])

    while k_e <= min(k_e_max, total_nodes - 1):
        top_k = order[:k_e]
        cand = set(top_k.tolist())
        cand.add(q)
        cc_q = _component_containing_q(q, cand, adj)
        best_cc = cc_q
        if len(cc_q) >= k_min:
            break
        k_e += k_e_step

    if len(best_cc) < k_min:
        # Fallback: q plus immediate neighbors
        best_cc = set([q]) | set(adj[q])
    return best_cc


def _set_objective(S: Set[int], sim_vec: np.ndarray, adj: List[Set[int]], beta: float, lam: float) -> float:
    """Compute F(S) = sum sim - lam * out_edges + beta * avg_internal_degree."""
    if not S:
        return -np.inf
    sim_sum = float(sim_vec[list(S)].sum())
    internal_edges_twice = sum(len(adj[u] & S) for u in S)
    internal_edges = internal_edges_twice / 2.0
    out_edges = sum(1 for u in S for v in adj[u] if v not in S)
    avg_internal_degree = (2.0 * internal_edges) / max(len(S), 1)
    return sim_sum - lam * out_edges + beta * avg_internal_degree


def connectivity_preserving_peeling(
    q: int,
    S0: Set[int],
    emb: np.ndarray,
    adj: List[Set[int]],
    alpha: float = 1.0,
    beta: float = 0.05,
    lam: float = 0.05,
    min_size: int = 5,
    stop_rule: str = "knee",
) -> Set[int]:
    """
    Peeling with connectivity constraint and joint embedding + structure scoring.
    """
    S: Set[int] = set(S0)
    sim_vec = emb @ emb[q]

    best_S = set(S)
    best_score = _set_objective(S, sim_vec, adj, beta, lam)

    while len(S) > min_size:
        scores = []
        nodes_list = []
        for u in S:
            if u == q:
                continue
            d_in = len(adj[u] & S)
            d_out = len(adj[u]) - d_in
            keep = alpha * sim_vec[u] + beta * d_in - lam * d_out
            scores.append(keep)
            nodes_list.append(u)
        if not nodes_list:
            break
        scores_arr = np.array(scores, dtype=np.float32)
        idx_min = int(np.argmin(scores_arr))
        u_remove = nodes_list[idx_min]
        S.remove(u_remove)

        # Preserve connectivity: keep only component containing q
        S = _component_containing_q(q, S, adj)

        current_score = _set_objective(S, sim_vec, adj, beta, lam)
        if current_score > best_score:
            best_score = current_score
            best_S = set(S)

    if stop_rule == "knee":
        return best_S
    return S


def search_community_for_query(
    q: int,
    emb: np.ndarray,
    adj: List[Set[int]],
    k_e_init: int = 200,
    k_e_max: int = 2000,
    k_e_step: int = 200,
    k_min: int = 30,
    min_size: int = 5,
    alpha: float = 1.0,
    beta: float = 0.05,
    lam: float = 0.05,
) -> list[int]:
    """
    Two-stage unsupervised community search (no GT usage).
    Returns sorted list of node indices in the predicted community.
    """
    S0 = candidate_connected_subgraph(
        q,
        emb,
        adj,
        k_e_init=k_e_init,
        k_e_max=k_e_max,
        k_e_step=k_e_step,
        k_min=k_min,
    )
    S = connectivity_preserving_peeling(
        q,
        S0,
        emb,
        adj,
        alpha=alpha,
        beta=beta,
        lam=lam,
        min_size=min_size,
    )
    return sorted(S)


def _load_anchor_list(path: str, nodes: list) -> List[int]:
    """Load anchor node ids from a text file (one id per line), filtered to existing nodes."""
    anchors: List[int] = []
    node_set = set(nodes)
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                n = int(line)
            except ValueError:
                continue
            if n in node_set:
                anchors.append(n)
    return anchors


def search(cfg: SearchConfig, query, custom_npz: str | None = None, anchors_file: Optional[str] = None):
    device = get_device(cfg.device)
    if custom_npz:
        G, node_to_comm, nodes, custom_tokens = _load_custom_npz(custom_npz)
        log(f"Loaded custom dataset from {custom_npz} | nodes {len(nodes)} | edges {G.number_of_edges()}")
    else:
        G, node_to_comm = load_demo_graph(cfg.graph_name, seed=cfg.seed, cache_dir=cfg.data_dir)
        nodes = list(G.nodes())
        custom_tokens = None

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    adj: List[Set[int]] = [set() for _ in nodes]
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            ui = node_to_idx[u]
            vi = node_to_idx[v]
            adj[ui].add(vi)
            adj[vi].add(ui)

    if cfg.embeddings is not None:
        emb = load_embeddings(cfg.embeddings, nodes).to(device)
    else:
        model = load_model(cfg.checkpoint_path, token_dim(G), cfg, device)
        with torch.no_grad():
            tokens = None
            if cfg.tokens_path:
                tokens = load_tokens(cfg.tokens_path, nodes).to(device)
            elif custom_tokens is not None:
                tokens = torch.from_numpy(custom_tokens).float().to(device)
            if tokens is None:
                tok = RandomWalkTokenizer(
                    G,
                    walk_length=cfg.walk_length or 1,
                    walk_lengths=cfg.walk_lengths,
                    num_walks=cfg.num_walks_per_node,
                    seed=cfg.seed,
                    restart_p=getattr(cfg, "restart_p", 0.0),
                    jaccard_bias=getattr(cfg, "jaccard_bias", False),
                )
                tokens = tok.batch_tokenize(nodes).to(device)
            emb = model(tokens)  # (N, D)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1).cpu().numpy()

    # Anchor queries:
    # - anchors_file provided: use listed anchors (filtered to existing nodes)
    # - no query: run all community anchors (if labels available)
    # - query provided: single anchor
    anchors = []
    if anchors_file:
        anchors = _load_anchor_list(anchors_file, nodes)
        if not anchors:
            raise ValueError(f"No valid anchors found in {anchors_file}")
    elif query is None:
        comm_ids = sorted(set(node_to_comm.values()))
        comm_to_nodes = {c: sorted([n for n, cid in node_to_comm.items() if cid == c]) for c in comm_ids}
        anchors = [comm_to_nodes[c][0] for c in comm_ids if comm_to_nodes[c]]
        if not anchors:
            raise ValueError("No labels available to select anchors; specify --query.")
    else:
        if query not in nodes:
            raise ValueError(f"Query node {query} not found in graph.")
        anchors = [query]

    # Hyperparameters for the new search (pull from cfg if present)
    k_e_init = getattr(cfg, "k_e_init", 200)
    k_e_max = getattr(cfg, "k_e_max", 2000)
    k_e_step = getattr(cfg, "k_e_step", 200)
    k_min = getattr(cfg, "k_min", 30)
    alpha = getattr(cfg, "alpha", 1.0)
    beta = getattr(cfg, "beta", 0.05)
    lam = getattr(cfg, "lam", 0.05)
    min_size = getattr(cfg, "min_size", 5)

    results = []
    for q_node in anchors:
        q_idx = node_to_idx[q_node]
        pred_idx = search_community_for_query(
            q_idx,
            emb,
            adj,
            k_e_init=k_e_init,
            k_e_max=k_e_max,
            k_e_step=k_e_step,
            k_min=k_min,
            min_size=min_size,
            alpha=alpha,
            beta=beta,
            lam=lam,
        )
        pred_nodes = [nodes[i] for i in pred_idx]
        gt_metrics = {}
        if node_to_comm:
            gt_metrics = evaluate_query(q_node, pred_nodes, node_to_comm)
        results.append(gt_metrics)
        if query is not None:
            log(f"Query node: {q_node}")
            log(f"Predicted community (size {len(pred_nodes)}): {pred_nodes}")
            if gt_metrics:
                log(
                    "GT overlap: "
                    f"prec {gt_metrics['precision']:.3f} | rec {gt_metrics['recall']:.3f} | "
                    f"f1 {gt_metrics['f1']:.3f} | iou {gt_metrics['iou']:.3f} | "
                    f"gt_size {gt_metrics['gt_size']}"
                )
    if query is None and node_to_comm and results:
        prec = float(np.mean([m["precision"] for m in results if m]))
        rec = float(np.mean([m["recall"] for m in results if m]))
        f1 = float(np.mean([m["f1"] for m in results if m]))
        iou = float(np.mean([m["iou"] for m in results if m]))
        log(f"Anchors: {len(results)} | Avg prec {prec:.4f} | rec {rec:.4f} | f1 {f1:.4f} | iou {iou:.4f}")


def _sanity_check():
    """
    Lightweight sanity check: two cliques with a single bridge edge.
    Embeddings aligned to clusters; verifies connectivity and query inclusion.
    """
    G = nx.Graph()
    cluster_a = list(range(5))
    cluster_b = list(range(5, 10))
    for u in cluster_a:
        for v in cluster_a:
            if u < v:
                G.add_edge(u, v)
    for u in cluster_b:
        for v in cluster_b:
            if u < v:
                G.add_edge(u, v)
    G.add_edge(4, 5)  # bridge

    nodes = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    adj: List[Set[int]] = [set() for _ in nodes]
    for u, v in G.edges():
        ui = node_to_idx[u]
        vi = node_to_idx[v]
        adj[ui].add(vi)
        adj[vi].add(ui)

    emb = np.zeros((len(nodes), 2), dtype=np.float32)
    for n in cluster_a:
        emb[node_to_idx[n]] = np.array([1.0, 0.0], dtype=np.float32)
    for n in cluster_b:
        emb[node_to_idx[n]] = np.array([0.0, 1.0], dtype=np.float32)

    q_node = 1
    q_idx = node_to_idx[q_node]
    pred_idx = search_community_for_query(
        q_idx,
        emb,
        adj,
        k_e_init=5,
        k_e_max=10,
        k_e_step=2,
        k_min=3,
        min_size=2,
        alpha=1.0,
        beta=0.1,
        lam=0.1,
    )
    pred_set = set(pred_idx)
    assert q_idx in pred_set, "Query node missing from result."
    comp = _component_containing_q(q_idx, pred_set, adj)
    assert comp == pred_set, "Result must remain connected."
    log(f"Sanity check passed. Predicted community (idx): {pred_idx}")


def main():
    parser = argparse.ArgumentParser(description="Community search with trained Mamba encoder.")
    parser.add_argument("--graph", default="toy", help="Graph name (must have cache at data_dir/<graph>_seed<seed>.pkl)")
    parser.add_argument("--checkpoint", default="checkpoints/mamba_comm.pt")
    parser.add_argument("--query", type=str, default=None, help="Query node id. If omitted, run all community anchors.")
    parser.add_argument("--data-dir", default=None, help="Directory containing cached graphs.")
    parser.add_argument("--seed", type=int, default=None, help="Graph seed for cached data.")
    parser.add_argument("--walk-length", type=int, default=None, help="Random walk length at search.")
    parser.add_argument("--walk-lengths", type=str, default=None, help="Comma-separated walk lengths for multi-scale.")
    parser.add_argument("--num-walks-per-node", type=int, default=None, help="Number of walks per node at search.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Force device (cpu/cuda).")
    parser.add_argument("--min-size", type=int, default=None, help="Minimum community size during peeling.")
    parser.add_argument("--embeddings", type=str, default=None, help="Optional path to precomputed embeddings (npz/npy).")
    parser.add_argument("--tokens-path", type=str, default=None, help="Optional path to precomputed tokens (npz/pkl).")
    parser.add_argument("--custom-npz", type=str, default=None, help="Custom dataset npz with nodes/labels/edges (and optional tokens).")
    parser.add_argument("--k-e-init", type=int, default=200, help="Initial top-k for embedding-guided candidate subgraph.")
    parser.add_argument("--k-e-max", type=int, default=2000, help="Max top-k for embedding-guided candidate subgraph.")
    parser.add_argument("--k-e-step", type=int, default=200, help="Step size when growing candidate subgraph.")
    parser.add_argument("--k-min", type=int, default=30, help="Minimum candidate size to stop Stage A (else fallback).")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for embedding similarity in peeling keep-score.")
    parser.add_argument("--beta", type=float, default=0.05, help="Weight for internal degree in peeling keep-score/objective.")
    parser.add_argument("--lam", type=float, default=0.05, help="Weight for outbound edges penalty in peeling/objective.")
    parser.add_argument("--sanity-check", action="store_true", help="Run a lightweight sanity check and exit.")
    parser.add_argument("--anchors-file", type=str, default=None, help="Optional text file with anchor node ids (one per line).")
    args = parser.parse_args()

    if args.sanity_check:
        _sanity_check()
        return

    cfg = SearchConfig(graph_name=args.graph, checkpoint_path=args.checkpoint)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.device:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    if args.walk_length is not None:
        cfg.walk_length = args.walk_length
    if args.walk_lengths:
        cfg.walk_lengths = [int(x) for x in args.walk_lengths.split(",") if x.strip()]
    if args.num_walks_per_node is not None:
        cfg.num_walks_per_node = args.num_walks_per_node
    if args.min_size is not None:
        cfg.min_size = args.min_size
    if args.embeddings:
        cfg.embeddings = args.embeddings
    if args.tokens_path:
        cfg.tokens_path = args.tokens_path
    cfg.k_e_init = args.k_e_init
    cfg.k_e_max = args.k_e_max
    cfg.k_e_step = args.k_e_step
    cfg.k_min = args.k_min
    cfg.alpha = args.alpha
    cfg.beta = args.beta
    cfg.lam = args.lam

    # Convert query node type to int when possible
    if args.query is None:
        query_node = None
    else:
        try:
            query_node = int(args.query)
        except ValueError:
            query_node = args.query

    search(cfg, query_node, custom_npz=args.custom_npz, anchors_file=args.anchors_file)


if __name__ == "__main__":
    main()
