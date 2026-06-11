#!/usr/bin/env python
"""
Synthetic graph + random-walk token generator for ABCD-like and LFR graphs.
Outputs NPZ with fields tokens/nodes/labels/edges/meta (compatible with Mamba data format in this repo).
"""

from __future__ import annotations

import argparse
import os
import random
from typing import List, Tuple, Dict
from collections import defaultdict

# Set env vars before importing numpy to avoid OpenBLAS shared memory issues
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_MAIN_FREE", "1")

import numpy as np
import networkx as nx
from graph_data import token_dim  # only used for consistency in meta
from rw_tokenizer import RandomWalkTokenizer


# ----------------- Graph Generators -----------------

def sample_community_sizes(n: int, min_size: int = 20) -> List[int]:
    sizes = []
    remaining = n
    while remaining > 0:
        k = int(np.random.zipf(2.0))
        k = max(min_size, k)
        k = min(k, remaining)
        sizes.append(k)
        remaining -= k
    if len(sizes) > 1 and sizes[-1] < min_size // 2:
        sizes[-2] += sizes[-1]
        sizes = sizes[:-1]
    return sizes


def build_abcd_graph(n_nodes: int, avg_deg: float, mu: float, seed: int) -> Tuple[nx.Graph, np.ndarray]:
    """
    ABCD-like generator using per-node degree targets + stub matching for internal/external edges.
    Steps:
      1) Sample community sizes (power-law-ish), assign nodes.
      2) For each node, sample total degree ~ Poisson(avg_deg).
      3) Split into d_in = round((1-mu)*d), d_out = d - d_in.
      4) Wire internal stubs within each community (configuration style).
      5) Wire external stubs across different communities (simple random pairing with swaps).
      6) Optionally nudge mu if mean deviates >0.05 by adding/removing few inter edges.
    """
    rng = np.random.default_rng(seed)
    sizes = sample_community_sizes(n_nodes)
    comm_ids = []
    for cid, sz in enumerate(sizes):
        comm_ids.extend([cid] * sz)
    comm_ids = np.array(comm_ids, dtype=np.int64)
    n = len(comm_ids)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # degrees
    degs = rng.poisson(avg_deg, size=n)
    degs = np.maximum(degs, 1)
    d_in = np.rint((1.0 - mu) * degs).astype(int)
    d_out = degs - d_in

    # community mapping
    comm_to_nodes = defaultdict(list)
    for i, c in enumerate(comm_ids):
        comm_to_nodes[c].append(i)

    # internal wiring
    for cid, nodes in comm_to_nodes.items():
        stubs = []
        for u in nodes:
            stubs.extend([u] * d_in[u])
        rng.shuffle(stubs)
        for i in range(0, len(stubs) - 1, 2):
            u, v = stubs[i], stubs[i + 1]
            if u != v:
                G.add_edge(u, v)

    # external wiring
    out_stubs = [u for u in range(n) for _ in range(d_out[u])]
    rng.shuffle(out_stubs)
    i = 0
    max_attempts = len(out_stubs) * 5
    attempts = 0
    while i < len(out_stubs) - 1 and attempts < max_attempts:
        u = out_stubs[i]
        v = out_stubs[i + 1]
        attempts += 1
        if comm_ids[u] != comm_ids[v] and u != v:
            G.add_edge(u, v)
            i += 2
        else:
            swap_idx = rng.integers(i + 2, len(out_stubs)) if i + 2 < len(out_stubs) else None
            if swap_idx is not None:
                out_stubs[i + 1], out_stubs[swap_idx] = out_stubs[swap_idx], out_stubs[i + 1]
            else:
                i += 2

    G.remove_edges_from(nx.selfloop_edges(G))

    def current_mu_mean():
        mu_c = compute_comm_mu(G, comm_ids)
        vals = [v for v in mu_c.values() if v is not None]
        return float(np.mean(vals)) if vals else 0.0

    mu_mean = current_mu_mean()
    tol = 0.05
    steps = 0
    while abs(mu_mean - mu) > tol and steps < 50:
        if mu_mean < mu:
            u, v = rng.integers(0, n, size=2)
            if u != v and comm_ids[u] != comm_ids[v]:
                G.add_edge(u, v)
        else:
            inter_edges = [(u, v) for u, v in G.edges() if comm_ids[u] != comm_ids[v]]
            if inter_edges:
                u, v = inter_edges[rng.integers(len(inter_edges))]
                G.remove_edge(u, v)
            else:
                break
        G.remove_edges_from(nx.selfloop_edges(G))
        mu_mean = current_mu_mean()
        steps += 1

    return G, comm_ids


def build_lfr_graph(n: int, mu: float, seed: int) -> Tuple[nx.Graph, np.ndarray]:
    """Wrapper around networkx LFR benchmark with defaults from demo_dataset.py."""
    tau1 = 3
    tau2 = 1.5
    G = nx.generators.community.LFR_benchmark_graph(
        n,
        tau1,
        tau2,
        mu,
        average_degree=8,
        max_degree=30,
        min_community=10,
        max_community=40,
        seed=seed,
    )
    node_to_comm: Dict[int, int] = {}
    comm_id = 0
    comm_map = {}
    for node, attr in G.nodes(data=True):
        communities = list(attr["community"])
        if not communities:
            continue
        key = tuple(sorted(communities))
        if key not in comm_map:
            comm_map[key] = comm_id
            comm_id += 1
        node_to_comm[node] = comm_map[key]
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    # Relabel to contiguous ids
    mapping = {old: i for i, old in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, mapping)
    comm_ids = np.array([node_to_comm[old] for old in sorted(node_to_comm.keys())], dtype=np.int64)
    return G, comm_ids


# ----------------- RW Tokenizer -----------------

def compute_comm_mu(G: nx.Graph, comm_ids: np.ndarray):
    m_in = defaultdict(int)
    m_out = defaultdict(int)
    for u, v in G.edges():
        cu = comm_ids[u]
        cv = comm_ids[v]
        if cu == cv:
            m_in[cu] += 1
        else:
            m_out[cu] += 1
            m_out[cv] += 1
    mu_c = {}
    for c in set(comm_ids):
        tot = m_in[c] + m_out[c]
        mu_c[c] = m_out[c] / tot if tot > 0 else None
    return mu_c


def jaccard_scores(G: nx.Graph, anchor, neighbors_map):
    anchor_nei = set(neighbors_map.get(anchor, []))
    scores = {}
    for nb in neighbors_map.get(anchor, []):
        nb_nei = set(neighbors_map.get(nb, []))
        inter = len(anchor_nei & nb_nei)
        uni = len(anchor_nei | nb_nei)
        scores[nb] = inter / uni if uni > 0 else 0.0
    return scores


# ----------------- Main -----------------

def parse_cli():
    ap = argparse.ArgumentParser(description="Build synthetic graphs (ABCD-like or LFR) and RW tokens.")
    ap.add_argument("--output", required=True, help="Output .npz path")
    ap.add_argument("--mode", choices=["abcd", "lfr"], default="abcd", help="Graph generator")
    ap.add_argument("--n-nodes", type=int, default=2000, help="Number of nodes (for ABCD/LFR)")
    ap.add_argument("--avg-degree", type=float, default=8.0, help="Target avg degree (ABCD)")
    ap.add_argument("--mu", type=float, default=0.2, help="Mixing parameter")
    ap.add_argument("--walk-lengths", type=str, default="2,3,4", help="Comma-separated walk lengths")
    ap.add_argument("--num-walks-per-node", type=int, default=6, help="Walks per node per length")
    ap.add_argument("--restart-p", type=float, default=0.1, help="Restart probability")
    ap.add_argument("--jaccard-bias", action="store_true", help="Use Jaccard-based neighbor sampling")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    return ap.parse_args()


def main():
    args = parse_cli()
    random.seed(args.seed)
    np.random.seed(args.seed)
    walk_lengths = [int(x) for x in args.walk_lengths.split(',') if x.strip()]
    print(f"[config] mode={args.mode} n={args.n_nodes} mu={args.mu} avg_deg={args.avg_degree} walk_lengths={walk_lengths}")

    if args.mode == "abcd":
        G, labels = build_abcd_graph(args.n_nodes, args.avg_degree, args.mu, seed=args.seed)
    else:
        G, labels = build_lfr_graph(args.n_nodes, args.mu, seed=args.seed)

    avg_deg = 2 * G.number_of_edges() / max(1, G.number_of_nodes())
    print(f"[graph] nodes={G.number_of_nodes()} edges={G.number_of_edges()} avg_deg={avg_deg:.2f}")
    mu_c = compute_comm_mu(G, labels)
    vals = [v for v in mu_c.values() if v is not None]
    if vals:
        print(
            f"[mixing] communities={len(mu_c)} min={min(vals):.4f} mean={float(np.mean(vals)):.4f} "
            f"median={float(np.median(vals)):.4f} max={max(vals):.4f}"
        )

    # Use structural-token RandomWalkTokenizer to align with existing format (tokens: N x W x L x F)
    nodes_list = sorted(G.nodes())
    tok = RandomWalkTokenizer(
        G,
        walk_length=walk_lengths[0] if walk_lengths else 1,
        walk_lengths=walk_lengths,
        num_walks=args.num_walks_per_node,
        seed=args.seed,
        restart_p=args.restart_p,
        jaccard_bias=args.jaccard_bias,
    )
    tokens = tok.batch_tokenize(nodes_list).numpy()
    print(f"[walks] tokens shape {tokens.shape} (N, W, L, F)")

    nodes = np.array(nodes_list, dtype=np.int64)
    labels = labels[nodes] if isinstance(labels, np.ndarray) else np.array([labels[n] for n in nodes_list], dtype=np.int64)
    edges = np.array([[u, v] for u, v in G.edges()], dtype=object)

    np.savez_compressed(
        args.output,
        tokens=tokens,
        nodes=nodes,
        labels=labels,
        edges=edges,
        meta={
            "mode": args.mode,
            "walk_lengths": walk_lengths,
            "num_walks_per_node": args.num_walks_per_node,
            "restart_p": args.restart_p,
            "jaccard_bias": args.jaccard_bias,
            "avg_degree": args.avg_degree,
            "mu": args.mu,
            "n_nodes": args.n_nodes,
            "token_dim": tokens.shape[-1],
        },
    )
    print(f"[save] wrote {args.output}")


if __name__ == "__main__":
    main()
 