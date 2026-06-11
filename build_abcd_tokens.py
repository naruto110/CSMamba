#!/usr/bin/env python
"""
Generate an ABCD-like synthetic graph, run biased random walks, and save tokens/metadata to NPZ.
Output format follows the Mamba data used in this repo (tokens/nodes/labels/edges fields),
so it can be consumed similarly to amazon_connected_* npz files.
"""

from __future__ import annotations

import argparse
import random
from typing import List, Tuple
from collections import defaultdict

import networkx as nx
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description="Generate ABCD-like graph and random-walk tokens.")
    ap.add_argument("--output", required=True, help="Path to output .npz")
    ap.add_argument("--n-nodes", type=int, default=2000, help="Number of nodes")
    ap.add_argument("--avg-degree", type=float, default=8.0, help="Target average degree")
    ap.add_argument("--mu", type=float, default=0.2, help="Mixing parameter (fraction of cross-community edges)")
    ap.add_argument(
        "--walk-lengths",
        type=str,
        default="2,3,4",
        help="Comma-separated walk lengths (number of nodes per segment).",
    )
    ap.add_argument("--num-walks-per-node", type=int, default=6, help="Walks per node per walk length")
    ap.add_argument("--restart-p", type=float, default=0.1, help="Restart probability during walks")
    ap.add_argument("--jaccard-bias", action="store_true", help="Use Jaccard-based neighbor sampling")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    return ap.parse_args()


def sample_community_sizes(n: int, min_size: int = 20) -> List[int]:
    """Sample power-law-like community sizes that sum to n."""
    sizes = []
    remaining = n
    while remaining > 0:
        # Zipf-like draw; clamp to remaining
        k = int(np.random.zipf(2.0))
        k = max(min_size, k)
        k = min(k, remaining)
        sizes.append(k)
        remaining -= k
    # Merge tiny tail if too many small communities
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

    # 1) degrees
    degs = rng.poisson(avg_deg, size=n)
    degs = np.maximum(degs, 1)  # avoid isolates
    # 2) in/out split
    d_in = np.rint((1.0 - mu) * degs).astype(int)
    d_out = degs - d_in

    # 3) community membership
    comm_to_nodes = defaultdict(list)
    for i, c in enumerate(comm_ids):
        comm_to_nodes[c].append(i)

    # 4) internal wiring (configuration model per community)
    for cid, nodes in comm_to_nodes.items():
        stubs = []
        for u in nodes:
            stubs.extend([u] * d_in[u])
        rng.shuffle(stubs)
        for i in range(0, len(stubs) - 1, 2):
            u, v = stubs[i], stubs[i + 1]
            if u != v:
                G.add_edge(u, v)

    # 5) external wiring (pair stubs from different communities)
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
            # swap with later stub to try different community pairing
            swap_idx = rng.integers(i + 2, len(out_stubs)) if i + 2 < len(out_stubs) else None
            if swap_idx is not None:
                out_stubs[i + 1], out_stubs[swap_idx] = out_stubs[swap_idx], out_stubs[i + 1]
            else:
                i += 2

    # Simple graph (drop self-loops/dups)
    G.remove_edges_from(nx.selfloop_edges(G))

    def current_mu_mean():
        mu_c = compute_comm_mu(G, comm_ids)
        vals = [v for v in mu_c.values() if v is not None]
        return float(np.mean(vals)) if vals else 0.0

    # 6) Nudge mu toward target if off by >0.05
    mu_mean = current_mu_mean()
    tol = 0.05
    steps = 0
    while abs(mu_mean - mu) > tol and steps < 50:
        if mu_mean < mu:
            # add random inter edge
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


def biased_walks(
    G: nx.Graph,
    walk_lengths: List[int],
    num_walks: int,
    restart_p: float,
    use_jaccard: bool,
    seed: int,
):
    rng = random.Random(seed)
    neighbors = {n: list(G.neighbors(n)) for n in G.nodes()}
    max_len = sum(walk_lengths) + (len(walk_lengths) - 1)  # if we insert SEP
    sep_token = -1
    tokens = []  # per node => list of walks arrays
    for u in G.nodes():
        walks_for_node = []
        jac_scores_cache = jaccard_scores(G, u, neighbors) if use_jaccard else None
        for _ in range(num_walks):
            seq_parts = []
            for idx, L in enumerate(walk_lengths):
                walk = [u]
                cur = u
                for _ in range(L - 1):
                    if rng.random() < restart_p:
                        cur = u
                    nei = neighbors.get(cur, [])
                    if nei:
                        if use_jaccard:
                            weights = [jac_scores_cache.get(n, 0.0) + 1e-6 for n in nei]
                            s = sum(weights)
                            if s <= 0:
                                cur = rng.choice(nei)
                            else:
                                r = rng.random() * s
                                acc = 0.0
                                for n, w in zip(nei, weights):
                                    acc += w
                                    if acc >= r:
                                        cur = n
                                        break
                        else:
                            cur = rng.choice(nei)
                    walk.append(cur)
                seq_parts.extend(walk)
                if idx != len(walk_lengths) - 1:
                    seq_parts.append(sep_token)
            # pad to max_len
            if len(seq_parts) < max_len:
                seq_parts.extend([sep_token] * (max_len - len(seq_parts)))
            walks_for_node.append(seq_parts)
        tokens.append(walks_for_node)
    tokens_arr = np.array(tokens, dtype=np.int64)  # (N, num_walks, max_len)
    return tokens_arr, max_len


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    walk_lengths = [int(x) for x in args.walk_lengths.split(',') if x.strip()]
    print(f"[config] n={args.n_nodes} avg_deg={args.avg_degree} mu={args.mu} walk_lengths={walk_lengths}")

    G, comm_ids = build_abcd_graph(args.n_nodes, args.avg_degree, args.mu, seed=args.seed)
    print(f"[graph] nodes={G.number_of_nodes()} edges={G.number_of_edges()} avg_deg={2*G.number_of_edges()/G.number_of_nodes():.2f}")
    mu_c = compute_comm_mu(G, comm_ids)
    vals = [v for v in mu_c.values() if v is not None]
    if vals:
        print(
            f"[mixing] communities={len(mu_c)} min={min(vals):.4f} mean={float(np.mean(vals)):.4f} "
            f"median={float(np.median(vals)):.4f} max={max(vals):.4f}"
        )

    tokens, max_len = biased_walks(
        G,
        walk_lengths=walk_lengths,
        num_walks=args.num_walks_per_node,
        restart_p=args.restart_p,
        use_jaccard=args.jaccard_bias,
        seed=args.seed,
    )
    print(f"[walks] tokens shape {tokens.shape} max_len={max_len} sep_token=-1")

    nodes = np.arange(G.number_of_nodes(), dtype=np.int64)
    labels = comm_ids
    edges = np.array([[u, v] for u, v in G.edges()], dtype=object)

    np.savez_compressed(
        args.output,
        tokens=tokens,
        nodes=nodes,
        labels=labels,
        edges=edges,
        meta={
            "walk_lengths": walk_lengths,
            "num_walks_per_node": args.num_walks_per_node,
            "restart_p": args.restart_p,
            "jaccard_bias": args.jaccard_bias,
            "avg_degree": args.avg_degree,
            "mu": args.mu,
            "n_nodes": args.n_nodes,
            "sep_token": -1,
            "max_len": int(max_len),
        },
    )
    print(f"[save] wrote {args.output}")


if __name__ == "__main__":
    main()
 