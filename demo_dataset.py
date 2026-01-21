"""Demo graph generators for structural community experiments."""

from __future__ import annotations

import argparse
import os
import pickle
import random
from typing import Dict, Tuple

import networkx as nx


def _toy_graph(seed: int) -> Tuple[nx.Graph, Dict[int, int]]:
    rng = random.Random(seed)
    G = nx.Graph()

    # Two dense blobs plus a small triad connected by a bridge path.
    blob1 = list(range(0, 8))
    blob2 = list(range(8, 16))
    triad = [16, 17, 18]
    bridge_nodes = [7, 19, 8]  # connect blob1 -> bridge -> blob2

    for nodes in (blob1, blob2):
        for u in nodes:
            for v in nodes:
                if u < v and rng.random() < 0.65:
                    G.add_edge(u, v)

    # Triad internal edges
    G.add_edges_from([(16, 17), (17, 18), (16, 18)])

    # Bridge path with a possible detour edge to make it slightly noisy
    G.add_edge(bridge_nodes[0], bridge_nodes[1])
    G.add_edge(bridge_nodes[1], bridge_nodes[2])
    if rng.random() < 0.3:
        G.add_edge(bridge_nodes[0], bridge_nodes[2])

    # Connect triad to blob2 with a single anchor
    G.add_edge(18, 9)

    # Assign communities
    node_to_comm: Dict[int, int] = {}
    for n in blob1:
        node_to_comm[n] = 0
    for n in blob2:
        node_to_comm[n] = 1
    for n in triad:
        node_to_comm[n] = 2
    node_to_comm[19] = 3  # bridge node as its own structural role

    return G, node_to_comm


def _lfr_graph(seed: int, n: int = 250, mu: float = 0.08) -> Tuple[nx.Graph, Dict[int, int]]:
    # Modest-size LFR for quick experiments; n/mu can be overridden
    tau1 = 3  # degree distribution exponent
    tau2 = 1.5  # community size exponent
    G = nx.generators.community.LFR_benchmark_graph(
        n,
        tau1,
        tau2,
        mu,
        average_degree=8,  # specify only one of min_degree/average_degree
        max_degree=30,
        min_community=10,
        max_community=40,
        seed=seed,
    )

    # LFR returns a Graph with frozenset communities per node
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

    # Convert to simple undirected graph without self loops
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G, node_to_comm


def cache_path_for(name: str, seed: int, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"{name}_seed{seed}.pkl")


def generate_and_cache(name: str = "toy", seed: int = 0, cache_dir: str = "data", **kwargs) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    name = name.lower()
    if name == "toy":
        G, node_to_comm = _toy_graph(seed)
    elif name == "lfr":
        G, node_to_comm = _lfr_graph(seed, **kwargs)
    else:
        raise ValueError(f"Unknown demo graph name: {name}")
    path = cache_path_for(name, seed, cache_dir)
    with open(path, "wb") as f:
        pickle.dump({"graph": G, "node_to_comm": node_to_comm}, f)
    return path


def load_demo_graph(name: str = "toy", seed: int = 0, cache_dir: str = "data") -> Tuple[nx.Graph, Dict[int, int]]:
    """
    name: "toy" or "lfr"
    returns: G (networkx.Graph), node_to_comm (dict: node -> int)
    """
    path = cache_path_for(name, seed, cache_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Graph cache not found at {path}. Generate it first via:\n"
            f"  python demo_dataset.py --name {name} --seed {seed} --data-dir {cache_dir} --generate"
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["graph"], data["node_to_comm"]


def main():
    parser = argparse.ArgumentParser(description="Generate and cache demo graphs.")
    parser.add_argument("--name", default="toy", choices=["toy", "lfr"], help="Which demo graph to generate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--generate", action="store_true", help="Generate and write graph cache.")
    parser.add_argument("--n", type=int, default=250, help="Number of nodes for LFR.")
    parser.add_argument("--mu", type=float, default=0.08, help="Mixing parameter for LFR.")
    args = parser.parse_args()

    if args.generate:
        extra = {}
        if args.name == "lfr":
            extra = {"n": args.n, "mu": args.mu}
        path = generate_and_cache(args.name, args.seed, args.data_dir, **extra)
        print(f"Saved graph cache to {path}")
    else:
        print("Nothing to do. Use --generate to create cached graphs.")


if __name__ == "__main__":
    main()
