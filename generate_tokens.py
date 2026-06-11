"""Precompute random-walk tokens for all nodes and save to npz."""

from __future__ import annotations

import argparse
import numpy as np
import torch

from demo_dataset import load_demo_graph
from rw_tokenizer import RandomWalkTokenizer
from graph_data import token_dim
from utils import log


def main():
    ap = argparse.ArgumentParser(description="Precompute random-walk tokens for all nodes.")
    ap.add_argument("--graph", required=True, help="Graph name (must have cache at data_dir/<graph>_seed<seed>.pkl)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=None, help="Optional node count (for generation if missing cache).")
    ap.add_argument("--mu", type=float, default=None, help="Optional mu (for generation if missing cache).")
    ap.add_argument("--walk-length", type=int, default=8)
    ap.add_argument("--walk-lengths", type=str, default=None, help="Comma-separated lengths for multi-scale.")
    ap.add_argument("--num-walks-per-node", type=int, default=4)
    ap.add_argument("--restart-p", type=float, default=0.0)
    ap.add_argument("--jaccard-bias", action="store_true")
    ap.add_argument("--output", default=None, help="Optional npz output (if omitted, only pkl is updated).")
    ap.add_argument("--into-pkl", action="store_true", help="Save tokens into the dataset pkl.")
    args = ap.parse_args()

    walk_lengths = None
    if args.walk_lengths:
        walk_lengths = [int(x) for x in args.walk_lengths.split(",") if x.strip()]

    # If cache missing and n/mu provided, allow generation via demo_dataset defaults
    try:
        G, _ = load_demo_graph(args.graph, seed=args.seed, cache_dir=args.data_dir)
    except FileNotFoundError:
        if args.graph == "lfr" and args.n is not None and args.mu is not None:
            from demo_dataset import generate_and_cache
            generate_and_cache("lfr", args.seed, args.data_dir, n=args.n, mu=args.mu)
            G, _ = load_demo_graph(args.graph, seed=args.seed, cache_dir=args.data_dir)
        else:
            raise
    nodes = list(G.nodes())
    tok = RandomWalkTokenizer(
        G,
        walk_length=args.walk_length,
        walk_lengths=walk_lengths,
        num_walks=args.num_walks_per_node,
        seed=args.seed,
        restart_p=args.restart_p,
        jaccard_bias=args.jaccard_bias,
    )
    log("Generating tokens...")
    with torch.no_grad():
        tokens = tok.batch_tokenize(nodes).numpy()
    meta = {
        "graph": args.graph,
        "seed": args.seed,
        "walk_length": args.walk_length,
        "walk_lengths": walk_lengths,
        "num_walks_per_node": args.num_walks_per_node,
        "restart_p": args.restart_p,
        "jaccard_bias": args.jaccard_bias,
    }
    if args.output:
        np.savez(args.output, tokens=tokens, nodes=np.array(nodes), meta=meta)
        log(f"Saved tokens to {args.output}")
    if args.into_pkl:
        import pickle
        pkl_path = f"{args.data_dir}/{args.graph}_seed{args.seed}.pkl"
        with open(pkl_path, "rb") as f:
            pkg = pickle.load(f)
        pkg["tokens"] = tokens
        pkg["tokens_meta"] = meta
        with open(pkl_path, "wb") as f:
            pickle.dump(pkg, f)
        log(f"Saved tokens into {pkl_path}")


if __name__ == "__main__":
    main()
 