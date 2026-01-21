"""
Regenerate random-walk tokens for an existing dataset NPZ without rebuilding the graph.

Usage:
  python scripts/regenerate_tokens_from_npz.py \
    --input data/lfr_n500_mu02_w246_tokens.npz \
    --output data/lfr_n500_mu02_w246_tokens_new.npz \
    --walk-lengths 2,4,6 \
    --num-walks-per-node 6 \
    --restart-p 0.1 \
    --jaccard-bias \
    --seed 42

The script loads nodes/edges (and labels if present) from the input NPZ,
recomputes tokens with the given random-walk parameters, and saves a new NPZ
with the same nodes/edges/labels plus updated tokens/meta.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import networkx as nx

# Allow running as a standalone script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rw_tokenizer import RandomWalkTokenizer
from utils import log


def parse_args():
    ap = argparse.ArgumentParser(description="Regenerate tokens for an existing NPZ dataset.")
    ap.add_argument("--input", required=True, help="Path to existing npz with nodes/edges/(labels)/tokens.")
    ap.add_argument("--output", required=True, help="Path to write new npz with regenerated tokens.")
    ap.add_argument("--walk-length", type=int, default=None, help="Single walk length (ignored if --walk-lengths set).")
    ap.add_argument("--walk-lengths", type=str, default=None, help="Comma-separated walk lengths for multi-scale.")
    ap.add_argument("--num-walks-per-node", type=int, default=6, help="Walks per node.")
    ap.add_argument("--restart-p", type=float, default=0.0, help="Restart probability for random walk.")
    ap.add_argument("--jaccard-bias", action="store_true", help="Use Jaccard-based neighbor sampling.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for tokenizer.")
    return ap.parse_args()


def main():
    args = parse_args()
    data = np.load(args.input, allow_pickle=True)
    nodes = list(data["nodes"])
    edges = data["edges"]
    labels = data["labels"] if "labels" in data else None

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges.tolist())

    walk_lengths = None
    if args.walk_lengths:
        walk_lengths = [int(x) for x in args.walk_lengths.split(",") if x.strip()]
    else:
        if args.walk_length is not None:
            walk_lengths = [args.walk_length]
        else:
            raise ValueError("Specify --walk-length or --walk-lengths.")

    tok = RandomWalkTokenizer(
        G,
        walk_length=walk_lengths[0],
        walk_lengths=walk_lengths,
        num_walks=args.num_walks_per_node,
        seed=args.seed,
        restart_p=args.restart_p,
        jaccard_bias=args.jaccard_bias,
    )
    log("Generating tokens...")
    tokens = tok.batch_tokenize(nodes).numpy()

    meta = {
        "walk_lengths": walk_lengths,
        "num_walks_per_node": args.num_walks_per_node,
        "restart_p": args.restart_p,
        "jaccard_bias": args.jaccard_bias,
        "seed": args.seed,
        "token_dim": tokens.shape[-1],
    }

    np.savez_compressed(
        args.output,
        tokens=tokens,
        nodes=np.array(nodes),
        edges=edges,
        labels=labels if labels is not None else np.array([]),
        meta=meta,
    )
    log(f"Saved regenerated tokens to {args.output}")


if __name__ == "__main__":
    main()
