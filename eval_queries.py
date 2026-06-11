"""Evaluate Mamba model on per-community anchor queries (first node per community)."""

from __future__ import annotations

import argparse
import numpy as np
import torch

from demo_dataset import load_demo_graph
from models import MambaCommunityModel
from rw_tokenizer import RandomWalkTokenizer
from metrics import sweep_cut_best, community_pr_re_f1
from graph_data import token_dim
from utils import get_device
from pathlib import Path


def load_model(checkpoint_path: str, token_dim_size: int, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = MambaCommunityModel(
        token_dim_size,
        cfg.get("token_hidden_dim", 64),
        cfg.get("d_model", 96),
        cfg.get("d_state", 16),
        cfg.get("d_conv", 4),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    walk_length = ckpt.get("walk_length", cfg.get("walk_length", 6))
    walk_lengths = ckpt.get("walk_lengths") or cfg.get("walk_lengths")
    num_walks = ckpt.get("num_walks", cfg.get("num_walks_per_node", 8))
    return model, walk_length, walk_lengths, num_walks


def load_embeddings(path: str, nodes: list) -> torch.Tensor:
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


def evaluate(cfg_graph: str, seed: int, ckpt_path: str, min_size: int = 3, max_size: int | None = None, density_weight: float = 1.0, embeddings_path: str | None = None):
    device = get_device()
    G, node_to_comm = load_demo_graph(cfg_graph, seed=seed, cache_dir="data")
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}

    if embeddings_path:
        emb = load_embeddings(embeddings_path, nodes).to(device)
    else:
        model, walk_length, walk_lengths, num_walks = load_model(ckpt_path, token_dim(G), device)
        tok = RandomWalkTokenizer(G, walk_length=walk_length, walk_lengths=walk_lengths, num_walks=num_walks, seed=seed)

        with torch.no_grad():
            emb = model(tok.batch_tokenize(nodes).to(device)).cpu()
    sim = emb @ emb.t()

    # Build anchor list: first node per community (sorted)
    comm_ids = sorted(set(node_to_comm.values()))
    comm_to_nodes = {c: sorted([n for n, cid in node_to_comm.items() if cid == c]) for c in comm_ids}
    anchors = [comm_to_nodes[c][0] for c in comm_ids if comm_to_nodes[c]]

    metrics = []
    for q in anchors:
        q_idx = idx[q]
        sims = sim[q_idx]
        ranked = sorted(nodes, key=lambda n: float(sims[idx[n]]), reverse=True)
        best_set, history = sweep_cut_best(G, ranked, min_size=min_size, max_size=max_size, density_weight=density_weight)
        gt = community_pr_re_f1(best_set, node_to_comm, q)
        if gt:
            metrics.append(gt)

    if not metrics:
        print("No metrics computed.")
        return
    prec = np.mean([m["precision"] for m in metrics])
    rec = np.mean([m["recall"] for m in metrics])
    f1 = np.mean([m["f1"] for m in metrics])
    iou = np.mean([m["iou"] for m in metrics])
    print(f"Anchors: {len(metrics)}")
    print(f"Avg precision {prec:.4f} | recall {rec:.4f} | f1 {f1:.4f} | iou {iou:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate Mamba model on per-community anchors.")
    ap.add_argument("--graph", default="lfr", choices=["toy", "lfr"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", default="checkpoints/mamba_comm.pt")
    ap.add_argument("--embeddings", default=None, help="Optional path to precomputed embeddings (npz/npy).")
    ap.add_argument("--min-size", type=int, default=3)
    ap.add_argument("--max-size", type=int, default=None)
    ap.add_argument("--density-weight", type=float, default=1.0)
    args = ap.parse_args()
    evaluate(
        args.graph,
        args.seed,
        args.checkpoint,
        min_size=args.min_size,
        max_size=args.max_size,
        density_weight=args.density_weight,
        embeddings_path=args.embeddings,
    )


if __name__ == "__main__":
    main()
 