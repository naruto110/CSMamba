"""Utility to inspect community separability of trained embeddings."""

from __future__ import annotations

import numpy as np
import torch

from demo_dataset import load_demo_graph
from graph_data import token_dim
from models import MambaCommunityModel
from rw_tokenizer import RandomWalkTokenizer
from utils import get_device
import numpy as np
import networkx as nx
import os


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


def inspect(ckpt_path: str | None = None,
            device_override: str | None = None,
            embeddings_path: str | None = None,
            custom_npz: str | None = None):
    if not embeddings_path and not ckpt_path:
        raise ValueError("Provide --embeddings or --checkpoint.")

    # If embeddings only, we still need graph/labels; require custom_npz
    if embeddings_path and not custom_npz and not ckpt_path:
        raise ValueError("When using --embeddings without checkpoint, please pass --custom-npz for graph/labels.")

    cfg = {}
    graph_name = "lfr"
    seed = 43
    walk_lengths = None
    walk_length = 6
    num_walks = 8
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("config", {})
        graph_name = ckpt.get("graph_name", "lfr")
        seed = cfg.get("seed", 43)
        walk_lengths = ckpt.get("walk_lengths") or cfg.get("walk_lengths")
        walk_length = ckpt.get("walk_length", cfg.get("walk_length", 6))
        num_walks = ckpt.get("num_walks", cfg.get("num_walks_per_node", 8))

    # Load graph and labels
    if custom_npz:
        data = np.load(custom_npz, allow_pickle=True)
        nodes = list(data["nodes"])  # original node ids, keep order
        labels = list(data["labels"])
        edges = data["edges"]
        G = nx.Graph()
        G.add_nodes_from(nodes)
        for u, v in edges:
            if u == v:
                continue
            G.add_edge(int(u), int(v))
        node_to_comm = {int(n): int(c) for n, c in zip(nodes, labels)}
        idx = {n: i for i, n in enumerate(nodes)}
    else:
        G, node_to_comm = load_demo_graph(graph_name, seed=seed, cache_dir=cfg.get("data_dir", "data"))
        nodes = list(G.nodes())
        idx = {n: i for i, n in enumerate(nodes)}

    device = get_device(device_override or cfg.get("device"))

    if embeddings_path:
        emb = load_embeddings(embeddings_path, nodes)
    else:
        model = MambaCommunityModel(
            ckpt.get("token_dim", token_dim(G)),
            cfg.get("token_hidden_dim", 64),
            cfg.get("d_model", 96),
            cfg.get("d_state", 16),
            cfg.get("d_conv", 4),
            dropout=cfg.get("dropout", 0.1),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            tok = RandomWalkTokenizer(
                G,
                walk_length=walk_length,
                walk_lengths=walk_lengths,
                num_walks=num_walks,
                seed=seed,
            )
            emb = model(tok.batch_tokenize(nodes).to(device)).cpu()

    sim = emb @ emb.t()
    comm_ids = sorted(set(node_to_comm.values()))
    intra_sims, inter_sims = [], []
    per = []

    for c in comm_ids:
        members = [n for n in nodes if node_to_comm[n] == c]
        if len(members) < 2:
            continue
        ids = [idx[m] for m in members]
        sub = sim[np.ix_(ids, ids)]
        mask = ~np.eye(len(ids), dtype=bool)
        intra = sub[mask].mean()
        rest = [i for i in range(len(nodes)) if i not in ids]
        inter = sim[np.ix_(ids, rest)].mean() if rest else np.nan
        intra_sims.append(intra)
        inter_sims.append(inter)
        per.append((c, len(ids), float(intra), float(inter), float(intra - inter)))

    print(f"Communities: {len(comm_ids)}")
    if intra_sims:
        print(f"Avg intra sim: {float(np.mean(intra_sims)):.4f}")
    if inter_sims:
        print(f"Avg inter sim: {float(np.mean(inter_sims)):.4f}")
    print("Communities by intra-inter gap (sorted desc):")
    for c, size, intra, inter, gap in sorted(per, key=lambda x: x[4], reverse=True):
        print(f"comm {c} size {size} intra {intra:.3f} inter {inter:.3f} gap {gap:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect trained embeddings community separability.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Force device (cpu/cuda).")
    parser.add_argument("--embeddings", default=None, help="Optional path to precomputed embeddings (npz/npy).")
    parser.add_argument("--custom-npz", default=None, help="Custom npz with nodes/edges/labels (bypass demo cache).")
    args = parser.parse_args()
    inspect(args.checkpoint, device_override=args.device, embeddings_path=args.embeddings, custom_npz=args.custom_npz)
