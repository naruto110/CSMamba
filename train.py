"""Train the Mamba-based structural community model on a demo graph."""

from __future__ import annotations

import argparse
import math
import os
import random
from typing import List
import torch
import numpy as np

from config import TrainConfig
from demo_dataset import load_demo_graph
from graph_data import shortest_path_matrix, token_dim, compute_hop_dists, boundary_softness, compute_rw_cooc_for_anchor, compute_local_jaccard_for_anchor, get_structural_pos_neg
from losses import contrastive_info_nce, variance_regularizer, structural_contrastive_loss, supervised_contrastive_loss, edge_contrastive_loss
from models import MambaCommunityModel
from rw_tokenizer import RandomWalkTokenizer
from utils import get_device, log, set_seed
import networkx as nx
import numpy as np


def build_masks(G, nodes: List, pos_hops: int, neg_hops: int, max_hop: int, device: torch.device, neg_weight_scale: float = 1.0):
    hop_dists = compute_hop_dists(G, nodes, max_hop=max_hop)
    N = len(nodes)
    pos = torch.zeros((N, N), dtype=torch.bool, device=device)
    neg = torch.zeros((N, N), dtype=torch.bool, device=device)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = hop_dists[i, j]
            if d <= pos_hops:
                pos[i, j] = True
            if d >= neg_hops or d > max_hop:
                neg[i, j] = True
    # Boundary-aware softness: lower weight when either endpoint is ambiguous
    soft = boundary_softness(G, nodes)
    soft_t = torch.from_numpy(soft).to(device)
    internal_strength = 1.0 - soft_t  # internal -> 1, boundary -> smaller
    pair_strength = torch.sqrt(torch.outer(internal_strength, internal_strength))
    pos_w = pair_strength * pos.float()
    neg_w = pair_strength * neg.float()
    if neg_weight_scale != 1.0:
        hop_factor = torch.from_numpy(hop_dists).to(device)
        hop_factor = torch.where(hop_factor > max_hop, torch.full_like(hop_factor, max_hop, dtype=torch.float32), hop_factor)
        hop_factor = hop_factor / float(max_hop + 1e-9)
        neg_w = neg_w * (1.0 + (neg_weight_scale - 1.0) * hop_factor)
    return pos, neg, pos_w, neg_w


def _load_custom_npz(path: str, device: torch.device):
    """
    Load custom dataset npz with fields:
      tokens: (N, W, L, F)
      nodes : (N,)
      labels: (N,) community ids (optional)
      edges : (M, 2) undirected edges among nodes (required)
    """
    data = np.load(path, allow_pickle=True)
    tokens = torch.from_numpy(data["tokens"]).float().to(device)
    nodes = list(data["nodes"])
    labels = data["labels"] if "labels" in data else None
    edges = data["edges"] if "edges" in data else None
    if edges is None:
        raise ValueError("Custom npz must contain 'edges' array of shape (M,2).")
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges.tolist())
    node_to_comm = {}
    if labels is not None and len(labels) == len(nodes):
        node_to_comm = {n: int(lbl) for n, lbl in zip(nodes, labels)}
    return G, nodes, node_to_comm, tokens


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    log(f"Using device: {device}")

    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt_parent = os.path.dirname(cfg.checkpoint_path) or "."
    os.makedirs(ckpt_parent, exist_ok=True)

    tokenizer = None
    precomputed_tokens = None
    # Custom npz pathway: bypass demo graph
    if cfg.tokens_path and cfg.tokens_path.endswith(".npz") and cfg.graph_name == "custom_npz":
        G, nodes, node_to_comm, precomputed_tokens = _load_custom_npz(cfg.tokens_path, device)
        log(f"Loaded custom dataset from {cfg.tokens_path} | nodes {len(nodes)} | edges {G.number_of_edges()}")
    else:
        G, node_to_comm = load_demo_graph(cfg.graph_name, seed=cfg.seed, cache_dir=cfg.data_dir)
        nodes = list(G.nodes())
    def _load_tokens(path):
        import numpy as np
        if path.endswith(".pkl"):
            import pickle
            with open(path, "rb") as f:
                pkg = pickle.load(f)
            if "tokens" not in pkg or "node_to_comm" not in pkg:
                raise ValueError("pkl does not contain tokens or node_to_comm")
            nodes_sorted = sorted(pkg["node_to_comm"].keys())
            return pkg["tokens"], np.array(nodes_sorted)
        elif path.endswith(".npz"):
            data = np.load(path, allow_pickle=True)
            return data["tokens"], data["nodes"]
        else:
            raise ValueError("Unsupported tokens_path format (use .pkl or .npz)")

    if cfg.tokens_path and precomputed_tokens is not None:
        pass  # already loaded via custom npz
    elif cfg.tokens_path:
        tokens_np, nodes_loaded = _load_tokens(cfg.tokens_path)
        precomputed_tokens = torch.from_numpy(tokens_np).float().to(device)
        if len(nodes_loaded) != len(nodes):
            raise ValueError("Precomputed tokens node count mismatch.")
        log(f"Loaded precomputed tokens from {cfg.tokens_path}")
    else:
        tokenizer = RandomWalkTokenizer(
            G,
            walk_length=cfg.walk_length,
            walk_lengths=cfg.walk_lengths,
            num_walks=cfg.num_walks_per_node,
            seed=cfg.seed,
            restart_p=cfg.restart_p,
            jaccard_bias=cfg.jaccard_bias,
        )
    model = MambaCommunityModel(
        token_dim(G),
        cfg.token_hidden_dim,
        cfg.d_model,
        cfg.d_state,
        cfg.d_conv,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    pos_mask, neg_mask, pos_w, neg_w = build_masks(
        G, nodes, cfg.pos_hops, cfg.neg_hops, cfg.max_hop_mask, device, neg_weight_scale=cfg.neg_weight_scale
    )
    # Precompute neighbors list for Jaccard
    neighbors = {n: list(G.neighbors(n)) for n in nodes}
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    # Precompute RW-based pos/neg if enabled
    struct_pos_dict = {}
    struct_neg_dict = {}
    if cfg.use_structure_guided_ns:
        rng = random.Random(cfg.seed)
        base_walk_len = cfg.walk_lengths[0] if cfg.walk_lengths else cfg.walk_length
        for n in nodes:
            walks = []
            for _ in range(cfg.num_walks_per_node):
                w = [n]
                cur = n
                for _ in range(base_walk_len - 1):
                    nei = list(G.neighbors(cur))
                    if nei:
                        cur = rng.choice(nei)
                    w.append(cur)
                walks.append(w)
            rw_scores = compute_rw_cooc_for_anchor(n, walks)
            candidates = list(rw_scores.keys())
            j_scores = compute_local_jaccard_for_anchor(n, candidates, neighbors)
            pos_nodes, neg_nodes = get_structural_pos_neg(
                n,
                rw_scores,
                j_scores,
                pos_top_ratio=cfg.pos_top_ratio,
                neg_bottom_ratio=cfg.neg_bottom_ratio,
                jaccard_low_thresh=cfg.jaccard_low_thresh,
            )
            struct_pos_dict[node_to_idx[n]] = [node_to_idx[x] for x in pos_nodes if x in node_to_idx]
            struct_neg_dict[node_to_idx[n]] = [node_to_idx[x] for x in neg_nodes if x in node_to_idx]
    # Supervised label sampling (uses node_to_comm if available)
    supervised_labels = torch.full((len(nodes),), -1, dtype=torch.long, device=device)
    if cfg.use_supervised_labels and node_to_comm:
        comm_to_nodes = {}
        for n, c in node_to_comm.items():
            if n in node_to_idx:
                comm_to_nodes.setdefault(c, []).append(n)
        rng = random.Random(cfg.seed)
        labeled_ids = []
        for c, members in comm_to_nodes.items():
            members = sorted(members)
            k = max(1, int(len(members) * cfg.supervised_sample_ratio))
            k = min(k, cfg.supervised_max_per_comm, len(members))
            sampled = rng.sample(members, k)
            labeled_ids.extend(sampled)
        for n in labeled_ids:
            supervised_labels[node_to_idx[n]] = node_to_comm[n]
        labeled_anchor_ids = [node_to_idx[n] for n in labeled_ids]
    else:
        labeled_anchor_ids = []
    # Edge contrastive: collect pos edges and sample neg edges
    pos_edges = []
    neg_edges = []
    if cfg.use_edge_contrastive:
        for u, v in G.edges():
            pos_edges.append((node_to_idx[u], node_to_idx[v]))
            pos_edges.append((node_to_idx[v], node_to_idx[u]))
        rng = random.Random(cfg.seed)
        for u in nodes:
            u_idx = node_to_idx[u]
            neighbors_set = set(neighbors[u])
            candidates = [x for x in nodes if x != u and x not in neighbors_set]
            if not candidates:
                continue
            sampled_neg = rng.sample(candidates, min(cfg.num_neg_edges, len(candidates)))
            for v in sampled_neg:
                neg_edges.append((u_idx, node_to_idx[v]))

    batch_size = cfg.batch_size or len(nodes)
    debug_masks = os.getenv("DEBUG_MASK_STATS", "") != ""

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(nodes))
        for start in range(0, len(nodes), batch_size):
            end = start + batch_size
            idx_batch = perm[start:end].tolist()
            batch_nodes = [nodes[i] for i in idx_batch]
            optimizer.zero_grad()

            if precomputed_tokens is not None:
                tokens = precomputed_tokens[idx_batch]
            else:
                tokens = tokenizer.batch_tokenize(batch_nodes).to(device)
            emb = model(tokens)

            # Slice masks/weights for this batch
            pos_b = pos_mask[idx_batch][:, idx_batch]
            neg_b = neg_mask[idx_batch][:, idx_batch]
            pos_w_b = pos_w[idx_batch][:, idx_batch]
            neg_w_b = neg_w[idx_batch][:, idx_batch]

            if debug_masks and epoch == 1 and start == 0:
                log(
                    f"[DEBUG] batch pos_pairs={pos_b.sum().item():.0f} "
                    f"neg_pairs={neg_b.sum().item():.0f} "
                    f"pos_w>0={(pos_w_b>0).sum().item():.0f} "
                    f"neg_w>0={(neg_w_b>0).sum().item():.0f}"
                )

            loss_contrast = contrastive_info_nce(
                emb,
                pos_b,
                neg_b,
                temperature=cfg.temperature,
                pos_weight=pos_w_b,
                neg_weight=neg_w_b,
            )
            loss_var = variance_regularizer(emb)
            loss = loss_contrast + 0.1 * loss_var

            if cfg.use_structure_guided_ns:
                # remap struct pos/neg into local batch indices
                local_map = {g_idx: i for i, g_idx in enumerate(idx_batch)}
                struct_pos_local = {}
                struct_neg_local = {}
                for g_idx in idx_batch:
                    if g_idx not in struct_pos_dict:
                        continue
                    anchor_local = local_map[g_idx]
                    pos_list = [local_map[p] for p in struct_pos_dict.get(g_idx, []) if p in local_map]
                    neg_list = [local_map[n] for n in struct_neg_dict.get(g_idx, []) if n in local_map]
                    struct_pos_local[anchor_local] = pos_list
                    struct_neg_local[anchor_local] = neg_list
                if struct_pos_local or struct_neg_local:
                    struct_loss = structural_contrastive_loss(
                        emb,
                        list(range(len(idx_batch))),
                        struct_pos_local,
                        struct_neg_local,
                        margin=cfg.struct_margin,
                    )
                    loss = loss + cfg.struct_loss_weight * struct_loss

            if cfg.use_supervised_labels and labeled_anchor_ids:
                anchors_local = [local_map[a] for a in labeled_anchor_ids if a in local_map]
                if anchors_local:
                    sup_loss = supervised_contrastive_loss(
                        emb,
                        supervised_labels[idx_batch],
                        anchors_local,
                        margin=cfg.supervised_margin,
                    )
                    loss = loss + cfg.supervised_loss_weight * sup_loss

            if cfg.use_edge_contrastive and pos_edges and neg_edges:
                local_map = {g_idx: i for i, g_idx in enumerate(idx_batch)}
                pos_e_local = [(local_map[u], local_map[v]) for (u, v) in pos_edges if u in local_map and v in local_map]
                neg_e_local = [(local_map[u], local_map[v]) for (u, v) in neg_edges if u in local_map and v in local_map]
                if pos_e_local and neg_e_local:
                    edge_loss = edge_contrastive_loss(
                        emb,
                        pos_e_local,
                        neg_e_local,
                        margin=cfg.edge_margin,
                    )
                    loss = loss + cfg.edge_loss_weight * edge_loss

            loss.backward()
            optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            with torch.no_grad():
                if precomputed_tokens is not None:
                    tokens_full = precomputed_tokens
                else:
                    tokens_full = tokenizer.batch_tokenize(nodes).to(device)
                emb_full = model(tokens_full)
                pos_sim = (emb_full @ emb_full.t() * pos_mask).sum() / (pos_mask.sum() + 1e-8)
                neg_sim = (emb_full @ emb_full.t() * neg_mask).sum() / (neg_mask.sum() + 1e-8)
            log(f"Epoch {epoch:03d} | loss {loss.item():.4f} | pos_sim {pos_sim:.3f} | neg_sim {neg_sim:.3f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "graph_name": cfg.graph_name,
            "token_dim": token_dim(G),
            "walk_length": cfg.walk_length,
            "walk_lengths": cfg.walk_lengths,
            "num_walks": cfg.num_walks_per_node,
        },
        cfg.checkpoint_path,
    )
    log(f"Saved checkpoint to {cfg.checkpoint_path}")

    # Save embeddings to file for faster reuse
    model.eval()
    with torch.no_grad():
        if precomputed_tokens is not None:
            tokens_full = precomputed_tokens
        else:
            tokens_full = tokenizer.batch_tokenize(nodes).to(device)
        emb_full = model(tokens_full).cpu().numpy()
    os.makedirs(os.path.dirname(cfg.embedding_path) or ".", exist_ok=True)
    np.savez(cfg.embedding_path, emb=emb_full, nodes=np.array(nodes))
    log(f"Saved embeddings to {cfg.embedding_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Mamba structural community model.")
    parser.add_argument("--graph", default="toy", help="Graph name (must have cache at data_dir/<graph>_seed<seed>.pkl)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None, help="Directory to cache generated graphs.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to store checkpoints.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Force device (cpu/cuda).")
    parser.add_argument("--pos-hops", type=int, default=None, help="Positive sample hop threshold.")
    parser.add_argument("--neg-hops", type=int, default=None, help="Negative sample hop threshold.")
    parser.add_argument("--temperature", type=float, default=None, help="InfoNCE temperature.")
    parser.add_argument("--walk-length", type=int, default=None, help="Random walk length.")
    parser.add_argument("--walk-lengths", type=str, default=None, help="Comma-separated walk lengths for multi-scale.")
    parser.add_argument("--num-walks-per-node", type=int, default=None, help="Number of walks per node.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay.")
    parser.add_argument("--token-hidden-dim", type=int, default=None, help="Token MLP hidden dim.")
    parser.add_argument("--d-model", type=int, default=None, help="Mamba d_model.")
    parser.add_argument("--d-state", type=int, default=None, help="Mamba d_state.")
    parser.add_argument("--d-conv", type=int, default=None, help="Mamba d_conv.")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout.")
    parser.add_argument("--max-hop-mask", type=int, default=None, help="Cutoff hop for mask computation.")
    parser.add_argument("--embedding-path", type=str, default=None, help="Path to save embeddings (npz).")
    parser.add_argument("--tokens-path", type=str, default=None, help="Path to precomputed tokens (.pkl or .npz).")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size for node subsets.")
    parser.add_argument("--no-loss-symmetric", action="store_true", help="Disable symmetric loss.")
    parser.add_argument("--neg-weight-scale", type=float, default=None, help="Scale >1 to harden negatives with hop distance.")
    parser.add_argument("--use-structure-guided-ns", action="store_true", help="Enable structure-guided NS and loss.")
    parser.add_argument("--struct-loss-weight", type=float, default=None, help="Weight for structural contrastive loss.")
    parser.add_argument("--struct-margin", type=float, default=None, help="Margin for structural contrastive loss.")
    parser.add_argument("--pos-top-ratio", type=float, default=None, help="Top ratio for structural positives.")
    parser.add_argument("--neg-bottom-ratio", type=float, default=None, help="Bottom ratio for structural negatives.")
    parser.add_argument("--jaccard-low-thresh", type=float, default=None, help="Jaccard threshold for structural negatives.")
    parser.add_argument("--use-supervised-labels", action="store_true", help="Use ground-truth communities as supervised anchors.")
    parser.add_argument("--supervised-loss-weight", type=float, default=None, help="Weight for supervised contrastive loss.")
    parser.add_argument("--supervised-margin", type=float, default=None, help="Margin for supervised contrastive loss.")
    parser.add_argument("--supervised-sample-ratio", type=float, default=None, help="Fraction of nodes per community to label.")
    parser.add_argument("--supervised-max-per-comm", type=int, default=None, help="Max labeled nodes per community.")
    parser.add_argument("--restart-p", type=float, default=None, help="Restart probability for random walk.")
    parser.add_argument("--jaccard-bias", action="store_true", help="Enable Jaccard-biased neighbor sampling.")
    parser.add_argument("--use-edge-contrastive", action="store_true", help="Enable edge-level contrastive loss.")
    parser.add_argument("--edge-loss-weight", type=float, default=None, help="Weight for edge contrastive loss.")
    parser.add_argument("--edge-margin", type=float, default=None, help="Margin for edge contrastive loss.")
    parser.add_argument("--num-neg-edges", type=int, default=None, help="Negatives per positive edge for edge loss.")
    parser.add_argument("--custom-npz", type=str, default=None, help="Custom dataset npz (tokens/nodes/labels/edges). Overrides graph name to custom_npz and tokens-path to this file.")
    args = parser.parse_args()

    cfg = TrainConfig(graph_name=args.graph)
    if args.custom_npz:
        cfg.graph_name = "custom_npz"
        cfg.tokens_path = args.custom_npz
    if args.device:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    if args.epochs:
        cfg.max_epochs = args.epochs
    if args.checkpoint:
        cfg.checkpoint_path = args.checkpoint
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.pos_hops is not None:
        cfg.pos_hops = args.pos_hops
    if args.neg_hops is not None:
        cfg.neg_hops = args.neg_hops
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.walk_length is not None:
        cfg.walk_length = args.walk_length
    if args.walk_lengths:
        cfg.walk_lengths = [int(x) for x in args.walk_lengths.split(",") if x.strip()]
    if args.num_walks_per_node is not None:
        cfg.num_walks_per_node = args.num_walks_per_node
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.token_hidden_dim is not None:
        cfg.token_hidden_dim = args.token_hidden_dim
    if args.d_model is not None:
        cfg.d_model = args.d_model
    if args.d_state is not None:
        cfg.d_state = args.d_state
    if args.d_conv is not None:
        cfg.d_conv = args.d_conv
    if args.dropout is not None:
        cfg.dropout = args.dropout
    if args.max_hop_mask is not None:
        cfg.max_hop_mask = args.max_hop_mask
    if args.embedding_path is not None:
        cfg.embedding_path = args.embedding_path
    if args.tokens_path is not None:
        cfg.tokens_path = args.tokens_path
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.no_loss_symmetric:
        cfg.loss_symmetric = False
    if args.neg_weight_scale is not None:
        cfg.neg_weight_scale = args.neg_weight_scale
    if args.use_structure_guided_ns:
        cfg.use_structure_guided_ns = True
    if args.struct_loss_weight is not None:
        cfg.struct_loss_weight = args.struct_loss_weight
    if args.struct_margin is not None:
        cfg.struct_margin = args.struct_margin
    if args.pos_top_ratio is not None:
        cfg.pos_top_ratio = args.pos_top_ratio
    if args.neg_bottom_ratio is not None:
        cfg.neg_bottom_ratio = args.neg_bottom_ratio
    if args.jaccard_low_thresh is not None:
        cfg.jaccard_low_thresh = args.jaccard_low_thresh
    if args.use_supervised_labels:
        cfg.use_supervised_labels = True
    if args.supervised_loss_weight is not None:
        cfg.supervised_loss_weight = args.supervised_loss_weight
    if args.supervised_margin is not None:
        cfg.supervised_margin = args.supervised_margin
    if args.supervised_sample_ratio is not None:
        cfg.supervised_sample_ratio = args.supervised_sample_ratio
    if args.supervised_max_per_comm is not None:
        cfg.supervised_max_per_comm = args.supervised_max_per_comm
    if args.restart_p is not None:
        cfg.restart_p = args.restart_p
    if args.jaccard_bias:
        cfg.jaccard_bias = True
    if args.use_edge_contrastive:
        cfg.use_edge_contrastive = True
    if args.edge_loss_weight is not None:
        cfg.edge_loss_weight = args.edge_loss_weight
    if args.edge_margin is not None:
        cfg.edge_margin = args.edge_margin
    if args.num_neg_edges is not None:
        cfg.num_neg_edges = args.num_neg_edges

    train(cfg)


if __name__ == "__main__":
    main()
 