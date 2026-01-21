"""Contrastive objectives for community-consistent embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_info_nce(
    emb: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    temperature: float = 0.2,
    pos_weight: torch.Tensor | None = None,
    neg_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    emb: (N, D)
    pos_mask, neg_mask: (N, N) boolean/float masks
    pos_weight, neg_weight: (N, N) optional non-negative weights; defaults to 1 where mask is True.
    """
    sim = emb @ emb.t() / temperature  # cosine sim already normalized
    sim = sim - torch.diag(torch.diag(sim))  # drop self-sim

    pos_logits = sim.masked_fill(~pos_mask, float("-inf"))
    neg_logits = sim.masked_fill(~neg_mask, float("-inf"))

    if pos_weight is None:
        pos_weight = torch.ones_like(pos_logits)
    if neg_weight is None:
        neg_weight = torch.ones_like(neg_logits)
    pos_weight = pos_weight * pos_mask
    neg_weight = neg_weight * neg_mask

    # For numerical stability: replace -inf rows with very negative numbers
    pos_logits = torch.where(torch.isfinite(pos_logits), pos_logits, torch.full_like(pos_logits, -1e9))
    neg_logits = torch.where(torch.isfinite(neg_logits), neg_logits, torch.full_like(neg_logits, -1e9))

    pos_max = torch.max(pos_logits, dim=1, keepdim=True).values
    neg_max = torch.max(neg_logits, dim=1, keepdim=True).values
    max_all = torch.maximum(pos_max, neg_max)

    pos_exp = torch.exp(pos_logits - max_all) * pos_weight
    neg_exp = torch.exp(neg_logits - max_all) * neg_weight

    pos_sum = pos_exp.sum(dim=1) + 1e-8
    denom = pos_sum + neg_exp.sum(dim=1) + 1e-8

    loss = -torch.log(pos_sum / denom)
    loss = torch.where(pos_sum > 1e-7, loss, torch.zeros_like(loss))
    return loss.mean()


def variance_regularizer(emb: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Discourage dimensional collapse."""
    std = torch.sqrt(emb.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


def structural_contrastive_loss(
    embeddings: torch.Tensor,
    anchor_ids: list[int],
    struct_pos_dict: dict,
    struct_neg_dict: dict,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    For each anchor i, find hardest pos/neg and apply margin loss.
    """
    if not anchor_ids:
        return torch.tensor(0.0, device=embeddings.device)
    loss_list = []
    emb = F.normalize(embeddings, p=2, dim=1)
    for a in anchor_ids:
        pos_nodes = struct_pos_dict.get(a, [])
        neg_nodes = struct_neg_dict.get(a, [])
        if not pos_nodes or not neg_nodes:
            continue
        a_emb = emb[a : a + 1]  # assumes anchors indexed by position in embeddings
        pos_emb = emb[pos_nodes]
        neg_emb = emb[neg_nodes]
        pos_sim = torch.matmul(a_emb, pos_emb.t()).squeeze(0)
        neg_sim = torch.matmul(a_emb, neg_emb.t()).squeeze(0)
        hard_pos = pos_sim.min()
        hard_neg = neg_sim.max()
        loss_list.append(F.relu(hard_neg - hard_pos + margin))
    if not loss_list:
        return torch.tensor(0.0, device=embeddings.device)
    return torch.stack(loss_list).mean()


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    anchor_ids: list[int],
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Labels: shape (N,), -1 for unlabeled.
    For each anchor with label, use same-label as pos, different-label as neg.
    """
    if not anchor_ids:
        return torch.tensor(0.0, device=embeddings.device)
    emb = F.normalize(embeddings, p=2, dim=1)
    loss_list = []
    for a in anchor_ids:
        la = labels[a].item()
        if la < 0:
            continue
        pos_idx = (labels == la).nonzero(as_tuple=True)[0]
        pos_idx = pos_idx[pos_idx != a]
        neg_idx = (labels != la).nonzero(as_tuple=True)[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        a_emb = emb[a : a + 1]
        pos_sim = torch.matmul(a_emb, emb[pos_idx].t()).squeeze(0)
        neg_sim = torch.matmul(a_emb, emb[neg_idx].t()).squeeze(0)
        hard_pos = pos_sim.min()
        hard_neg = neg_sim.max()
        loss_list.append(F.relu(hard_neg - hard_pos + margin))
    if not loss_list:
        return torch.tensor(0.0, device=embeddings.device)
    return torch.stack(loss_list).mean()


def edge_contrastive_loss(
    embeddings: torch.Tensor,
    pos_edges: list[tuple[int, int]],
    neg_edges: list[tuple[int, int]],
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Contrast edges: positives are existing edges, negatives are sampled non-edges.
    """
    if not pos_edges or not neg_edges:
        return torch.tensor(0.0, device=embeddings.device)
    emb = F.normalize(embeddings, p=2, dim=1)
    loss_list = []
    # For simplicity, iterate positives and pick hardest neg per source
    neg_by_src = {}
    for u, v in neg_edges:
        neg_by_src.setdefault(u, []).append(v)
    for u, v in pos_edges:
        u_emb = emb[u : u + 1]
        pos_sim = torch.matmul(u_emb, emb[v : v + 1].t()).squeeze()
        neg_candidates = neg_by_src.get(u, [])
        if not neg_candidates:
            continue
        neg_emb = emb[neg_candidates]
        neg_sim = torch.matmul(u_emb, neg_emb.t()).squeeze(0)
        hard_neg = neg_sim.max()
        loss_list.append(F.relu(hard_neg - pos_sim + margin))
    if not loss_list:
        return torch.tensor(0.0, device=embeddings.device)
    return torch.stack(loss_list).mean()
