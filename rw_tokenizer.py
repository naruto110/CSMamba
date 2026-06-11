"""Random-walk based structural tokenizer."""

from __future__ import annotations

import random
from typing import Iterable, List

import networkx as nx
import numpy as np
import torch

from graph_data import structural_token


class RandomWalkTokenizer:
    def __init__(
        self,
        G: nx.Graph,
        walk_length: int,
        num_walks: int,
        seed: int = 0,
        walk_lengths: list[int] | None = None,
        restart_p: float = 0.0,
        jaccard_bias: bool = False,
    ):
        self.G = G
        if walk_lengths:
            self.walk_lengths = list(walk_lengths)
        else:
            self.walk_lengths = [walk_length]
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.rng = random.Random(seed)
        # Determine feature dim for separator token
        sample_node = next(iter(G.nodes()))
        self.feat_dim = structural_token(G, sample_node).shape[0]
        self.sep_token = np.zeros((self.feat_dim,), dtype=np.float32)
        self.restart_p = restart_p
        self.jaccard_bias = jaccard_bias
        self.neighbors = {n: list(G.neighbors(n)) for n in G.nodes()}

    def _sample_walk(self, start, length: int) -> List:
        walk = [start]
        current = start
        base_nei = set(self.neighbors.get(start, []))
        for _ in range(length - 1):
            if self.restart_p > 0 and self.rng.random() < self.restart_p:
                current = start
            neighbors = self.neighbors.get(current, [])
            if neighbors:
                if self.jaccard_bias and base_nei:
                    weights = []
                    for nb in neighbors:
                        nei_nb = set(self.neighbors.get(nb, []))
                        inter = len(base_nei & nei_nb)
                        uni = len(base_nei | nei_nb)
                        jac = inter / uni if uni > 0 else 0.0
                        weights.append(jac + 1e-3)
                    s = sum(weights)
                    r = self.rng.random() * s
                    acc = 0.0
                    for nb, wgt in zip(neighbors, weights):
                        acc += wgt
                        if acc >= r:
                            current = nb
                            break
                else:
                    current = self.rng.choice(neighbors)
            walk.append(current)
        return walk

    def tokenize_node(self, node) -> torch.Tensor:
        tokens = []
        for _ in range(self.num_walks):
            parts = []
            for idx, L in enumerate(self.walk_lengths):
                walk = self._sample_walk(node, length=L)
                feats = [structural_token(self.G, n) for n in walk]
                parts.append(np.stack(feats, axis=0))
                if idx != len(self.walk_lengths) - 1:
                    parts.append(np.expand_dims(self.sep_token, axis=0))
            seq = np.concatenate(parts, axis=0)  # (total_L, F)
            tokens.append(seq)
        arr = np.stack(tokens, axis=0)  # (num_walks, total_L, F)
        return torch.from_numpy(arr).float()

    def batch_tokenize(self, nodes: Iterable) -> torch.Tensor:
        tensors = [self.tokenize_node(n) for n in nodes]
        return torch.stack(tensors, dim=0)  # (N, num_walks, total_L, F)
 