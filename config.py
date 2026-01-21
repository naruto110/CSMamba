"""Global configuration for the unsupervised community search demo."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Data
    graph_name: str = "toy"  # "toy" or "lfr"
    seed: int = 42
    data_dir: str = "data"
    device: str | None = None  # "cpu", "cuda", or None for auto

    # Random walk tokenizer
    walk_length: int = 8
    walk_lengths: list[int] | None = None  # if set, use multi-length concatenated walks
    num_walks_per_node: int = 4
    restart_p: float = 0.0  # random walk restart prob to anchor
    jaccard_bias: bool = False  # bias neighbor sampling by Jaccard to anchor

    # Model sizes
    token_hidden_dim: int = 64
    d_model: int = 96
    d_state: int = 16
    d_conv: int = 4
    dropout: float = 0.1

    # Optimization
    lr: float = 3e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    batch_size: int | None = None  # None => full batch

    # Contrastive loss
    temperature: float = 0.2
    pos_hops: int = 2
    neg_hops: int = 4
    loss_symmetric: bool = True
    neg_weight_scale: float = 1.0  # >1 makes negative pairs harsher with hop distance
    use_structure_guided_ns: bool = False
    struct_loss_weight: float = 0.1
    struct_margin: float = 0.2
    pos_top_ratio: float = 0.1
    neg_bottom_ratio: float = 0.5
    jaccard_low_thresh: float = 0.1
    # Optional supervised anchor loss (uses ground-truth communities when available)
    use_supervised_labels: bool = False
    supervised_loss_weight: float = 0.1
    supervised_margin: float = 0.2
    supervised_sample_ratio: float = 0.2  # fraction of nodes per community to sample
    supervised_max_per_comm: int = 10
    # Optional edge-level contrastive loss
    use_edge_contrastive: bool = False
    edge_loss_weight: float = 0.1
    edge_margin: float = 0.2
    num_neg_edges: int = 5  # negatives per positive edge

    # Checkpoint
    checkpoint_dir: str = "checkpoints"
    checkpoint_path: str = "checkpoints/mamba_comm.pt"
    max_hop_mask: int = 6  # cutoff for hop-distance computation
    embedding_path: str = "checkpoints/mamba_comm_embeddings.npz"  # saved embeddings after training
    tokens_path: str | None = None  # optional precomputed tokens npz


@dataclass
class SearchConfig:
    graph_name: str = "toy"
    data_dir: str = "data"
    checkpoint_path: str = "checkpoints/mamba_comm.pt"
    seed: int = 42
    device: str | None = None  # "cpu", "cuda", or None for auto
    walk_length: int | None = None
    walk_lengths: list[int] | None = None  # if set, use multi-length concatenated walks
    num_walks_per_node: int = 6
    min_size: int = 3
    density_weight: float = 1.0
    max_size: int | None = None
    embeddings: str | None = None  # optional path to precomputed embeddings
    tokens_path: str | None = None  # optional path to precomputed tokens
    ppr_weight: float = 0.0  # weight for PPR re-rank; 0 disables
    ppr_alpha: float = 0.85  # restart prob for PPR
    use_struct_rerank: bool = True  # neighbor support re-rank
    struct_rerank_alpha: float = 0.7
    struct_rerank_topm: int = 50
    # Unsupervised cutoff strategy for ranking -> predicted community
    cutoff_mode: str = "sweep"  # "sweep" (conductance/density) or "topk"
    fixed_k: int | None = None  # used when cutoff_mode == "topk"
