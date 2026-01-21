# CSMamba
CSMamba: Unsupervised Community Search via State Space Models

This project learns node embeddings from structural random-walk sequences and performs community search.

## Prepare data
Generate and cache graphs (only once per name/seed):
```
python demo_dataset.py --name toy --seed 0 --data-dir data --generate
python demo_dataset.py --name lfr --seed 42 --data-dir data --generate --n 500 --mu 0.2
# Optional: precompute tokens (fixed random walks) and embed into pkl (omit --output to skip npz)
python generate_tokens.py --graph lfr --seed 42 --data-dir data --n 500 --mu 0.2 --walk-lengths 2,3,4 --num-walks-per-node 6 --restart-p 0.1 --jaccard-bias --into-pkl
```

## Training (common options)
- Walks: `--walk-length` / `--walk-lengths`, `--num-walks-per-node`, `--restart-p`, `--jaccard-bias`
- Masks: `--pos-hops`, `--neg-hops`, `--temperature`, `--neg-weight-scale`
- Embeddings auto-saved: `--embedding-path` (npz with `emb` and `nodes`)

### Example (lfr_n500_mu02_w246_tokens.npz)
```
python train.py \
    --custom-npz data/lfr_n500_mu02_w246_tokens.npz \
    --seed 42 \
    --pos-hops 2 \
    --neg-hops 3 \
    --temperature 0.20 \
    --epochs 80 \
    --neg-weight-scale 2.0 \
    --batch-size 128 \
    --embedding-path checkpoints/lfr_n500_mu02_w246_emb.npz \
    --checkpoint checkpoints/lfr_n500_mu02_w246.pt
```

## Search / Evaluation
- Inputs: `--embeddings` (recommended) or model checkpoint; `--min-size/--max-size` prefix range; `--ppr-weight` (0 to disable); `--use-struct-rerank/--no-struct-rerank`, `--struct-rerank-alpha`, `--struct-rerank-topm`
- No `--query`: evaluate all community anchors (first node per community) and report avg precision/recall/F1/IoU.

### Example (lfr_n500_mu02_w246_tokens.npz)
python search.py \
    --custom-npz data/lfr_n500_mu02_w246_tokens.npz \
    --embeddings checkpoints/lfr_n500_mu02_w246_emb.npz \
    --k-e-init 120 --k-e-step 60 --k-e-max 650 --k-min 50 \
    --alpha 1.0 --beta 0.03 --lam 0.26 \
    --min-size 20

### Others regenerate_tokens_from_npz
 python scripts/regenerate_tokens_from_npz.py \
    --input data/lfr_n500_mu02_w246_tokens.npz \
    --output data/lfr_n500_mu02_w246_tokens_new.npz \
    --walk-lengths 2,4,6 \
    --num-walks-per-node 6 \
    --restart-p 0.1 \
    --jaccard-bias \
    --seed 42
