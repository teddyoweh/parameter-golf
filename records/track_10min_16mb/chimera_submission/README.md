# Chimera TTT: K-Projection LoRA

## Built on DeepQuant V10b (PR #596, AriaAnima)

One novel innovation applied to the current #1 submission's per-document LoRA TTT:

### K-Projection LoRA (TTT_K_LORA=1)

Standard TTT LoRA in this competition applies LoRA only to Q and V projections. We add LoRA to **K projections** as well, with a conservative 0.3x LR multiplier. Rationale: the key projection determines what information each position broadcasts for attention retrieval. Adapting K alongside Q/V gives the model more expressive per-document specialization at marginal compute cost (K shares the same rank-8 LoRA as Q/V, and GQA means K is only `num_kv_heads * head_dim = 256` output dims).

### Other Changes

- K-projection LR group: 0.3x base_lr (conservative — K is more sensitive than Q/V)

## Reproducibility

```bash
# Requires 8xH100 80GB SXM
DATA_PATH=data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=data/tokenizers/fineweb_1024_bpe.model \
MAX_WALLCLOCK_SECONDS=600 USE_COMPILE=1 \
TTT_K_LORA=1 \
SEED=1337 \
torchrun --nproc_per_node=8 train_gpt.py
```

## Ablation Controls

```bash
# Baseline (reproduce PR #596 exactly)
TTT_K_LORA=0
```

## Technical Details

The diff vs PR #596 is minimal (~20 lines changed):
- `BatchedTTTLoRA` gains optional K-projection LoRA adapters
- `CausalSelfAttention.forward()` accepts `k_delta` parameter
- `Block.forward()` passes K deltas through
- `GPT._run_blocks()` routes K LoRA modules
- `_build_ttt_optimizer()` adds K-projection LR group (0.3x)
- 1 new hyperparameter: `TTT_K_LORA`

Total: 1482 lines (under 1500 limit).

## Results

**Pending re-run.** Previous results included a disallowed min-NLL epoch selection technique (selecting the epoch with lowest per-document loss across multiple TTT epochs). That technique has been removed. New 3-seed benchmarks with K-LoRA only are needed.
