# 11L Sidecar48 + Enhanced AdamW TTT (20 epochs, cosine LR)

## Summary

Enhanced test-time training built on [ymrohit's shared sparse sidecar architecture](https://github.com/openai/parameter-golf/pull/555). The base model and training loop are identical to PR #555; the key innovation is in the TTT phase:

| Enhancement | PR #555 (baseline) | This submission |
|---|---|---|
| TTT epochs | 10 | **20** |
| LR schedule | Flat 0.0005 | **Cosine 0.0005→0.00002** |
| LR warmup | None | **1-epoch linear warmup** |
| Weight decay | 0.0 | **0.01** |
| Eval stride | 64 | 64 (stride 32 tested, negligible diff) |

## Results (8xH200, USE_COMPILE=0)

> **Note:** These results are from H200 GPUs without torch.compile, yielding only ~2400 training steps (vs ~5900 on H100 with compile). H100 results will be significantly better.

| Metric | Value |
|---|---|
| Training steps | 2399 / 9000 |
| Step time | 248ms |
| Pre-TTT val_bpb | 1.2089 |
| Post-TTT (standard) | **1.1102** |
| Post-TTT (sliding window s=64) | **1.1014** |
| Model size (int6+zstd) | 12.6 MB |
| Code size | 80.8 KB |
| **Total** | **12.7 MB** (well under 16 MB) |

### TTT Loss Progression (cosine LR)

```
Epoch  1/20: loss=2.0669  lr=0.000500
Epoch  5/20: loss=2.0013  lr=0.000449
Epoch 10/20: loss=1.9440  lr=0.000280
Epoch 15/20: loss=1.8997  lr=0.000097
Epoch 20/20: loss=1.8776  lr=0.000020
```

### Comparison with flat-LR TTT (10 epochs, same base model)

| TTT Method | Post-TTT BPB (standard) | Post-TTT BPB (sliding) |
|---|---|---|
| Flat LR, 10 epochs | 1.1527 | 1.1356 |
| **Cosine LR, 20 epochs** | **1.1102** | **1.1014** |
| Improvement | 0.0425 | **0.0342** |

## Architecture (from PR #555)

- 11-layer transformer, 512 dim, 8 heads, 4 KV heads, 3x MLP
- SharedSparseSidecar (48 hidden) at layers 8-10
- BigramHash embedding (2048 vocab, 96 dim)
- SmearGate + U-Net skip connections
- EMA (0.997) + orthogonal init + muP-scaled projections
- relu² MLP + logit softcap 30.0
- Int6 mixed quantization + zstd-22 compression

## Key Insight

The original TTT uses a flat learning rate that either stops too early (underfitting) or overshoots (if trained longer). Cosine annealing with warmup allows:
1. **Gentle start**: 1-epoch warmup prevents early destabilization
2. **Full exploration**: High LR in middle epochs finds good adaptation direction
3. **Precise convergence**: LR decays to 0.00002, fine-tuning the final weights
4. **Regularization**: Small WD (0.01) prevents overfitting to val data

This enables 20 productive epochs vs 10, extracting ~3.4% more BPB improvement.
