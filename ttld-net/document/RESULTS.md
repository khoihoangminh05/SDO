# Experimental results (honest snapshot)

> Update this file when new ablation JSON lands under `results/ablation/`.  
> Numbers below used the **Colab `--proplus`** profile (data subset + capped
> batches) unless marked otherwise — **not** the full YAML 100-epoch protocol.

## Protocol reminder

| Setting | Full YAML (paper) | `--proplus` (Colab) |
|---------|-------------------|---------------------|
| Epochs | 100 | 50 |
| Train subset | 100% | 80% |
| Max train batches / epoch | unlimited | 500 |
| Image size | config / default | 640×1120 |
| Eval conf (M1–M4) | 0.05 | 0.05 |

## M0 — YOLO26n baseline

- Source: Ultralytics training on BSTLD
- Approximate AP50 ≈ **0.54** at conf 0.5 (not directly comparable to M1@0.05)

## M1 — Shallow generator + Soft-NMS

| Metric | Value |
|--------|------:|
| AP50 | 0.2412 |
| mAP50-95 | 0.1447 |
| Recall @ 0.05 | 0.5355 |
| Precision @ 0.05 | 0.0885 |
| TP / FP / FN | 1751 / 18027 / 1519 |

Interpretation: Stage-1 is recall-oriented; low precision at conf=0.05 is expected.
M3/M4 verification is designed to raise precision.

## M2 / M3 / M4

Pending.
