# TTLD-Net — Project State

> Living status document for the research codebase.  
> Phase gates and specs: `TTLD_Net_PLAN.md`, `TTLD_Net_PROJECT_SPEC.md`.

---

## Current phase: 7 — Ablation & visualization (in progress)

```
[x] Phase 0 — Environment
[x] Phase 1 — Data pipeline & baseline
[x] Phase 2 — High-recall candidate generator
[x] Phase 3 — Semantic context branch
[x] Phase 4 — Implicit topology sampler
[x] Phase 5 — Verification & contrastive learning
[x] Phase 6 — End-to-end training
[ ] Phase 7 — Ablation study & visualization     ← in progress
[ ] Phase 8 — Cross-dataset / robustness         (blocked on Phase 7)
```

### Next

1. Finish M1 checkpoint persistence (Drive sync after train).
2. Run M2 → M3 → M4 (`scripts/run_ablation.py` or `scripts/continue_m2.sh`).
3. Produce `results/ablation_comparison_table.md` via `scripts/compare_ablation.py`.

### Validated (Colab `--proplus` protocol — subset / capped batches)

| Variant | Notes | AP50 | Recall @ 0.05 |
|---------|-------|------|----------------|
| M0 | Ultralytics YOLO26n | ~0.54 @ conf 0.5 | — |
| M1 | Shallow + Soft-NMS | ~0.24 | ~0.54 |

M2–M4: not completed. Incomplete rows must not be presented as final paper results.

Engineering fixes landed during Phase 7: lazy stack materialize-before-optim,
pretrained YOLO26n transfer into P2 YAML, softmax class decode, VOC AP50 +
true mAP@0.5:0.95 + size-bin AP, YOLO val denorm (no double-scale).

---

## Architecture (implemented)

```
Input → YOLO26-P2 (P1–P5) → FPN/PANet
      → TinyGenerator (P1–P3, Soft-NMS)
      → SemanticContext (P4/P5)
      → ImplicitTopologySampler (deformable attn)
      → VerificationMLP + InfoNCE
L = L_det + λ1·L_topology + λ2·L_verify
```

---

## Key entrypoints

| Task | Command |
|------|---------|
| Train | `python train.py --config configs/m4_full_ttld.yaml --device 0` |
| Eval | `python test.py --config … --weights checkpoints/….pth --output results/…` |
| Ablation | `python scripts/run_ablation.py --configs m1_shallow … --device 0` |
| Tests | `python -m pytest tests/ -q` |
| Colab | `ttld-net/document/COLAB.md` |

---
