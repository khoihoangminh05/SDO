# TTLD-Net

**Topology-Aware Tiny Traffic Light Detection Network**

A two-stage detector for extremely small traffic lights (Bosch BSTLD), combining
a high-recall shallow-feature generator with deformable-attention topology
sampling and contrastive verification.

```
L = L_det + λ₁ · L_topology + λ₂ · L_verify
```

| Stage | Module | Role |
|-------|--------|------|
| 1 | YOLO26-P2 backbone + FPN/PANet + TinyGenerator | High-recall candidates (P1–P3), Soft-NMS |
| 2a | ImplicitTopologySampler | Multi-scale deformable attention over P4/P5 |
| 2b | VerificationMLP + InfoNCE | Suppress false positives; contrastive topology |

---

## Ablation variants (M0–M4)

| ID | Config | Components |
|----|--------|------------|
| M0 | `configs/m0_baseline.yaml` | Ultralytics YOLO26n baseline |
| M1 | `configs/m1_shallow.yaml` | Shallow generator + Soft-NMS |
| M2 | `configs/m2_focal.yaml` | M1 + focal loss tuning |
| M3 | `configs/m3_topology.yaml` | M2 + topology sampler + InfoNCE (λ₁) |
| M4 | `configs/m4_full_ttld.yaml` | Full model (λ₁ + λ₂ verification) |

---

## Environment

```bash
cd ttld-net
conda env create -f environment.yml && conda activate ttld-net
# or: pip install -r requirements.txt

python test_env.py
python -m pytest tests/ -q
```

Optional: MMCV CUDA ops accelerate deformable attention; a pure-PyTorch fallback is built in.

---

## Data (BSTLD)

Bosch Small Traffic Lights Dataset under `../apps/worker/datasets/` (YOLO labels).
See `data/bosch/README.md`. Images are **not** committed to git.

---

## Train & evaluate

```bash
# Single variant (full protocol from YAML)
python train.py --config configs/m1_shallow.yaml --output logs/ablation/m1_shallow --device 0
python test.py  --config configs/m1_shallow.yaml \
  --weights checkpoints/m1_shallow_best.pth \
  --output results/ablation/m1_shallow_metrics.json --device 0

# Ablation runner
python scripts/run_ablation.py --configs m1_shallow m2_focal m3_topology m4_full_ttld --device 0

# Limited-budget profiles (disclose in papers — not the full YAML protocol)
python scripts/run_ablation.py --configs m1_shallow --device 0 --proplus   # Colab Pro+
python scripts/run_ablation.py --configs m1_shallow --device 0 --fast      # smoke / short
```

**Evaluation notes**

- TTLD variants report **VOC-style AP50**, **mAP@0.5:0.95**, and size-bin AP (`ap_tiny` / `ap_small` / `apsmall`).
- Operating-point precision/recall use `conf=0.05` (high-recall Stage-1).
- M0 baseline uses Ultralytics metrics at its own confidence (typically 0.5) — compare carefully.

---

## Repository layout

```
ttld-net/
├── models/          Backbone, FPN/PANet, TinyGenerator, topology, verification
├── losses/          Detection (focal CE), InfoNCE
├── data/            Bosch / YOLO loaders, transforms
├── configs/         M0–M4 YAML
├── utils/           Trainer, metrics, Colab helpers
├── scripts/         Ablation, EDA, packing
├── tests/           Phase gate tests
└── document/        Colab / ops notes
```

Upstream specs: `../document/TTLD_Net_PROJECT_SPEC.md`, `../document/TTLD_Net_PLAN.md`.  
Honest result snapshot: [`document/RESULTS.md`](document/RESULTS.md).

---

## Results status

Phase 7 (ablation) is **in progress**. Validated under the Colab `--proplus` protocol
(subset training, capped batches — see `utils/fast_train.py`):

| Variant | Protocol note | AP50 | Recall | Precision |
|---------|---------------|------|--------|-----------|
| M0 | Ultralytics baseline | ~0.54 @ conf 0.5 | — | — |
| M1 | `--proplus`, conf 0.05 | ~0.24 | ~0.54 | ~0.09 |

M2–M4: pending. Do not treat incomplete rows as final paper numbers.

---

## Citation

If you use this codebase, please cite the accompanying project report / thesis
and the Bosch BSTLD dataset.

## License

MIT — see `LICENSE`.
