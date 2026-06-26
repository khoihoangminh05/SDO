# TTLD-Net

Topology-Aware Tiny Traffic Light Detection Network — research codebase.

## Documentation

Read before coding (in order):

1. `../document/Context.md` — rules and constraints
2. `../document/TTLD_Net_PROJECT_SPEC.md` — architecture and hyperparameters
3. `../document/TTLD_Net_PLAN.md` — phase tasks and gates
4. `../document/PROJECT_STATE.md` — current progress

## Quick start

```bash
cd ttld-net
pip install -r requirements.txt

# Phase 0 — dev machine (Windows, no GPU)
python scripts/verify_dataset.py
python test_env.py --dev

# Phase 0 — GPU server (Linux)
conda env create -f environment.yml && conda activate ttld-net
bash scripts/setup_phase0.sh
python test_env.py --strict-gpu

# Unit tests
python -m pytest tests/
```

## Dataset

Bosch BSTLD images live under `../apps/worker/datasets/` (converted to YOLO format).
See `data/bosch/README.md` for paths used in config YAML files.

## Directory layout

```
ttld-net/
├── data/           Dataset, transforms, collate
├── models/         Backbone, neck, heads, TTLDNet
├── losses/         FocalLoss, InfoNCE
├── configs/        Ablation configs M0–M4
├── scripts/        EDA, ablation runner, visualization
├── utils/          Config loader, metrics, visualize
├── tests/          Phase gate unit tests
├── logs/           TensorBoard (gitignored)
├── checkpoints/    Weights (gitignored)
└── results/        Metrics JSON (gitignored)
```
