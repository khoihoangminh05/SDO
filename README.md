# SDO — Smart Detection & Optimization

Monorepo with two complementary parts:

| Part | Path | Purpose |
|------|------|---------|
| **TTLD-Net** | [`ttld-net/`](ttld-net/) | Research code: topology-aware tiny traffic-light detection |
| **SDO apps** | [`apps/`](apps/) | Demo stack: viewer, API, YOLO/HSV/OCR worker |

Research docs: [`document/`](document/) (`TTLD_Net_PROJECT_SPEC.md` → `TTLD_Net_PLAN.md` → `PROJECT_STATE.md`).

---

## Quick links

- **TTLD-Net README (start here for the thesis/code review):** [`ttld-net/README.md`](ttld-net/README.md)
- **Ablation configs:** `ttld-net/configs/m0_baseline.yaml` … `m4_full_ttld.yaml`
- **Phase status:** [`document/PROJECT_STATE.md`](document/PROJECT_STATE.md)

```bash
cd ttld-net
pip install -r requirements.txt
python -m pytest tests/ -q
python train.py --config configs/m1_shallow.yaml --output logs/ablation/m1_shallow --device 0
```

Dataset (BSTLD) lives under `apps/worker/datasets/` and is **not** shipped in git.

---

## Repository layout

```
.
├── document/           Research specs & project state
├── ttld-net/           Train / eval / ablation (PyTorch)
├── apps/               Frontend, orchestrator, worker
├── packages/types/     Shared TypeScript types
├── presentation/       Slides
└── report/             Project report assets
```

---

## License

TTLD-Net research code: MIT (`ttld-net/LICENSE`).  
Application packages may carry their own licenses where noted.

---

## SDO demo system

### Prerequisites

- Node.js 18+, pnpm, Conda, Docker

### Setup

```bash
docker compose up -d          # Redis
pnpm install
conda env create -f apps/worker/environment.yml
conda activate worker-env
```

### Run locally

```bash
pnpm --filter frontend dev          # :3001 (configure port)
pnpm --filter orchestrator start:dev  # :3000
cd apps/worker && uvicorn app.main:app --reload --port 8000
```

---

## Code quality

```bash
pnpm lint
pnpm format:check
black --check apps/worker/ ttld-net/
flake8 apps/worker/
```

---

## Archive

Root-level research scripts (`train_custom_yolo.py`, `yolov8-*.yaml`) moved to `archive/legacy-research/`.
