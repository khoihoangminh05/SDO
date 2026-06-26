# SDO — Smart Detection & Optimization

Monorepo gồm **hai phần**:

| Phần | Thư mục | Mục đích |
|------|---------|----------|
| **TTLD-Net** (research) | `ttld-net/` | Train & evaluate topology-aware tiny traffic-light detector |
| **SDO** (demo system) | `apps/` | Web UI + API + inference pipeline (YOLO, HSV, OCR) |

Tài liệu kế hoạch: `document/` (đọc `Context.md` → `TTLD_Net_PROJECT_SPEC.md` → `TTLD_Net_PLAN.md` → `PROJECT_STATE.md`).

---

## Repository layout

```
.
├── document/              # Research specs, plan, project state
├── ttld-net/              # TTLD-Net research codebase (train/test)
├── apps/
│   ├── frontend/          # Next.js viewer
│   ├── orchestrator/      # NestJS API + Redis cache
│   └── worker/            # FastAPI YOLO + HSV + OCR worker
├── packages/types/        # Shared TypeScript types
├── presentation/          # Research slides (HTML)
├── report/                # Project report (HTML)
├── archive/               # Legacy scripts moved from root
├── docker-compose.yml     # Redis (extend for full stack later)
└── pnpm-workspace.yaml
```

---

## TTLD-Net (research)

```bash
cd ttld-net
pip install -r requirements.txt
python test_env.py          # Phase 0 gate
python -m pytest tests/     # Unit tests for scaffold modules
```

Training / evaluation (after implementing phases):

```bash
python train.py --config configs/m0_baseline.yaml --output logs/ablation/m0_baseline
python test.py --config configs/m0_baseline.yaml --weights checkpoints/m0_baseline_best.pth --output results/ablation/m0_baseline_metrics.json
```

Dataset: Bosch BSTLD under `apps/worker/datasets/` (see `ttld-net/data/bosch/README.md`).

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
