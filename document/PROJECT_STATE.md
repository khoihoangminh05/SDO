# TTLD-Net — PROJECT STATE

> File này lưu **trạng thái thực thi hiện tại** của dự án.
> AI Agent đọc file này đầu tiên để biết đang ở đâu, đã làm gì, và cần làm gì tiếp theo.

---

## Current Phase: 0 — Hoàn thành (dev) | Chờ GPU server

---

## Phase Progress

```
[x] Phase 0 — Chuẩn bị Môi trường          ✅ DEV PASS (GPU server: chạy setup_phase0.sh)
[ ] Phase 1 — Data Pipeline & Baseline      ⏳ NEXT
[ ] Phase 2 — High-Recall Candidate Generator  ⏳ BLOCKED
[ ] Phase 3 — Semantic Context Branch       ⏳ BLOCKED
[ ] Phase 4 — Implicit Topology Sampler     ⏳ BLOCKED
[ ] Phase 5 — Verification & Contrastive    ⏳ BLOCKED
[ ] Phase 6 — End-to-End Training           ⏳ BLOCKED
[ ] Phase 7 — Ablation Study & Visualization  ⏳ BLOCKED
```

---

## Phase 0 Results (2026-06-25)

### Máy dev (Windows) — `python test_env.py --dev` ✅

| Check | Status |
|-------|--------|
| PyTorch 2.12.0 | ✅ |
| yaml, cv2, tensorboard, ultralytics, pytest | ✅ |
| BSTLD train.yaml (5093 entries, 10756 boxes) | ✅ |
| Val split (1000 YOLO images) | ✅ |
| CUDA | ❌ CPU only (expected) |
| MMCV DeformableAttention | ❌ Cần GPU server |

Log: `ttld-net/logs/env_check.txt`

### Dataset (T0.5) ✅

```
train: 5093 entries, 3153 images with boxes, 0 missing
val:   1000 png (YOLO format via bstld.yaml)
Top labels: Green=5207, Red=3057, RedLeft=1092, off=726, Yellow=444
```

Log: `ttld-net/logs/dataset_stats.txt`

### GPU server — việc cần làm

```bash
cd ttld-net
conda env create -f environment.yml   # hoặc: bash scripts/setup_phase0.sh
conda activate ttld-net
bash scripts/setup_phase0.sh          # cài MMCV + verify CUDA
python test_env.py --strict-gpu       # gate đầy đủ
```

---

## Next Task (Phase 1)

```
T1.1 — Implement scripts/eda_bosch.py (histogram bbox, class distribution)
```

Sau đó: T1.3 DataLoader smoke test, T1.6 train M0 baseline trên GPU server.

---

## Completed Phase 0 Tasks

| Task | File | Status |
|------|------|--------|
| T0.1 | CUDA check (dev: CPU) | ✅ dev / ⏳ GPU |
| T0.2 | pip install requirements.txt | ✅ |
| T0.3 | MMCV CUDA | ⏳ GPU server |
| T0.4 | Scaffold ttld-net/ | ✅ |
| T0.5 | verify_dataset.py | ✅ PASS |
| Gate | test_env.py --dev | ✅ |
| Setup scripts | setup_phase0.ps1, setup_phase0.sh | ✅ |
| Conda env | environment.yml | ✅ |

---

*Cập nhật file này sau mỗi phase hoàn thành.*
