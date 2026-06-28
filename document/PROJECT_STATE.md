# TTLD-Net — PROJECT STATE

> AI Agent: đọc file này trước. Không làm lại task đã [x].

---

## Current Phase: 1 — Data Pipeline & Baseline (in progress)

---

## Phase Progress

```
[x] Phase 0 — Chuẩn bị Môi trường          ✅ GPU + dev PASS
[~] Phase 1 — Data Pipeline & Baseline      🔄 IN PROGRESS
[ ] Phase 2 — High-Recall Candidate Generator  ⏳ BLOCKED
[ ] Phase 3 — Semantic Context Branch       ⏳ BLOCKED
[ ] Phase 4 — Implicit Topology Sampler     ⏳ BLOCKED
[ ] Phase 5 — Verification & Contrastive    ⏳ BLOCKED
[ ] Phase 6 — End-to-End Training           ⏳ BLOCKED
[ ] Phase 7 — Ablation Study & Visualization  ⏳ BLOCKED
```

---

## Phase 1 Status

| Task | File | Status |
|------|------|--------|
| T1.1 EDA | `scripts/eda_bosch.py` | ✅ |
| T1.2 CLASS_MAPPING | `data/dataset.py` | ✅ |
| T1.3 BoschDataset | `data/dataset.py` | ✅ |
| T1.4 Augmentation + Resize 720×1280 | `data/transforms.py` | ✅ |
| T1.5 DataLoader + collate | `data/dataset.py` | ✅ |
| T1.6 Train M0 baseline | `train.py` + `utils/yolo_baseline.py` | ⏳ chạy trên GPU |
| T1.C Eval + metrics JSON | `test.py` | ⏳ sau train |

### EDA results (`logs/eda/eda_summary.json`)

- 5093 entries, 10755 boxes, avg 2.1 boxes/image
- **59.12%** boxes with width < 10px (tiny objects confirmed)
- Class: Green 5417, Red 4163, Yellow 444, Off 731

### Tests PASS (dev)

```bash
python -m pytest tests/test_phase1.py -v   # 3 passed
```

---

## Next Task — T1.6 trên GPU server

```bash
cd ~/SDO/ttld-net
conda activate ttld-net
git pull   # lấy Phase 1 code mới

# Train M0 baseline (~2-4h)
python train.py --config configs/m0_baseline.yaml \
  --output logs/ablation/m0_baseline --device 0

# Evaluate → metrics JSON
python test.py --config configs/m0_baseline.yaml \
  --weights checkpoints/m0_baseline_best.pt \
  --output results/ablation/m0_baseline_metrics.json --device 0
```

**Gate Phase 1:** `results/ablation/m0_baseline_metrics.json` tồn tại với `ap50`, `apsmall`, `recall`, `precision`, `fpr`.

---

*Cập nhật sau mỗi task.*
