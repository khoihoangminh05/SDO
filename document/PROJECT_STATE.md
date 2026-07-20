# TTLD-Net — PROJECT STATE

> File này lưu **trạng thái thực thi hiện tại** của dự án.
> AI Agent đọc file này đầu tiên để biết đang ở đâu, đã làm gì, và cần làm gì tiếp theo.
> **Không làm lại code đã hoàn thành. Không nhảy sang phase khác.**

---

## Current Phase: 2 — High-Recall Candidate Generator

---

## Phase Progress

```
[x] Phase 0 — Chuẩn bị Môi trường          ✅ COMPLETED
[x] Phase 1 — Data Pipeline & Baseline      ✅ COMPLETED
[ ] Phase 2 — High-Recall Candidate Generator  🔄 IN PROGRESS  ← ĐANG Ở ĐÂY
[ ] Phase 3 — Semantic Context Branch       ⏳ BLOCKED (chờ Phase 2 gate)
[ ] Phase 4 — Implicit Topology Sampler     ⏳ BLOCKED (chờ Phase 3)
[ ] Phase 5 — Verification & Contrastive    ⏳ BLOCKED (chờ Phase 4)
[ ] Phase 6 — End-to-End Training           ⏳ BLOCKED (chờ Phase 5)
[ ] Phase 7 — Ablation Study & Visualization  ⏳ BLOCKED (chờ Phase 6)
[ ] Phase 8 — Cross-Dataset & Robustness Evaluation  ⏳ BLOCKED (chờ Phase 7)
```

---

## Next Task

```
T2.B — High-recall eval trên GPU server (M1 metrics)
```

**Mô tả**: Sync code Phase 2 lên server, chạy pytest Phase 2, rồi eval M0 weights với `conf=0.05` để ghi `results/ablation/m1_shallow_metrics.json`. So sánh recall với M0 (0.547).

**Lệnh server** (xem cuối file / chat):

```bash
cd ~/SDO/ttld-net && conda activate ttld-net
git pull   # hoặc rsync code mới
python -m pytest tests/test_phase2.py tests/test_modules.py -v
python scripts/probe_backbone_channels.py --height 640 --width 640
python test.py --config configs/m1_shallow.yaml \
  --weights checkpoints/m0_baseline_best.pt \
  --output results/ablation/m1_shallow_metrics.json --device 0
```

---

## Completed Tasks Detail

### ✅ Phase 0 — Chuẩn bị Môi trường

| Task | Status |
|---|---|
| T0.1–T0.5 Env + Bosch dataset | ✅ (server: `logs/env_check.txt`) |

### ✅ Phase 1 — Data Pipeline & Baseline

| Task | Status |
|---|---|
| T1.1–T1.5 DataLoader / EDA / transforms | ✅ |
| T1.6 YOLO Baseline Training (M0) | ✅ `checkpoints/m0_baseline_best.pt` |
| T1.C Baseline metrics JSON | ✅ (eval 2026-07-19) |

**Baseline metrics (M0 — YOLO thuần, conf=0.5)**:

```json
{
  "ap50": 0.5422,
  "apsmall": 0.4609,
  "recall": 0.547,
  "precision": 0.7307,
  "fpr": 0.2693,
  "map50_95": 0.3328
}
```

---

## In Progress — Phase 2: High-Recall Candidate Generator

**Mục tiêu**: Recall cao (Stage 1) + `fcand` cho Phase 4.

| Task | File | Status |
|---|---|---|
| T2.1 Override YOLO Backbone → P1–P5 | `models/backbones/yolo26.py` | ✅ DONE |
| T2.2 FPN + PANet Neck | `models/necks/fpn_panet.py` | ✅ DONE |
| T2.3 TinyGenerator Head | `models/heads/tiny_generator.py` | ✅ DONE |
| T2.4 Soft-NMS | `models/heads/soft_nms.py` + generator | ✅ DONE |
| T2.5 Focal Loss (γ=1.5, α=0.75) | `losses/focal_loss.py` | ✅ (đã có) |
| T2.6 Config m1_shallow.yaml | `configs/m1_shallow.yaml` | ✅ |
| T2.A Unit tests Phase 2 | `tests/test_phase2.py` | ✅ (chạy local/server) |
| **T2.B** High-recall metrics M1 | `results/ablation/m1_shallow_metrics.json` | 🔄 **NEXT (server)** |

**Ghi chú**: Train mode `shallow` end-to-end cho TinyGenerator vẫn thuộc Phase 6. Interim M1 = **cùng weights M0**, eval với `conf=0.05` (Stage-1 threshold) để đo recall lift.

**Key outputs**:
- `fcand: (B, N, 256)`
- `candidates: List[Dict]` bbox xywh + confidence + class_id
- Backbone channels (scale n @ 640): probe bằng `scripts/probe_backbone_channels.py`

---

## What's Blocked

| Phase | Blocked by |
|---|---|
| Phase 3 | Phase 2 gate (tests PASS + M1 metrics file) |
| Phase 4+ | như PLAN |

---

## Files Đã Tồn Tại (không tạo lại)

```
✅ models/backbones/yolo26.py      (P1–P5 forward)
✅ models/necks/fpn_panet.py       (FPN+PAN + FeatureProjection)
✅ models/heads/tiny_generator.py
✅ models/heads/soft_nms.py
✅ losses/focal_loss.py
✅ configs/m0_baseline.yaml / m1_shallow.yaml / m2_focal.yaml
✅ tests/test_phase2.py
✅ scripts/probe_backbone_channels.py
✅ checkpoints/m0_baseline_best.pt
✅ results/ablation/m0_baseline_metrics.json
```

## Files Cần Tạo tiếp

```
🔄 results/ablation/m1_shallow_metrics.json  ← T2.B server
🔄 logs/backbone_channels.txt                ← probe script
⏳ models/heads/implicit_topo.py (forward)   ← Phase 4
```

---

*Cập nhật 2026-07-19: Phase 1 metrics filled; Phase 2 modules implemented; awaiting server T2.B.*
