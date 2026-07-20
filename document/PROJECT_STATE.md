# TTLD-Net — PROJECT STATE

> File này lưu **trạng thái thực thi hiện tại** của dự án.
> AI Agent đọc file này đầu tiên để biết đang ở đâu, đã làm gì, và cần làm gì tiếp theo.
> **Không làm lại code đã hoàn thành. Không nhảy sang phase khác.**

---

## Current Phase: 7 — Ablation Study & Visualization

---

## Phase Progress

```
[x] Phase 0 — Chuẩn bị Môi trường          ✅ COMPLETED
[x] Phase 1 — Data Pipeline & Baseline      ✅ COMPLETED
[x] Phase 2 — High-Recall Candidate Generator  ✅ COMPLETED
[x] Phase 3 — Semantic Context Branch       ✅ COMPLETED
[x] Phase 4 — Implicit Topology Sampler     ✅ COMPLETED
[x] Phase 5 — Verification & Contrastive    ✅ COMPLETED
[x] Phase 6 — End-to-End Training           ✅ COMPLETED
[ ] Phase 7 — Ablation Study & Visualization  🔄 IN PROGRESS  ← ĐANG Ở ĐÂY
[ ] Phase 8 — Cross-Dataset & Robustness Evaluation  ⏳ BLOCKED (chờ Phase 7)
```

---

## Next Task

```
T7.1 — scripts/run_ablation.py + compare_ablation.py
```

**Gate**: 5 ablation configs M0–M4 chạy được, metrics JSON xuất ra `results/ablation/`.

---

## Completed — Phase 6: End-to-End Training

| Task | File | Status |
|---|---|---|
| T6.1 Training loop | `utils/trainer.py`, `train.py` | ✅ |
| T6.2 Loss combination | `models/ttld_net.py`, `losses/detection_loss.py` | ✅ |
| T6.3 TensorBoard logging | `utils/trainer.py` | ✅ |
| T6.4 Evaluation | `test.py`, `utils/metrics.py` | ✅ |

**Loss**: `L_total = L_det + λ1·L_topology + λ2·L_verify`

**Gate PASSED**: `tests/test_phase6.py` — T6.A + T6.B smoke.

**Train trên GPU server** (xem hướng dẫn bên dưới).

---

## Completed — Phase 5: Verification & Contrastive Learning

| Task | File | Status |
|---|---|---|
| T5.1 VerificationMLP | `models/heads/verification.py` | ✅ |
| T5.2 InfoNCE Loss | `losses/infonce_loss.py` | ✅ |
| T5.3 Hard-negative mining + GT matching | `models/heads/verification.py` | ✅ |
| T5.4 Positive pair builder | `models/heads/verification.py` | ✅ |
| Loss wiring | `models/ttld_net.py` (`compute_losses`) | ✅ |

**Gate PASSED**: `tests/test_phase5.py` — T5.A + T5.B + integration.

---

## Completed — Phase 4: Implicit Topology Sampler

| Task | File | Status |
|---|---|---|
| T4.1 Reference points + batch padding | `models/heads/implicit_topo.py` | ✅ |
| T4.2/T4.3 MultiScaleDeformableAttention | `models/heads/implicit_topo.py` | ✅ |
| T4.4 Unit tests | `tests/test_phase4.py` | ✅ |

**Outputs**:
- `zi: (B, N, 256)` topology embeddings via deformable attention over P4/P5
- MMCV CUDA on GPU server; pure-PyTorch fallback on dev (no mmcv)

**Gate PASSED**: `tests/test_phase4.py` — T4.A + T4.B + integration.

---

## Completed — Phase 3: Semantic Context Branch

| Task | File | Status |
|---|---|---|
| T3.1 FeatureProjection (Conv 1×1) | `models/necks/fpn_panet.py` | ✅ |
| T3.2 Xác nhận channel dims backbone | `scripts/probe_backbone_channels.py` | ✅ |
| T3.3 Context Branch Integration | `models/necks/context_branch.py`, `models/ttld_net.py` | ✅ |

**Outputs**:
- `fctx_p4: (B, 256, H/16, W/16)`
- `fctx_p5: (B, 256, H/32, W/32)`
- `fcand_proj: (B, N, 256)`

**Gate PASSED**: `tests/test_phase3.py` — T3.A + T3.B + T3.3 integration.

---

## Baseline M0 metrics (conf=0.5)

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

## Server verify (sau mỗi sync code)

```bash
cd ~/SDO/ttld-net && conda activate ttld-net
python -m pytest tests/test_phase6.py tests/test_phase5.py tests/test_phase4.py -v
```

### Train full model trên GPU server (bạn thao tác)

```bash
cd ~/SDO/ttld-net && conda activate ttld-net

# Smoke test (vài step, kiểm tra pipeline)
python train.py --config configs/m4_full_ttld.yaml \
  --output logs/ablation/m4_full_ttld \
  --device 0 --max-steps 10

# Train đầy đủ (~100 epoch, cần RTX 4090 / tương đương)
python train.py --config configs/m4_full_ttld.yaml \
  --output logs/ablation/m4_full_ttld \
  --device 0

# Evaluate
python test.py --config configs/m4_full_ttld.yaml \
  --weights checkpoints/m4_full_ttld_best.pth \
  --output results/ablation/m4_full_ttld_metrics.json
```

TensorBoard: `tensorboard --logdir logs/ablation/m4_full_ttld/tensorboard`

---

*Cập nhật 2026-07-20: Phase 6 hoàn thành — train.py + trainer + L_det wired.*
