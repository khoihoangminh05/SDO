# TTLD-Net — PROJECT STATE

> File này lưu **trạng thái thực thi hiện tại** của dự án.
> AI Agent đọc file này đầu tiên để biết đang ở đâu, đã làm gì, và cần làm gì tiếp theo.
> **Không làm lại code đã hoàn thành. Không nhảy sang phase khác.**

---

## Current Phase: 5 — Verification & Contrastive Learning

---

## Phase Progress

```
[x] Phase 0 — Chuẩn bị Môi trường          ✅ COMPLETED
[x] Phase 1 — Data Pipeline & Baseline      ✅ COMPLETED
[x] Phase 2 — High-Recall Candidate Generator  ✅ COMPLETED
[x] Phase 3 — Semantic Context Branch       ✅ COMPLETED
[x] Phase 4 — Implicit Topology Sampler     ✅ COMPLETED
[ ] Phase 5 — Verification & Contrastive    🔄 IN PROGRESS  ← ĐANG Ở ĐÂY
[ ] Phase 6 — End-to-End Training           ⏳ BLOCKED (chờ Phase 5)
[ ] Phase 7 — Ablation Study & Visualization  ⏳ BLOCKED (chờ Phase 6)
[ ] Phase 8 — Cross-Dataset & Robustness Evaluation  ⏳ BLOCKED (chờ Phase 7)
```

---

## Next Task

```
T5.1 — Verification MLP integration tests + hard-negative mining wiring
```

**File**: `models/heads/verification.py`, `losses/infonce_loss.py`  
**Gate**: Test T5.A shape + T5.B InfoNCE convergence.

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
python -m pytest tests/test_phase4.py tests/test_phase3.py tests/test_phase2.py -v
python scripts/probe_backbone_channels.py --height 640 --width 640
```

---

*Cập nhật 2026-07-20: Phase 4 hoàn thành — ImplicitTopologySampler wired vào TTLDNet (mode topology/full).*
