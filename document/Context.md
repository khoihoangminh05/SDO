# TTLD-Net AI Context

> File này là **ngữ cảnh bổ trợ cho AI Agent** — đọc file này trước khi đọc bất kỳ file nào khác trong dự án. Nó tóm tắt toàn bộ mục tiêu, kiến trúc, quy tắc làm việc và những điều cần tránh.

---

## Project Goal

**TTLD-Net** (Topology-Aware Tiny Traffic Light Detection Network) là framework nghiên cứu phát hiện đèn giao thông siêu nhỏ cho xe tự hành, sử dụng kiến trúc 2-stage kết hợp Deformable Attention.

Mục tiêu chính:
- Nghiên cứu Computer Vision
- Viết paper (kết quả, ablation study, visualization)
- Không phải production system

Ưu tiên:
1. Correctness
2. Reproducibility
3. Research value
4. Performance

Không ưu tiên:
- Over-engineering
- Micro-optimization
- Enterprise architecture

---

## Important Directories

```
ttld-net/
├── data/           # Dataset class, DataLoader, augmentation
├── models/
│   ├── backbones/  # YOLO26 backbone
│   ├── necks/      # FPN + PANet
│   └── heads/      # Generator, Topology Sampler, Verification
├── losses/         # FocalLoss, InfoNCE
├── configs/        # YAML config cho từng ablation variant (m0–m4)
├── scripts/        # EDA, ablation runner, attention visualization
├── utils/          # Visualization utilities
├── logs/           # TensorBoard logs
├── checkpoints/    # Model weights (.pth)
└── results/
    └── ablation/   # Metrics JSON từ 5 ablation variants
```

Không tự tạo thư mục mới nếu không được yêu cầu.

---

## Coding Standards

1. Python >= 3.10
2. Type hints bắt buộc cho tất cả function signature
3. Mỗi class phải có docstring
4. Mỗi module phải có unit test tương ứng
5. Không viết quá 300 dòng/file — tách module nếu cần
6. Không hard-code path — dùng config YAML hoặc argparse

---

## Fixed Constraints — Không được thay đổi

**Dataset training**: Bosch Small Traffic Lights Dataset (`data/bosch/train.yaml`, `test.yaml`) — dataset **duy nhất dùng để train** toàn bộ M0–M4.

**Backbone**: YOLO26 (file: `models/backbones/yolo26.py`)

**Không được**:
- Chuyển sang Faster R-CNN
- Chuyển sang DETR hoặc bất kỳ transformer detector nào
- Thay backbone khác
- Train hoặc fine-tune trên bất kỳ dataset nào khác ngoài Bosch (kể cả các dataset dùng cho Phase 8)

> Trừ khi được yêu cầu rõ ràng bởi người dùng.

### Ngoại lệ — Dataset chỉ dùng để Đánh giá (Phase 8)

Để chứng minh generalization (một trong ba thách thức cốt lõi ở trên), Phase 8 dùng thêm 4 dataset công khai **chỉ ở chế độ zero-shot inference/test — không train, không fine-tune**:

| Dataset | Vai trò | Vùng miền |
|---|---|---|
| DTLD (DriveU) | Test domain shift camera/độ phân giải, có pictogram + trạng thái đèn vàng-đỏ | Đức |
| S2TLD (SJTU) | Test domain shift lục địa khác, đèn nhấp nháy, vật thể dễ nhầm | Trung Quốc |
| LISA | Test benchmark chuẩn cùng niche, so sánh trực tiếp với literature | Mỹ |
| Cityscapes TL++ (CSTL) | Test mật độ đèn/ảnh khác biệt, so sánh SOTA công bố | Đức |

Chi tiết protocol: xem `PROJECT_SPEC.md` Mục 14 và `PLAN.md` Phase 8.

---

## Architecture Overview

Kiến trúc 2-stage song song, hội tụ tại Implicit Topology Sampler:

```
Input (B, 3, 720, 1280)
        │
        ▼
YOLO26 Backbone + PANet → P1 (stride 2), P2 (stride 4), P3 (stride 8)
                        → P4 (stride 16), P5 (stride 32)
        │                          │
        ▼ (shallow: P1, P2, P3)    ▼ (deep: P4, P5)
[Phase 2] High-Recall          [Phase 3] Semantic Context
Candidate Generator            Branch → Fctx (B, 256, H, W)
→ C = {c1..cN}
→ fcand (B, N, 256)
        │                          │
        └─────────┬─────────────────┘
                  ▼
        [Phase 4] Implicit Topology Sampler
        (Deformable Attention — MMCV)
        → zi (B, N, 256)
                  │
                  ▼
        [Phase 5] Verification MLP + InfoNCE
        → P(Valid) per candidate
        → Final Detection Set
```

**Triết lý**: "Thà bắt nhầm còn hơn bỏ sót" ở Stage 1 — Recall > 95%. Stage 2 lọc nghiêm bằng topology ngầm.

---

## Key Tensor Shapes (tham chiếu nhanh)

| Tensor | Shape | Ghi chú |
|---|---|---|
| Input | `(B, 3, 720, 1280)` | |
| P1–P3 | stride 2/4/8 | Shallow — dùng cho candidate generation |
| P4–P5 | stride 16/32 | Deep — dùng cho context branch |
| fcand | `(B, N, 256)` | Local feature mỗi ứng viên |
| Fctx | `(B, 256, H4, W4)` + `(B, 256, H5, W5)` | Đã project về 256-dim |
| zi | `(B, N, 256)` | Topology Embedding từ Deformable Attention |
| vi | `(B, N, 512)` | concat(fcand, zi) → input MLP |
| P(Valid) | `(B, N, 1)` | Output verification head |

---

## Key Hyperparameters (không đoán — đây là spec)

| Parameter | Value | Lý do |
|---|---|---|
| `conf_threshold` (Stage 1) | 0.05 | Giữ hard examples, Recall > 95% |
| `soft_nms_sigma` | 0.5 | Không xóa nhầm đèn đứng sát nhau |
| `focal_gamma` | 1.5 | Phạt FN nặng hơn default (2.0) |
| `focal_alpha` | 0.75 | Ưu tiên minority class |
| `fcand_dim` | 256 | Unified embedding space |
| `n_points K` | 4–8 | Số sampling points Deformable Attention |
| `n_heads` | 8 | MultiScaleDeformableAttention |
| `temperature τ` | 0.07 | InfoNCE sharpness |
| `n_hard_negatives` | 32 | Hard negative mining |
| `learning_rate` | 1e-4 | AdamW |
| `warmup_epochs` | 5 | Cosine Annealing warmup |
| `total_epochs` | 100 | Đủ cho TTLD-Net hội tụ |
| `batch_size` | 32 (OOM: 8–16) | Tùy VRAM |
| `lambda1, lambda2` | 1.0 | Loss weights cho topology và verify |
| `grad_clip_norm` | 1.0 | Tránh gradient explosion |

---

## Loss Function

```
L_total = L_det + λ1 * L_topology + λ2 * L_verify
```

- `L_det`: YOLO detection loss (bbox regression + classification)
- `L_topology`: InfoNCE Contrastive Loss (τ = 0.07) — file: `losses/infonce_loss.py`
- `L_verify`: Binary Cross-Entropy — file: `models/heads/verification.py`

---

## Execution Workflow

**Trước khi code**:
1. Đọc `PROJECT_SPEC.md` — giải thích *tại sao* và *kỹ thuật cụ thể*
2. Đọc `PLAN.md` — giải thích *làm gì, thứ tự nào, kiểm tra thế nào*

**Khi code**:
- Chỉ làm đúng phase hiện tại
- Không implement trước phase tương lai

**Sau khi code**:
- Tạo unit test
- Chạy test
- Báo kết quả

---

## Phase Dependency (thứ tự nghiêm ngặt)

```
Phase 0 (Env Setup)
    └── Phase 1 (Data Pipeline + Baseline YOLO)
            ├── Phase 2 (High-Recall Generator)    ─┐
            └── Phase 3 (Semantic Context Branch)  ─┤ song song OK
                                                    ▼
                                        Phase 4 (Topology Sampler) ← cần cả 2
                                            └── Phase 5 (Verification + InfoNCE)
                                                    └── Phase 6 (End-to-End Training)
                                                            └── Phase 7 (Ablation + Viz)
                                                                    └── Phase 8 (Cross-Dataset & Robustness Eval)
```

> Phase 8 chỉ cần checkpoint M0 và M4 đã train xong ở Phase 6/7 — không cần train lại, chỉ chạy inference zero-shot trên dataset ngoài.

**Phase Gates** — phải PASS trước khi sang phase tiếp:
- Phase 0: `test_env.py` pass, MMCV CUDA OK
- Phase 1: DataLoader test PASS + baseline metrics file tồn tại
- Phase 2: Recall > 95% trên validation set
- Phase 3: FeatureProjection shape test PASS
- Phase 4: ImplicitTopologySampler shape + gradient test PASS
- Phase 5: VerificationMLP + InfoNCE convergence test PASS
- Phase 6: APsmall +3%, FPR -15%, Recall > 95%
- Phase 7: 5 ablation runs hoàn thành, xu hướng metrics đúng hướng
- Phase 8: zero-shot inference PASS trên ≥3/4 dataset ngoài (DTLD, S2TLD, LISA, CSTL), báo cáo đầy đủ AP50/APsmall/FPR + size-stratified breakdown cho mỗi dataset

---

## Ablation Study — 5 Variants

| Config | File | Components |
|---|---|---|
| M0 | `m0_baseline.yaml` | YOLO thuần — baseline |
| M1 | `m1_shallow.yaml` | + P1/P2/P3, Soft-NMS, conf=0.05 |
| M2 | `m2_focal.yaml` | + Focal Loss γ=1.5, α=0.75 |
| M3 | `m3_topology.yaml` | + Implicit Topology Sampler |
| M4 | `m4_full_ttld.yaml` | Full model: + InfoNCE + Verification |

Xu hướng kỳ vọng: APsmall M0 < M1 < M2 < M3 < M4 / FPR M0 > M1 > M2 > M3 > M4.

---

## Definition of Done

Một task chỉ được coi là hoàn thành khi:

- [ ] Code chạy được
- [ ] Unit test pass
- [ ] Không có syntax error
- [ ] Có docstring
- [ ] Có type hints

---

## Common Errors & Fixes

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| Recall < 90% sau Phase 2 | Hard NMS còn sót | Override bằng Soft-NMS |
| Channel mismatch ở Phase 4 | Chưa project về 256-dim | Thêm Conv 1×1 cho P4, P5, fcand |
| MMCV import error | CUDA version mismatch | Dùng đúng URL download mmcv |
| L_topology không giảm | Thiếu F.normalize trong InfoNCE | `F.normalize(zi, dim=-1)` trước dot product |
| NaN loss | Gradient explosion | `clip_grad_norm_(model.parameters(), max_norm=1.0)` |
| OOM với batch_size=32 | VRAM không đủ | Giảm batch size hoặc K từ 8→4 |
| L_det tăng sau epoch 20 | Learning rate quá cao | Kiểm tra LR scheduler |

---

## What NOT to Do (tổng hợp từ SPEC)

- ❌ Không dùng NMS cứng (hard NMS) ở Stage 1 — dùng Soft-NMS
- ❌ Không lọc False Positives ở Stage 1 — cố ý để lại cho Stage 2 xử lý
- ❌ Không dùng confidence threshold > 0.1 ở Stage 1
- ❌ Không dùng random negatives thuần trong InfoNCE — dùng Hard Negative Mining (lấy FP có confidence cao nhất)
- ❌ Không hard-code path
- ❌ Không implement code phase tương lai khi chưa đến phase đó
- ❌ Không commit code chưa test

---

## Commit Format

```
feat:      thêm tính năng mới
fix:       sửa bug
docs:      cập nhật tài liệu
refactor:  cải thiện code không thay đổi logic
test:      thêm hoặc sửa test
```

Không commit code chưa test.

---

## Technology Stack

```
Python >= 3.10
torch >= 2.0.0
torchvision >= 0.15.0
ultralytics >= 8.0.0        # YOLO base
mmcv-full >= 2.0.0          # BẮT BUỘC cho Deformable Attention CUDA Kernel
pyyaml, pillow, opencv-python, matplotlib
tensorboard, tqdm, numpy
```

> MMCV phải cài đúng CUDA version. Kiểm tra bằng:
> `python -c "from mmcv.ops import MultiScaleDeformableAttention; print('OK')"`

---

*Đọc file này một lần trước khi bắt đầu. Mọi quyết định kiến trúc và hyperparameter đã được ghi rõ trong `PROJECT_SPEC.md`. Thứ tự implement theo `PLAN.md`. Không đoán — tra cứu spec.*
