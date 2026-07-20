# TTLD-Net — PROJECT PLAN
> **Tài liệu Lập kế hoạch Thực thi dành cho AI Coding Agent**
> Phiên bản: 1.0 | Bổ trợ cho: `TTLD_Net_PROJECT_SPEC.md`
> Tác giả: Huy, Khôi | Tháng 6, 2026

---

## ĐỌC FILE NÀY NHƯ THẾ NÀO

File này là **bản đồ thực thi** — nó trả lời câu hỏi **"Làm gì, làm theo thứ tự nào, kiểm tra thế nào"**.

`TTLD_Net_PROJECT_SPEC.md` trả lời **"Tại sao và làm như thế nào về mặt kỹ thuật"**.

> **Quy tắc làm việc**: Trước khi bắt đầu bất kỳ task nào, đọc section tương ứng trong `SPEC.md`. Sau khi hoàn thành task, chạy test tương ứng trong phần **Kiểm thử** của phase đó. **Không được bỏ qua bước kiểm thử.**

---

## MỤC LỤC

- [Tổng quan Dự án](#tổng-quan-dự-án)
- [Dependency Map — Thứ tự Phụ thuộc](#dependency-map)
- [Phase 0 — Chuẩn bị Môi trường](#phase-0--chuẩn-bị-môi-trường)
- [Phase 1 — Data Pipeline & Baseline](#phase-1--data-pipeline--baseline)
- [Phase 2 — High-Recall Candidate Generator](#phase-2--high-recall-candidate-generator)
- [Phase 3 — Semantic Context Branch](#phase-3--semantic-context-branch)
- [Phase 4 — Implicit Topology Sampler](#phase-4--implicit-topology-sampler)
- [Phase 5 — Verification & Contrastive Learning](#phase-5--verification--contrastive-learning)
- [Phase 6 — End-to-End Training](#phase-6--end-to-end-training)
- [Phase 7 — Ablation Study & Visualization](#phase-7--ablation-study--visualization)
- [Phase 8 — Cross-Dataset Generalization & Robustness Evaluation](#phase-8--cross-dataset-generalization--robustness-evaluation)
- [Tổng hợp Checklist Toàn Dự án](#tổng-hợp-checklist-toàn-dự-án)

---

## Tổng quan Dự án

### Mục tiêu Nghiên cứu

TTLD-Net là **framework phát hiện đèn giao thông siêu nhỏ** cho xe tự hành. Kiến trúc 2-stage:

- **Stage 1 (Generator)**: Recall > 95% — "thà bắt nhầm còn hơn bỏ sót", chấp nhận nhiều False Positives
- **Stage 2 (Verifier)**: Lọc nghiêm — dùng topology ngầm để phân biệt đèn thật vs nhiễu

### Kết quả Kỳ vọng (so với YOLO baseline)

| Metric | Mục tiêu |
|---|---|
| AP50 | +1.5% ~ +3.0% |
| APsmall | **+3.0% ~ +6.0%** |
| Recall | +2.0% ~ +5.0% |
| Precision | +5.0% ~ +8.0% |
| False Positive Rate | **−15% ~ −30%** |

### Kiến trúc Luồng Dữ liệu (Tóm tắt)

```
Input (1280×720)
    │
    ▼
YOLO26 Backbone → P1, P2, P3, P4, P5
    │                    │
    ▼ (shallow)          ▼ (deep)
[Phase 2]            [Phase 3]
High-Recall          Semantic Context
Generator            Branch (P4, P5)
→ candidates C       → Fctx (256-dim)
→ fcand (B,N,256)
    │                    │
    └──────────┬──────────┘
               ▼
           [Phase 4]
    Implicit Topology Sampler
    (Deformable Attention, MMCV)
    → zi (B, N, 256)
               │
               ▼
           [Phase 5]
    Verification MLP + InfoNCE
    → P(Valid) per candidate
    → Final Detection Set
```

---

## Dependency Map

Thứ tự phụ thuộc nghiêm ngặt — **không được đảo thứ tự**:

```
Phase 0 (Env Setup)
    └── Phase 1 (Data Pipeline + Baseline)
            └── Phase 2 (High-Recall Generator)
            │       └── Phase 4 (Topology Sampler) ← cần fcand từ Phase 2
            └── Phase 3 (Context Branch)
                    └── Phase 4 (Topology Sampler) ← cần Fctx từ Phase 3
                            └── Phase 5 (Verification + InfoNCE)
                                    └── Phase 6 (End-to-End Training)
                                            └── Phase 7 (Ablation + Viz)
                                                    └── Phase 8 (Cross-Dataset & Robustness Eval)
```

> Phase 2 và Phase 3 có thể phát triển **song song**, nhưng Phase 4 phải đợi cả hai hoàn thành.
> Phase 8 chỉ cần checkpoint `m0_baseline_best.pth` và `m4_full_ttld_best.pth` từ Phase 6/7 — không train lại, chỉ inference.

---

## Phase 0 — Chuẩn bị Môi trường

> **Tham chiếu SPEC**: Mục 11 (Stack Công nghệ & Dependencies)

### Mục tiêu

Đảm bảo toàn bộ dependencies — đặc biệt là MMCV với CUDA kernel — hoạt động đúng trước khi viết bất kỳ dòng model code nào.

### Danh sách Task

**T0.1 — Kiểm tra CUDA Environment**
```bash
# Chạy và ghi lại output để xác nhận versions
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```
- Output cần có: `cuda.is_available() = True`
- Ghi lại: CUDA version (vd: 11.8 hoặc 12.x) — dùng cho T0.3

**T0.2 — Cài PyTorch và Core Dependencies**
```bash
pip install torch>=2.0.0 torchvision>=0.15.0
pip install ultralytics>=8.0.0        # YOLOv8 base
pip install pyyaml pillow opencv-python matplotlib
pip install tensorboard tqdm numpy
```

**T0.3 — Cài MMCV với CUDA Support** ⚠️ CRITICAL
```bash
# Thay {cuda_ver} và {torch_ver} từ kết quả T0.1
# Ví dụ CUDA 11.8, PyTorch 2.0:
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html

# Ví dụ CUDA 12.1, PyTorch 2.1:
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
```

**T0.4 — Khởi tạo Cấu trúc Thư mục**
```bash
mkdir -p ttld-net/{data,models/{backbones,necks,heads},losses,configs,scripts,utils,logs,checkpoints,results/ablation}
touch ttld-net/train.py ttld-net/test.py ttld-net/requirements.txt ttld-net/README.md
```

**T0.5 — Tải Bosch Dataset**
- Download `train.yaml`, `test.yaml`, và thư mục ảnh PNG từ nguồn Bosch
- Đặt vào `data/bosch/` (hoặc path tùy chỉnh, cập nhật vào configs sau)
- Kiểm tra cấu trúc file YAML xem có đúng format `{image_path: ..., boxes: [...]}` không

### Kiểm thử Phase 0

```python
# test_env.py — chạy file này, tất cả phải PASS
import torch
from mmcv.ops import MultiScaleDeformableAttention
import yaml, cv2, tensorboard

print("✅ PyTorch:", torch.__version__)
print("✅ CUDA available:", torch.cuda.is_available())
print("✅ MMCV MultiScaleDeformableAttention: OK")

# Kiểm tra dataset file tồn tại
import os
assert os.path.exists("data/bosch/train.yaml"), "❌ train.yaml không tìm thấy"
print("✅ Bosch train.yaml: OK")
```

**Gate để sang Phase 1**: Tất cả 5 dòng `✅` phải xuất hiện. Nếu MMCV fail → đọc lại Phụ lục A.1 trong SPEC.md.

---

## Phase 1 — Data Pipeline & Baseline

> **Tham chiếu SPEC**: Mục 8 (Dữ liệu), Mục 9 Sprint 1

### Mục tiêu

Hiểu dataset Bosch, xây dựng DataLoader hoạt động, và thiết lập điểm số YOLO baseline làm "bia đỡ đạn" để so sánh sau này.

### Danh sách Task

**T1.1 — EDA Script** | File: `scripts/eda_bosch.py`

Implement hàm `analyze_bbox_distribution(yaml_path)` với các output bắt buộc:
- Histogram kích thước bbox (width, height) — lưu ra `logs/bbox_distribution.png`
- % đèn có `w < 10px` và `h < 10px` — in ra console
- Phân phối 4 classes (Green/Yellow/Red/Off) — lưu ra `logs/class_distribution.png`
- Số lượng đèn trung bình per image — in ra console (dùng để ước lượng N candidates cho batch)
- Min/max kích thước bbox — quan trọng để thiết kế anchor sizes

```bash
# Chạy EDA
python scripts/eda_bosch.py --yaml data/bosch/train.yaml
```

**T1.2 — CLASS_MAPPING** | File: `data/dataset.py`

Implement mapping từ 10+ class gốc Bosch về 4 class TTLD:
```python
CLASS_MAPPING = {
    'Green': 0, 'GreenLeft': 0, 'GreenRight': 0, 'GreenStraight': 0,
    'Yellow': 1,
    'Red': 2, 'RedLeft': 2, 'RedRight': 2, 'RedStraight': 2,
    'off': 3,
}
CLASS_NAMES = ['Green', 'Yellow', 'Red', 'Off']
```

**T1.3 — BoschDataset Class** | File: `data/dataset.py`

Implement class kế thừa `torch.utils.data.Dataset`:
- `__init__(yaml_path, transform, split)` — load toàn bộ YAML
- `_load_yaml(yaml_path)` — parse thành list of `{image_path, boxes}`
- `__getitem__(idx)` — trả về `(image_tensor, boxes_tensor)`
- `_parse_boxes(raw_boxes)` — convert sang `Tensor (N, 5): x_center, y_center, w, h, class_id`

Lưu ý xử lý: ảnh thiếu boxes (empty list → Tensor shape `(0, 5)`), ảnh bị corrupt (try-except).

**T1.4 — Augmentation Pipeline** | File: `data/transforms.py`

Implement các augmentation theo thứ tự:
1. `RandomHorizontalFlip(p=0.5)` — lật ngang, nhớ flip bbox x_center
2. `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)`
3. `Mosaic(p=0.3)` — ghép 4 ảnh, resize về 1280×720 (optional nhưng recommend)
4. `ToTensor()` + `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`

**T1.5 — DataLoader với Custom Collate** | File: `data/dataset.py`

```python
def custom_collate(batch):
    """Xử lý batch có số boxes khác nhau giữa các ảnh — KHÔNG dùng default_collate"""
    images, targets = zip(*batch)
    images = torch.stack(images)       # (B, 3, H, W) — OK vì ảnh cùng size
    # targets: list of Tensor (Ni, 5) — giữ nguyên dạng list
    return images, list(targets)

def create_dataloader(yaml_path, batch_size=16, num_workers=4, shuffle=True):
    dataset = BoschDataset(yaml_path, transform=get_train_transforms())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      collate_fn=custom_collate)
```

**T1.6 — YOLO Baseline Training**

Chạy YOLO gốc (không có TTLD modifications) trên Bosch dataset để có baseline metrics:
```bash
python train.py --config configs/m0_baseline.yaml --output logs/ablation/m0_baseline
python test.py  --config configs/m0_baseline.yaml \
                --weights checkpoints/m0_baseline_best.pth \
                --output results/ablation/m0_baseline_metrics.json
```

Config `configs/m0_baseline.yaml`:
```yaml
model:
  backbone: yolo26
  mode: baseline          # Không có TTLD modifications
training:
  epochs: 50              # Baseline chỉ cần ít epoch hơn
  batch_size: 16
  optimizer: AdamW
  lr: 1.0e-4
data:
  train_yaml: data/bosch/train.yaml
  val_yaml: data/bosch/val.yaml
  conf_threshold: 0.25    # Threshold mặc định YOLO
  num_workers: 4
```

### Kiểm thử Phase 1

**Test T1.A — DataLoader smoke test**
```python
def test_dataloader():
    loader = create_dataloader("data/bosch/train.yaml", batch_size=4)
    images, targets = next(iter(loader))

    assert images.shape == (4, 3, 720, 1280), f"Image shape sai: {images.shape}"
    assert isinstance(targets, list) and len(targets) == 4
    assert all(t.shape[1] == 5 for t in targets if len(t) > 0), "Box format sai (cần N×5)"
    assert not torch.isnan(images).any(), "NaN trong ảnh"
    print("✅ DataLoader test PASS")
```

**Test T1.B — EDA sanity check**
- Xác nhận: % đèn có `w < 10px` > 30% (dataset này nổi tiếng có nhiều tiny objects)
- Xác nhận: Class 'Red' chiếm > 40% (phổ biến nhất)
- Xác nhận: Số đèn trung bình per image > 2

**Test T1.C — Baseline metrics được ghi lại**
- File `results/ablation/m0_baseline_metrics.json` tồn tại
- Chứa keys: `ap50`, `apsmall`, `recall`, `precision`, `fpr`
- `apsmall` < 0.5 (expected — đây chính là vấn đề TTLD-Net giải quyết)

**Gate để sang Phase 2**: T1.A và T1.C đều PASS.

---

## Phase 2 — High-Recall Candidate Generator

> **Tham chiếu SPEC**: Mục 3 (Module 1), Mục 9 Sprint 2

### Mục tiêu

**Recall > 95%** — sinh ra nhiều ứng viên (N ~ vài nghìn per ảnh), ưu tiên không bỏ sót bất kỳ đèn thật nào. False Positives được chấp nhận và sẽ bị lọc ở Phase 4-5.

### Bối cảnh Kỹ thuật

Module này can thiệp vào YOLO backbone để trích xuất feature maps nông (P1, P2, P3) thay vì chỉ dùng (P3, P4, P5) như YOLO thông thường. Ba kỹ thuật quan trọng:
1. **Hạ conf_threshold = 0.05** (mặc định 0.25~0.5)
2. **Soft-NMS** thay thế Hard NMS (giữ lại đèn đứng sát nhau)
3. **Focal Loss γ=1.5, α=0.75** (phạt nặng bỏ sót, ưu tiên minority class)

### Danh sách Task

**T2.1 — Override YOLO Backbone Forward Pass** | File: `models/backbones/yolo26.py`

```python
class YOLO26Backbone(nn.Module):
    def forward(self, x):
        p1 = self.layer1(x)      # (B, C1, H/2, W/2)   stride 2
        p2 = self.layer2(p1)     # (B, C2, H/4, W/4)   stride 4
        p3 = self.layer3(p2)     # (B, C3, H/8, W/8)   stride 8
        p4 = self.layer4(p3)     # (B, C4, H/16, W/16) stride 16
        p5 = self.layer5(p4)     # (B, C5, H/32, W/32) stride 32

        # QUAN TRỌNG: trả về TẤT CẢ 5 levels, không chỉ (P3, P4, P5)
        return p1, p2, p3, p4, p5
```

Nếu dùng `ultralytics` YOLO: cần hook hoặc subclass, không sửa trực tiếp source.

**T2.2 — FPN + PANet Neck** | File: `models/necks/fpn_panet.py`

Top-down pathway (P5→P4→P3) + bottom-up pathway (P3→P4→P5). Output: enhanced P1, P2, P3 cho Local Branch. Đây là neck tiêu chuẩn, có thể dùng implementation sẵn từ `ultralytics` hoặc `mmdet`.

**T2.3 — Tiny Generator Head** | File: `models/heads/tiny_generator.py`

Input: P1 `(B, C1, H/2, W/2)`, P2 `(B, C2, H/4, W/4)`, P3 `(B, C3, H/8, W/8)`

Output:
- `candidates`: `List[Dict]` với mỗi candidate chứa `{bbox: (x,y,w,h), confidence, class_id}`
- `fcand`: `Tensor (B, N, 256)` — local feature per candidate

Implementation notes:
- Mỗi scale (P1, P2, P3) có detection head riêng
- Gộp candidates từ 3 scales, áp dụng Soft-NMS toàn cục
- Trích xuất feature vector 256-dim cho từng candidate bằng RoIAlign hoặc index sampling

**T2.4 — Soft-NMS** | File: `models/heads/tiny_generator.py`

```python
def soft_nms(boxes, scores, sigma=0.5, score_thresh=0.001):
    """
    Gaussian Soft-NMS: score_new = score * exp(-IoU^2 / sigma)
    Thay thế Hard NMS trong inference pipeline.
    sigma = 0.5 theo config m1_shallow.yaml
    """
    # Giảm score dần dần thay vì xóa hoàn toàn
    # Giữ lại box nếu score sau giảm vẫn > score_thresh
    ...
```

Override hàm NMS trong YOLO inference wrapper. Không sửa `ultralytics` source trực tiếp.

**T2.5 — Focal Loss** | File: `losses/focal_loss.py`

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.75):
        """
        gamma = 1.5  (giảm từ 2.0 để phạt FN nặng hơn)
        alpha = 0.75 (tăng trọng số đèn — minority class trong Bosch)
        """
```

Tham chiếu SPEC Mục 3.3(C) để hiểu lý do chọn các giá trị này.

**T2.6 — Config m1_shallow.yaml** | File: `configs/m1_shallow.yaml`

```yaml
model:
  backbone: yolo26
  use_shallow_features: true    # P1, P2, P3 extraction
  use_soft_nms: true
heads:
  tiny_generator:
    conf_threshold: 0.05        # CRITICAL: thấp hơn 5x so với default
    soft_nms_sigma: 0.5
    fcand_dim: 256
losses:
  focal_gamma: 1.5
  focal_alpha: 0.75
```

### Kiểm thử Phase 2

**Test T2.A — Forward pass shape test**
```python
def test_generator_forward():
    B = 2
    model = TinyGenerator(...)
    P1 = torch.randn(B, C1, 360, 640)
    P2 = torch.randn(B, C2, 180, 320)
    P3 = torch.randn(B, C3, 90, 160)

    candidates, fcand = model(P1, P2, P3)

    N = len(candidates[0])  # số candidates ảnh đầu
    assert N > 100, f"Quá ít candidates: {N} (expected > 100 với conf=0.05)"
    assert fcand.shape == (B, N, 256), f"fcand shape sai: {fcand.shape}"
    assert not torch.isnan(fcand).any(), "NaN trong fcand"
    print(f"✅ Generator test PASS — N = {N} candidates")
```

**Test T2.B — Recall target verification**
```bash
# Chạy inference trên validation set với conf=0.05
python test.py --config configs/m1_shallow.yaml --weights checkpoints/m1_shallow_best.pth \
               --output results/ablation/m1_shallow_metrics.json
```
- `recall >= 0.95` — **Gate bắt buộc**. Nếu < 0.95: hạ thêm conf_threshold, kiểm tra Soft-NMS

**Test T2.C — Soft-NMS vs Hard-NMS comparison**
- Lấy 1 ảnh có nhiều đèn trên cùng 1 cột (từ Bosch dataset)
- So sánh số candidates được giữ lại: Soft-NMS phải giữ >= Hard-NMS
- Log kết quả vào `logs/nms_comparison.txt`

**Test T2.D — Focal Loss convergence**
- Chạy 10 epoch với toy dataset (100 ảnh)
- `L_det` phải giảm đều từ epoch 1
- Không xuất hiện NaN hoặc Inf trong loss

**Gate để sang Phase 3 & 4**: T2.B (Recall > 95%) là bắt buộc.

---

## Phase 3 — Semantic Context Branch

> **Tham chiếu SPEC**: Mục 4 (Module 2)

### Mục tiêu

Cung cấp **"tầm nhìn rộng"** — context features từ P4 và P5 để xác minh từng candidate. Module này nhìn "toàn cảnh" (cột đèn, đường phố, bầu trời, ngã tư) trong khi Phase 2 nhìn "cận cảnh".

### Bối cảnh Kỹ thuật

P4 (stride 16, 45×80) mang thông tin cấu trúc vùng. P5 (stride 32, 23×40) mang thông tin toàn cảnh layout. Cần chiếu cả hai về cùng 256-dim trước khi đưa vào Phase 4.

### Danh sách Task

**T3.1 — FeatureProjection Module** | File: `models/necks/fpn_panet.py` hoặc `models/heads/implicit_topo.py`

```python
class FeatureProjection(nn.Module):
    """Conv 1×1: không thay đổi spatial resolution, chỉ đổi số kênh về 256"""
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.proj(x)
```

Cần 3 projectors riêng biệt:
- `proj_p4 = FeatureProjection(C4, 256)` → `Fctx_p4: (B, 256, 45, 80)`
- `proj_p5 = FeatureProjection(C5, 256)` → `Fctx_p5: (B, 256, 23, 40)`
- `proj_cand = FeatureProjection(C_cand, 256)` → `fcand_proj: (B, N, 256)`

**T3.2 — Xác nhận Channel Dimensions của Backbone**

Trước khi implement projections, xác nhận C4 và C5 từ backbone thực tế:
```python
backbone = YOLO26Backbone()
dummy = torch.randn(1, 3, 720, 1280)
p1, p2, p3, p4, p5 = backbone(dummy)
print(f"C4 = {p4.shape[1]}, C5 = {p5.shape[1]}")  # Log giá trị này
```
Ghi vào `logs/backbone_channels.txt` để tham chiếu sau.

**T3.3 — Context Branch Integration**

Đảm bảo trong forward pass tổng thể:
1. Backbone trả về P4, P5 (đã có từ T2.1)
2. Áp dụng `proj_p4` và `proj_p5` để có `Fctx_p4` và `Fctx_p5`
3. Pass `[Fctx_p4, Fctx_p5]` vào Phase 4 (Topology Sampler)

### Kiểm thử Phase 3

**Test T3.A — Projection shape test**
```python
def test_feature_projection():
    B, C4, C5 = 2, 512, 1024  # thay bằng giá trị thực từ T3.2
    proj_p4 = FeatureProjection(C4, 256)
    proj_p5 = FeatureProjection(C5, 256)

    P4 = torch.randn(B, C4, 45, 80)
    P5 = torch.randn(B, C5, 23, 40)

    Fctx_p4 = proj_p4(P4)
    Fctx_p5 = proj_p5(P5)

    assert Fctx_p4.shape == (B, 256, 45, 80), f"Fctx_p4 shape sai: {Fctx_p4.shape}"
    assert Fctx_p5.shape == (B, 256, 23, 40), f"Fctx_p5 shape sai: {Fctx_p5.shape}"
    assert not torch.isnan(Fctx_p4).any()
    assert not torch.isnan(Fctx_p5).any()
    print("✅ FeatureProjection test PASS")
```

**Test T3.B — No gradient blocking**
```python
# Đảm bảo gradient chạy qua projection layers
Fctx_p4 = proj_p4(P4)
loss = Fctx_p4.sum()
loss.backward()
assert proj_p4.proj.weight.grad is not None, "❌ Gradient bị block tại FeatureProjection"
print("✅ Gradient flow qua FeatureProjection: OK")
```

**Gate để sang Phase 4**: T3.A PASS.

---

## Phase 4 — Implicit Topology Sampler

> **Tham chiếu SPEC**: Mục 5 (Module 3) — ĐỌC KỸ toàn bộ mục 5 trước khi bắt đầu

### Mục tiêu

**Đây là module kỹ thuật quan trọng nhất của TTLD-Net.** Thay vì cắt vùng ảnh tĩnh, module này tự học vươn đến các vùng có giá trị xác thực cao nhất thông qua Deformable Attention.

Ví dụ: Khi đánh giá candidate "đèn tại (x, y)", hệ thống có thể học rằng "nếu phía dưới có cột đèn thẳng, đây nhiều khả năng là đèn thật" — các "xúc tu" tự vươn đến cột đèn bên dưới mà không cần nhãn tường minh.

### Điều kiện Tiên quyết

- Phase 2 PASS (có `fcand` và tọa độ candidates)
- Phase 3 PASS (có `Fctx_p4`, `Fctx_p5` đã projected về 256-dim)
- MMCV được cài đúng với CUDA support (Phase 0 T0.3)

### Danh sách Task

**T4.1 — Chuẩn bị Reference Points** | File: `models/heads/implicit_topo.py`

Từ `candidates` list, trích xuất tọa độ normalized `pq: (B, N, 2)`:
```python
def extract_reference_points(candidates, image_size=(720, 1280)):
    """
    Chuyển bbox (x_center, y_center, w, h) về tọa độ normalized [0, 1]
    pq[b, i, :] = [x_center_normalized, y_center_normalized]
    """
    H, W = image_size
    pq = []
    for c in candidates:
        x_norm = c['bbox'][0] / W
        y_norm = c['bbox'][1] / H
        pq.append([x_norm, y_norm])
    return torch.tensor(pq, dtype=torch.float32)  # (N, 2)
```

**T4.2 — Offset Predictor và Attention Weight Predictor**

```python
# Trong class ImplicitTopologySampler.__init__:
self.offset_predictor  = nn.Linear(d_model, n_points * 2)  # (B,N,256) → (B,N,K*2)
self.attention_weights = nn.Linear(d_model, n_points)       # (B,N,256) → (B,N,K)
self.proj = nn.Linear(d_model, d_model)                     # Ma trận W
```

Lưu ý: `attention_weights` cần qua `.softmax(dim=-1)` để tổng = 1.

**T4.3 — MultiScaleDeformableAttention Integration** | File: `models/heads/implicit_topo.py`

```python
from mmcv.ops import MultiScaleDeformableAttention

class ImplicitTopologySampler(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_points=4):
        super().__init__()
        self.deform_attn = MultiScaleDeformableAttention(
            embed_dims=d_model,
            num_heads=n_heads,
            num_points=n_points,
        )
        # ... offset_predictor, attention_weights, proj

    def forward(self, fcand, Fctx_list, pq):
        """
        Args:
            fcand:      (B, N, 256)
            Fctx_list:  [(B, 256, 45, 80), (B, 256, 23, 40)]
            pq:         (B, N, 2) — normalized [0,1]
        Returns:
            zi:         (B, N, 256)
        """
        offsets = self.offset_predictor(fcand)            # (B, N, K*2)
        attn_w  = self.attention_weights(fcand).softmax(-1)  # (B, N, K)
        offsets = offsets.view(*offsets.shape[:2], self.n_points, 2)  # (B, N, K, 2)

        # Flatten Fctx_list cho MultiScaleDeformableAttention
        # Tham chiếu MMCV docs cho đúng format input_flatten
        zi = self.deform_attn(
            query=fcand,
            reference_points=pq.unsqueeze(2),  # (B, N, 1, 2)
            input_flatten=...,                  # flatten Fctx
            input_spatial_shapes=...,           # [(45,80), (23,40)]
            input_level_start_index=...,
        )
        return zi  # (B, N, 256)
```

Đọc kỹ MMCV docs để hiểu đúng format `input_flatten` và `input_spatial_shapes`.

**T4.4 — Unit Test** (viết test TRƯỚC KHI integrate)

Xem section kiểm thử bên dưới — **viết test T4.A trước**, sau đó mới viết implementation.

### Kiểm thử Phase 4

**Test T4.A — Shape và NaN test** (từ SPEC Mục 5.6)
```python
def test_topology_sampler():
    B, N, K = 2, 100, 4
    sampler = ImplicitTopologySampler(d_model=256, n_heads=8, n_points=K)
    sampler = sampler.cuda()

    fcand    = torch.randn(B, N, 256).cuda()
    Fctx_p4  = torch.randn(B, 256, 45, 80).cuda()
    Fctx_p5  = torch.randn(B, 256, 23, 40).cuda()
    pq       = torch.rand(B, N, 2).cuda()  # normalized [0,1]

    zi = sampler(fcand, [Fctx_p4, Fctx_p5], pq)

    assert zi.shape == (B, N, 256), f"❌ Expected (B,N,256), got {zi.shape}"
    assert not torch.isnan(zi).any(), "❌ NaN trong topology embedding"
    assert not torch.isinf(zi).any(), "❌ Inf trong topology embedding"
    print(f"✅ ImplicitTopologySampler PASS — zi.shape = {zi.shape}")
```

**Test T4.B — Gradient flow test**
```python
zi = sampler(fcand, [Fctx_p4, Fctx_p5], pq)
loss = zi.sum()
loss.backward()
assert sampler.offset_predictor.weight.grad is not None, "❌ Gradient không chạy qua offset_predictor"
assert sampler.attention_weights.weight.grad is not None, "❌ Gradient không chạy qua attention_weights"
print("✅ Gradient flow PASS")
```

**Test T4.C — Memory test**
```python
# Test với batch size và N lớn hơn để kiểm tra memory
B, N = 4, 500
# Nếu OOM: giảm K từ 8 → 4, hoặc giảm N
# Ghi lại peak VRAM: torch.cuda.max_memory_allocated() / 1e9
```

**Test T4.D — Attention offsets có ý nghĩa**
```python
# Sau khi train vài epoch, visualize offsets
# Xem offsets có tập trung vào các hướng có ý nghĩa (xuống phía cột đèn) không
# Dùng scripts/draw_attention.py (implement ở Phase 7)
```

**Gate để sang Phase 5**: T4.A và T4.B đều PASS.

---

## Phase 5 — Verification & Contrastive Learning

> **Tham chiếu SPEC**: Mục 6 (Module 4), Mục 7 (Hàm Loss)

### Mục tiêu

**Màng lọc cuối cùng** — phân loại mỗi candidate thành `Hợp lệ` hoặc `Loại bỏ` dựa trên cả ngoại hình (`fcand`) và topology context (`zi`). InfoNCE Loss ép latent space phân tách rõ "đèn thật" vs "đèn giả".

### Bối cảnh Kỹ thuật

- `vi = concat(fcand, zi)` → shape `(B, N, 512)`
- MLP: `512 → 256 → 128 → 1` với ReLU và Sigmoid cuối
- Loss: `L_verify = BCE(p_valid, labels)` + `L_topology = InfoNCE(zi, z_pos, z_neg)`

### Danh sách Task

**T5.1 — Verification MLP** | File: `models/heads/verification.py`

```python
class VerificationMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, fcand, zi):
        vi = torch.cat([fcand, zi], dim=-1)  # (B, N, 512)
        return self.mlp(vi)                  # (B, N, 1)
```

**T5.2 — InfoNCE Loss** | File: `losses/infonce_loss.py`

Công thức: `L = -log[ exp(sim(zi, z+)/τ) / Σ_j exp(sim(zi, z-_j)/τ) ]`

```python
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature  # τ = 0.07

    def forward(self, zi, z_pos, z_neg_list):
        # CRITICAL: phải normalize trước dot product
        zi_norm  = F.normalize(zi, dim=-1)     # (B, N, 256)
        zp_norm  = F.normalize(z_pos, dim=-1)  # (B, N, 256)
        zn_norm  = F.normalize(z_neg_list, dim=-1)  # (B, N, M, 256)

        pos_sim = (zi_norm * zp_norm).sum(-1) / self.tau  # (B, N)
        neg_sim = torch.einsum('bnd,bnmd->bnm', zi_norm, zn_norm) / self.tau  # (B, N, M)

        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (B, N, 1+M)
        labels = torch.zeros(logits.shape[:2], dtype=torch.long)  # positive = index 0
        return F.cross_entropy(logits.view(-1, 1+neg_sim.shape[-1]), labels.view(-1))
```

**Lỗi phổ biến**: Nếu `L_topology` không giảm → kiểm tra xem `F.normalize` đã được gọi chưa (xem SPEC Phụ lục A.1).

**T5.3 — Hard Negative Mining** | File: `models/heads/verification.py`

```python
def mine_hard_negatives(candidates, n_hard=32):
    """
    Lọc FP candidates có confidence CAO NHẤT làm Hard Negatives.
    "Trông giống đèn thật nhất nhưng thực ra là nhiễu" = hard to learn from.
    """
    fp_candidates = [c for c in candidates if c['is_false_positive']]
    fp_candidates.sort(key=lambda c: c['confidence'], reverse=True)
    return fp_candidates[:n_hard]
```

Lưu ý: Cần có ground truth để biết đâu là FP. Trong training loop, so sánh candidates với GT boxes (IoU threshold = 0.5).

**T5.4 — Tạo Positive Pairs cho InfoNCE**

Positive pair `(zi, z_pos)`:
- `zi`: topology embedding của candidate là đèn thật (True Positive)
- `z_pos`: context embedding tại vị trí đèn thật trong một ảnh khác trong batch

Cách đơn giản: augment cùng ảnh (flip, color jitter) → positive pair tự nhiên.

### Kiểm thử Phase 5

**Test T5.A — Verification MLP shape test**
```python
def test_verification_mlp():
    B, N = 2, 100
    mlp = VerificationMLP(input_dim=512)
    fcand = torch.randn(B, N, 256)
    zi    = torch.randn(B, N, 256)

    p_valid = mlp(fcand, zi)
    assert p_valid.shape == (B, N, 1), f"Shape sai: {p_valid.shape}"
    assert (p_valid >= 0).all() and (p_valid <= 1).all(), "Giá trị ngoài [0,1]"
    print("✅ VerificationMLP test PASS")
```

**Test T5.B — InfoNCE Loss convergence test**
```python
def test_infonce_convergence():
    """InfoNCE phải nhỏ hơn khi positive pairs gần nhau hơn negative pairs"""
    loss_fn = InfoNCELoss(temperature=0.07)
    B, N, M = 2, 10, 5

    # Case 1: positive và anchor rất gần nhau
    zi    = F.normalize(torch.randn(B, N, 256), dim=-1)
    z_pos = zi + 0.01 * torch.randn(B, N, 256)  # gần anchor
    z_neg = F.normalize(torch.randn(B, N, M, 256), dim=-1)  # random (xa)
    loss_easy = loss_fn(zi, z_pos, z_neg)

    # Case 2: positive random (xa anchor)
    z_pos_hard = F.normalize(torch.randn(B, N, 256), dim=-1)
    loss_hard = loss_fn(zi, z_pos_hard, z_neg)

    assert loss_easy < loss_hard, f"❌ InfoNCE không hoạt động: {loss_easy:.4f} >= {loss_hard:.4f}"
    print(f"✅ InfoNCE convergence PASS: easy={loss_easy:.4f} < hard={loss_hard:.4f}")
```

**Test T5.C — L_topology monitoring**

Trong training (Phase 6), `L_topology` phải:
- Bắt đầu giảm trước epoch 5
- Ổn định sau epoch 10-20
- KHÔNG bao giờ NaN

Nếu không giảm: kiểm tra lại `F.normalize`, Hard Negative Mining, và τ = 0.07.

**Gate để sang Phase 6**: T5.A và T5.B đều PASS.

---

## Phase 6 — End-to-End Training

> **Tham chiếu SPEC**: Mục 9 Sprint 5, Mục 7 (Loss), Phụ lục A.2, A.3

### Mục tiêu

Ghép toàn bộ pipeline thành một model hoàn chỉnh và huấn luyện end-to-end trên Bosch dataset.

### Bối cảnh Kỹ thuật

Loss tổng thể:
```
L_total = L_det + λ1 * L_topology + λ2 * L_verify
         (λ1 = λ2 = 1.0, tune sau nếu cần)
```

Training config:
- Optimizer: AdamW, lr = 1e-4
- Scheduler: Cosine Annealing với 5 epoch warmup
- Epochs: 100 (đủ cho TTLD-Net hội tụ)
- Batch size: 32 (giảm xuống 8-16 nếu OOM)
- Gradient clipping: max_norm = 1.0

### Danh sách Task

**T6.1 — Main Training Script** | File: `train.py`

```python
# train.py — cấu trúc tổng thể
def train(config_path: str, output_dir: str):
    cfg = load_config(config_path)

    # 1. Data
    train_loader = create_dataloader(cfg.data.train_yaml, cfg.training.batch_size)
    val_loader   = create_dataloader(cfg.data.val_yaml, shuffle=False)

    # 2. Model
    backbone  = YOLO26Backbone()
    neck      = FPNPANet(backbone)
    generator = TinyGenerator(...)
    projector_p4 = FeatureProjection(C4, 256)
    projector_p5 = FeatureProjection(C5, 256)
    topo_sampler = ImplicitTopologySampler(...)
    verifier     = VerificationMLP(...)

    model = TTLDNet(backbone, neck, generator, projector_p4, projector_p5,
                    topo_sampler, verifier)

    # 3. Loss functions
    focal_loss  = FocalLoss(gamma=cfg.losses.focal_gamma, alpha=cfg.losses.focal_alpha)
    infonce_loss = InfoNCELoss(temperature=0.07)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs,
                                  eta_min=1e-6)

    # 5. Training loop
    for epoch in range(cfg.training.epochs):
        loss_det, loss_topo, loss_verify = train_one_epoch(
            model, train_loader, optimizer, focal_loss, infonce_loss, cfg
        )
        val_metrics = evaluate(model, val_loader)

        # Gradient clipping — CRITICAL
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scheduler.step()
        log_to_tensorboard(epoch, loss_det, loss_topo, loss_verify, val_metrics)
        save_best_checkpoint(model, val_metrics, output_dir)
```

**T6.2 — Loss Combination trong Training Loop**

```python
loss_total = loss_det + cfg.loss.lambda1 * loss_topology + cfg.loss.lambda2 * loss_verify
optimizer.zero_grad()
loss_total.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**T6.3 — TensorBoard Logging** (theo dõi 3 loss riêng biệt)

```python
writer.add_scalar('Loss/detection',  loss_det.item(),    epoch)
writer.add_scalar('Loss/topology',   loss_topo.item(),   epoch)
writer.add_scalar('Loss/verify',     loss_verify.item(), epoch)
writer.add_scalar('Loss/total',      loss_total.item(),  epoch)
writer.add_scalar('Metrics/AP50',    val_metrics['ap50'],    epoch)
writer.add_scalar('Metrics/APsmall', val_metrics['apsmall'], epoch)
writer.add_scalar('Metrics/Recall',  val_metrics['recall'],  epoch)
writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
```

**T6.4 — Evaluation Script** | File: `test.py`

```python
def evaluate(model, dataloader, conf_threshold=0.5):
    """
    conf_threshold = 0.5 ở đây (STRICT — khác với 0.05 trong Stage 1)
    """
    metrics = {
        'ap50':      compute_ap50(preds, targets),
        'apsmall':   compute_apsmall(preds, targets),
        'recall':    compute_recall(preds, targets),
        'precision': compute_precision(preds, targets),
        'fpr':       compute_false_positive_rate(preds, targets),
    }
    return metrics
```

**T6.5 — Multi-GPU Setup (nếu có)** | Optional

```python
# Distributed Data Parallel
if torch.cuda.device_count() > 1:
    model = torch.nn.parallel.DistributedDataParallel(model)
```

**T6.6 — Config m4_full_ttld.yaml** | File: `configs/m4_full_ttld.yaml`

```yaml
model:
  backbone: yolo26
  neck: fpn_panet
  heads: [tiny_generator, implicit_topo_sampler, verification_mlp]

training:
  epochs: 100
  batch_size: 32
  optimizer: AdamW
  lr: 1.0e-4
  scheduler: CosineAnnealing
  warmup_epochs: 5
  grad_clip_norm: 1.0

loss:
  lambda1: 1.0
  lambda2: 1.0

data:
  conf_threshold: 0.05
  soft_nms_sigma: 0.5
  focal_gamma: 1.5
  focal_alpha: 0.75
  num_workers: 4
  pin_memory: true
```

### Kiểm thử Phase 6

**Test T6.A — End-to-end forward pass**
```python
def test_e2e_forward():
    B = 2
    model = TTLDNet(...)
    dummy_images = torch.randn(B, 3, 720, 1280).cuda()

    candidates, fcand, zi, p_valid = model(dummy_images)

    assert p_valid.shape[-1] == 1
    assert not torch.isnan(p_valid).any()
    print("✅ End-to-end forward PASS")
```

**Test T6.B — Training Health Check** (theo SPEC Phụ lục A.3)

Sau 20 epoch đầu tiên, mở TensorBoard và kiểm tra:
- `Loss/detection`: giảm đều từ epoch 1 ✅
- `Loss/topology`: bắt đầu giảm trước epoch 10, ổn định sau epoch 20 ✅
- `Loss/verify`: bắt đầu giảm từ epoch 5-10 ✅
- Không có NaN trong bất kỳ loss nào ✅
- Learning rate giảm đúng theo Cosine Annealing ✅

Nếu `Loss/topology` không giảm → xem lại T5.B và kiểm tra InfoNCE implementation.

**Test T6.C — Final Model Metrics**

Chạy `test.py` trên validation set với model tốt nhất:
```bash
python test.py --config configs/m4_full_ttld.yaml \
               --weights checkpoints/best_model.pth \
               --output results/ablation/m4_full_ttld_metrics.json
```

So sánh với baseline (M0):
- `recall >= 0.95` ✅
- `apsmall > baseline_apsmall + 0.03` ✅ (tăng ít nhất 3%)
- `fpr < baseline_fpr * 0.85` ✅ (giảm ít nhất 15%)

**Gate để sang Phase 7**: T6.C PASS với cả 3 metrics trên.

---

## Phase 7 — Ablation Study & Visualization

> **Tham chiếu SPEC**: Mục 13 (Ablation Study), Mục 9 Sprint 6

### Mục tiêu

Chứng minh đóng góp của từng component qua 5 biến thể thực nghiệm (M0–M4). Tạo hình ảnh heatmap "xúc tu" Deformable Attention để visualize cơ chế hoạt động.

### 5 Biến thể Ablation

| Config | File | Components | Mục đích |
|---|---|---|---|
| M0 | `m0_baseline.yaml` | YOLO thuần | Điểm mốc gốc |
| M1 | `m1_shallow.yaml` | + P1/P2/P3, Soft-NMS, conf=0.05 | Giá trị của shallow features |
| M2 | `m2_focal.yaml` | + Focal Loss γ=1.5, α=0.75 | Giá trị của Focal tuning |
| M3 | `m3_topology.yaml` | + Implicit Topology Sampler | Giá trị của Deformable Attention |
| M4 | `m4_full_ttld.yaml` | + InfoNCE + Verification | Full model |

### Danh sách Task

**T7.1 — 5 Config Files** | Files: `configs/m*.yaml`

Tạo từng file config dựa trên M4, lần lượt tắt các components:
- M3: comment out `infonce_loss` và `verification_mlp`
- M2: comment out thêm `implicit_topo_sampler`
- M1: comment out thêm `focal_loss` tuning
- M0: comment out tất cả TTLD modifications

**T7.2 — Ablation Runner Script** | File: `scripts/run_ablation.py`

```python
# scripts/run_ablation.py
configs = ['m0_baseline', 'm1_shallow', 'm2_focal', 'm3_topology', 'm4_full_ttld']

for config_name in configs:
    print(f"\n{'='*50}")
    print(f"Running ablation: {config_name}")
    os.system(f"python train.py --config configs/{config_name}.yaml "
              f"--output logs/ablation/{config_name}")
    os.system(f"python test.py --config configs/{config_name}.yaml "
              f"--weights checkpoints/{config_name}_best.pth "
              f"--output results/ablation/{config_name}_metrics.json")
```

**T7.3 — Kết quả Tổng hợp** | File: `scripts/compare_ablation.py`

```python
# Đọc tất cả *_metrics.json và tạo bảng so sánh
# Output: results/ablation_comparison_table.md
```

**T7.4 — Attention Visualization** | File: `scripts/draw_attention.py`

Visualize các sampling points (offsets) của Deformable Attention lên ảnh gốc:
- Với mỗi candidate đèn, vẽ K điểm lấy mẫu (các "xúc tu")
- Tô màu theo attention weight (weight cao = màu sáng hơn)
- Overlay lên ảnh gốc bằng OpenCV

```python
def draw_deformable_attention(image, candidate, offsets, attn_weights):
    """
    offsets:      (K, 2) — vị trí absolute của K sampling points
    attn_weights: (K,)   — attention weight cho mỗi điểm
    """
    for k in range(len(offsets)):
        x, y = int(offsets[k, 0]), int(offsets[k, 1])
        alpha = float(attn_weights[k])
        color = (int(255*alpha), int(100*(1-alpha)), 0)  # high weight = đỏ
        cv2.circle(image, (x, y), radius=3, color=color, thickness=-1)
    return image
```

**T7.5 — Visualization Utilities** | File: `utils/visualize.py`

```python
def draw_predictions(image, boxes, scores, labels, class_names):
    """Vẽ bounding boxes và labels lên ảnh"""

def draw_attention_heatmap(image, attention_map):
    """Overlay heatmap attention lên ảnh"""
```

**T7.6 — README.md và Code Cleanup**

README phải có:
- Setup instructions (conda/pip, MMCV install với CUDA)
- Quick start: training command, evaluation command
- Kết quả ablation table
- Hình ảnh minh họa (attention heatmap, sample detections)
- Citation và acknowledgement

### Kiểm thử Phase 7

**Test T7.A — Tất cả 5 ablation runs hoàn thành**
```bash
# Kiểm tra tất cả metrics files tồn tại
ls results/ablation/m{0,1,2,3,4}*_metrics.json
```

**Test T7.B — Xu hướng metrics đúng hướng**

```python
# Từ 5 metrics files, kiểm tra:
# APsmall: M0 < M1 < M2 < M3 < M4 (tăng dần)
# FPR: M0 > M1 > M2 > M3 > M4 (giảm dần)
# Recall: M0 < M1, và M1..M4 tất cả > 0.95
```

Nếu xu hướng không đúng → phân tích nguyên nhân và báo cáo trong paper.

**Test T7.C — Attention visualization hợp lý**

- Load model M4 (full TTLD-Net)
- Chọn 10 ảnh validation có đèn thật
- Visualize attention offsets cho các True Positive candidates
- Kiểm tra bằng mắt: sampling points có tập trung gần cột đèn/khu vực hợp lý không?

---

## Phase 8 — Cross-Dataset Generalization & Robustness Evaluation

> **Tham chiếu SPEC**: Mục 14 (Cross-Dataset Generalization & Robustness Evaluation)

### Mục tiêu

Chứng minh TTLD-Net thực sự giải quyết được **Domain Shift** (Mục 1.2 SPEC), không chỉ overfit vào phân phối Bosch. Chạy zero-shot inference (không train lại) trên 4 dataset ngoài, đo robustness với corruption tổng hợp, đo latency thực tế, và kiểm tra ý nghĩa thống kê qua nhiều seed. Đây là bước bắt buộc trước khi viết paper draft.

**Không train hoặc fine-tune trên bất kỳ dataset nào trong phase này** — chỉ dùng checkpoint đã có từ Phase 6/7.

### Danh sách Task

**T8.1 — Chuẩn bị Dataset Ngoài** | File: `scripts/prepare_external_datasets.py`

Download và convert 4 dataset về format tương thích `BoschDataset`:
- DTLD (DriveU) — Đức, 2MP, có pictogram + trạng thái vàng-đỏ
- S2TLD (SJTU) — Trung Quốc, 5 class (thêm "wait-on")
- LISA — Mỹ
- Cityscapes TL++ (CSTL) — Đức

Với mỗi dataset, viết hàm `remap_classes_to_ttld(raw_label) -> {0:Green, 1:Yellow, 2:Red, 3:Off}` và **ghi rõ docstring** quy tắc mapping cho từng class không tương thích (xem SPEC Mục 14.6). Loại bỏ nhãn không thể map (vd pictogram arrow riêng biệt) — không đoán.

```bash
python scripts/prepare_external_datasets.py --dataset dtld --output data/external/dtld/
python scripts/prepare_external_datasets.py --dataset s2tld --output data/external/s2tld/
python scripts/prepare_external_datasets.py --dataset lisa --output data/external/lisa/
python scripts/prepare_external_datasets.py --dataset cstl --output data/external/cstl/
```

**T8.2 — Cross-Dataset Evaluation Script** | File: `scripts/eval_cross_dataset.py`

Implement Protocol A + B + C từ SPEC Mục 14.3:
- Load checkpoint `m0_baseline_best.pth` và `m4_full_ttld_best.pth`
- Chạy inference zero-shot trên từng dataset ngoài (conf_threshold **strict**, giống `test.py` ở Phase 6 — không phải 0.05 của Stage 1)
- Tính AP50/APsmall/Recall/Precision/FPR — cả tổng và size-stratified (`<8px`, `8-16px`, `16-32px`)
- Nếu dataset có metadata ngày/đêm (DTLD, LISA) → tách thêm theo điều kiện

```bash
python scripts/eval_cross_dataset.py --weights checkpoints/m0_baseline_best.pth --dataset dtld --output results/cross_dataset/dtld_m0.json
python scripts/eval_cross_dataset.py --weights checkpoints/m4_full_ttld_best.pth --dataset dtld --output results/cross_dataset/dtld_m4.json
# lặp lại cho s2tld, lisa, cstl
```

**T8.3 — Robustness / Corruption Script** | File: `scripts/eval_robustness.py`

Implement Protocol D: áp 5 loại corruption (Gaussian noise, motion blur, gamma thấp, fog synthetic, JPEG compression) × 3 severity level lên chính **Bosch test set gốc** (không cần dataset mới). Đo % giảm APsmall so với ảnh sạch cho M0 và M4.

```python
CORRUPTIONS = ["gaussian_noise", "motion_blur", "low_gamma", "synthetic_fog", "jpeg_compress"]
SEVERITIES = [1, 2, 3]
```

**T8.4 — Latency Benchmark** | File: `scripts/benchmark_latency.py`

Đo trên RTX 4090 thực tế, batch_size=1, 100 lần warm-up + 200 lần đo trung bình:
- Params (M), FLOPs (G) — dùng `torchinfo` hoặc `fvcore`
- ms/frame, FPS cho M0 và M4
- Ghi rõ có/không dùng gradient checkpointing khi đo (phải tắt khi benchmark inference)

**T8.5 — Multi-Seed Variance** | File: `scripts/eval_multi_seed.py`

Train lại M4 với 2 seed khác (ngoài seed gốc đã có ở Phase 6) → tổng 3 seed. Tính mean ± std cho AP50/APsmall/FPR trên Bosch test set.

> Đây là task duy nhất trong Phase 8 có train lại — nhưng chỉ train M4 trên Bosch (đúng dataset gốc), không train trên dataset ngoài.

### Kiểm thử Phase 8

**Test T8.A — Dataset Conversion Sanity Check**
```python
# Với mỗi dataset đã convert, kiểm tra:
assert len(dataset) > 0
sample_img, sample_boxes = dataset[0]
assert sample_boxes.shape[-1] == 5  # x,y,w,h,class
assert sample_boxes[:, -1].max() <= 3  # class_id trong {0,1,2,3}
```

**Test T8.B — Cross-Dataset Metrics Tồn tại**
```bash
ls results/cross_dataset/{dtld,s2tld,lisa,cstl}_m{0,4}.json
```
Tối thiểu 3/4 dataset phải có kết quả đầy đủ (một dataset có thể fail do vấn đề license/download — ghi rõ lý do nếu bỏ qua).

**Test T8.C — Gap Analysis**
```python
# Với mỗi dataset ngoài, so sánh gap giữa M0 và M4:
gap_m0 = metric_bosch_test["apsmall"] - metric_external["apsmall"]  # (M0)
gap_m4 = metric_bosch_test["apsmall"] - metric_external["apsmall"]  # (M4)
# Kỳ vọng: gap_m4 < gap_m0 (M4 generalize tốt hơn M0, ít bị domain shift hơn)
# Nếu không đúng hướng → phân tích nguyên nhân, KHÔNG che giấu trong paper
```

**Test T8.D — Robustness & Latency Files Tồn tại**
```bash
ls results/cross_dataset/robustness_corruption.json results/cross_dataset/latency_benchmark.json results/cross_dataset/multi_seed_variance.json
```

**Gate để coi Phase 8 hoàn thành**: T8.A, T8.B (≥3/4 dataset), T8.D đều PASS. T8.C không bắt buộc đúng hướng 100% nhưng bắt buộc phải được phân tích và báo cáo trung thực.

---

## Tổng hợp Checklist Toàn Dự án

### Phase Gates — Phải PASS trước khi sang phase tiếp theo

```
[ ] Phase 0 Gate: test_env.py — tất cả ✅ (đặc biệt MMCV CUDA)
[ ] Phase 1 Gate: DataLoader test PASS + baseline metrics file tồn tại
[ ] Phase 2 Gate: Recall > 95% trên validation set
[ ] Phase 3 Gate: FeatureProjection shape test PASS
[ ] Phase 4 Gate: ImplicitTopologySampler shape + gradient test PASS
[ ] Phase 5 Gate: VerificationMLP + InfoNCE convergence test PASS
[ ] Phase 6 Gate: Final metrics — APsmall +3%, FPR -15%, Recall > 95%
[ ] Phase 7 Gate: 5 ablation runs hoàn thành, xu hướng đúng hướng
[ ] Phase 8 Gate: zero-shot inference PASS trên ≥3/4 dataset ngoài (DTLD/S2TLD/LISA/CSTL),
                  size-stratified + robustness + latency + multi-seed đều có file kết quả
```

### Files Cần Tạo (theo thứ tự)

```
Phase 0:  requirements.txt
Phase 1:  scripts/eda_bosch.py
          data/dataset.py        (BoschDataset, create_dataloader, custom_collate)
          data/transforms.py
          configs/m0_baseline.yaml
Phase 2:  models/backbones/yolo26.py
          models/necks/fpn_panet.py
          models/heads/tiny_generator.py
          losses/focal_loss.py
          configs/m1_shallow.yaml
          configs/m2_focal.yaml
Phase 3:  [FeatureProjection — trong implicit_topo.py hoặc fpn_panet.py]
Phase 4:  models/heads/implicit_topo.py
          configs/m3_topology.yaml
Phase 5:  models/heads/verification.py
          losses/infonce_loss.py
Phase 6:  train.py
          test.py
          configs/m4_full_ttld.yaml
Phase 7:  scripts/run_ablation.py
          scripts/draw_attention.py
          scripts/compare_ablation.py
          utils/visualize.py
          README.md
Phase 8:  scripts/prepare_external_datasets.py
          scripts/eval_cross_dataset.py
          scripts/eval_robustness.py
          scripts/benchmark_latency.py
          scripts/eval_multi_seed.py
          data/external/{dtld,s2tld,lisa,cstl}/
```

### Hyperparameter Reference (không phải đoán — đây là spec)

| Parameter | Value | Lý do | Phase |
|---|---|---|---|
| `conf_threshold` | 0.05 | Recall > 95% cho tiny objects | Phase 2 |
| `soft_nms_sigma` | 0.5 | Giữ đèn đứng sát nhau | Phase 2 |
| `focal_gamma` | 1.5 | Phạt FN nặng hơn default (2.0) | Phase 2 |
| `focal_alpha` | 0.75 | Ưu tiên minority class | Phase 2 |
| `fcand_dim` | 256 | Unified embedding space | Phase 2, 3, 4 |
| `n_points (K)` | 4~8 | Số "xúc tu" deformable attention | Phase 4 |
| `n_heads` | 8 | MultiScaleDeformableAttention | Phase 4 |
| `temperature τ` | 0.07 | InfoNCE sharpness | Phase 5 |
| `n_hard_negatives` | 32 | Hard negative mining | Phase 5 |
| `learning_rate` | 1e-4 | AdamW | Phase 6 |
| `warmup_epochs` | 5 | Cosine Annealing warmup | Phase 6 |
| `total_epochs` | 100 | Đủ cho TTLD-Net hội tụ | Phase 6 |
| `batch_size` | 32 (OOM: 8~16) | Tùy VRAM | Phase 6 |
| `lambda1, lambda2` | 1.0 | Loss weights | Phase 6 |
| `grad_clip_norm` | 1.0 | Tránh gradient explosion | Phase 6 |

### Common Errors Quick Reference (từ SPEC Phụ lục A.1)

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| Recall < 90% sau Phase 2 | Hard NMS còn sót | Override bằng Soft-NMS |
| Channel mismatch error ở Phase 4 | Chưa project về 256-dim | Thêm FeatureProjection cho P4, P5, fcand |
| MMCV import error | CUDA version mismatch | Dùng đúng URL download mmcv |
| L_topology không giảm | Thiếu F.normalize trong InfoNCE | Đảm bảo normalize trước dot product |
| NaN loss | Gradient explosion | Thêm grad clipping max_norm=1.0 |
| OOM với batch_size=32 | VRAM không đủ | Giảm batch size hoặc giảm K từ 8→4 |
| L_det tăng sau epoch 20 | Learning rate quá cao | Kiểm tra LR scheduler |

---

*File này được tạo để bổ trợ cho `TTLD_Net_PROJECT_SPEC.md`. Luôn đọc SPEC trước khi implement, và chạy test sau mỗi task. Đừng skip test.*
