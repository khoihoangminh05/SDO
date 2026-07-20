# TTLD-Net — Topology-Aware Tiny Traffic Light Detection Network
> **Tài liệu Đặc tả Kỹ thuật dành cho AI Coding Agent**
> Phiên bản: 1.0 | Tác giả: Huy, Khôi | Tháng 6, 2026

---

## MỤC LỤC

1. [Bối cảnh & Động lực Nghiên cứu](#1-bối-cảnh--động-lực-nghiên-cứu)
2. [Tổng quan Kiến trúc Hệ thống](#2-tổng-quan-kiến-trúc-hệ-thống)
3. [Module 1 — High-Recall Candidate Generator](#3-module-1--high-recall-candidate-generator)
4. [Module 2 — Semantic Context Branch (P4, P5)](#4-module-2--semantic-context-branch-p4-p5)
5. [Module 3 — Implicit Topology Sampler](#5-module-3--implicit-topology-sampler)
6. [Module 4 — Verification & Contrastive Learning](#6-module-4--verification--contrastive-learning)
7. [Hàm Loss Tổng thể](#7-hàm-loss-tổng-thể)
8. [Dữ liệu — Bosch Small Traffic Lights Dataset](#8-dữ-liệu--bosch-small-traffic-lights-dataset)
9. [Sprint Roadmap — Kế hoạch Phát triển](#9-sprint-roadmap--kế-hoạch-phát-triển)
10. [Cấu trúc Thư mục Dự án](#10-cấu-trúc-thư-mục-dự-án)
11. [Stack Công nghệ & Dependencies](#11-stack-công-nghệ--dependencies)
12. [Chỉ số Đánh giá & Kết quả Kỳ vọng](#12-chỉ-số-đánh-giá--kết-quả-kỳ-vọng)
13. [Ablation Study — 5 Biến thể Thực nghiệm](#13-ablation-study--5-biến-thể-thực-nghiệm)
14. [Cross-Dataset Generalization & Robustness Evaluation](#14-cross-dataset-generalization--robustness-evaluation)

---

## 1. Bối cảnh & Động lực Nghiên cứu

### 1.1. Vấn đề cần giải quyết

TTLD-Net (Topology-Aware Tiny Traffic Light Detection Network) là một **framework phát hiện đối tượng hai giai đoạn** được đề xuất để giải quyết bài toán **phát hiện đèn giao thông siêu nhỏ** trong hệ thống lái xe tự hành (autonomous driving).

Các detector thông thường (YOLOv8, Faster R-CNN, ...) thất bại trong bài toán này vì chúng **chỉ dựa vào đặc trưng ngoại hình cục bộ** (local appearance), dễ bị lừa bởi các nguồn nhiễu có hình dạng tương tự. TTLD-Net khắc phục bằng cách học thêm **ngữ cảnh không gian ngầm** (implicit spatial topology) xung quanh mỗi ứng viên để xác minh tính hợp lệ.

### 1.2. Ba Thách thức Cốt lõi

| Thách thức | Mô tả chi tiết | Hệ quả nếu không giải quyết |
|---|---|---|
| **Feature Degradation** | Đối tượng siêu nhỏ (< 10px, ~8×8px) mất thông tin phân biệt sau nhiều lần down-sampling trong mạng sâu | Detector bỏ sót đèn nhỏ ở khoảng cách xa |
| **Appearance Ambiguity** | Đèn giao thông nhỏ dễ nhầm lẫn với đèn phanh xe tải, biển quảng cáo LED, biển báo phản quang | False Positive Rate cao, hệ thống AV không tin cậy được |
| **Domain Shift** | Hiệu suất suy giảm mạnh khi chuyển điều kiện ngày/đêm hoặc loại camera khác | Mô hình không generalize tốt ra ngoài tập huấn luyện |

### 1.3. Đóng góp Khoa học

- **Implicit Topology Sampler**: Lần đầu tiên áp dụng Deformable Attention để học bối cảnh không gian ngầm định cho tiny object detection, không cần nhãn tường minh về cột đèn hay cấu trúc đường phố.
- **Topology Contrastive Objective (InfoNCE)**: Tạo không gian tiềm ẩn phân tách rõ ràng `đèn thật` vs `nhiễu` thông qua contrastive learning.
- **High-Recall Two-Stage Pipeline**: Phân tách rõ vai trò giai đoạn 1 (không bỏ sót) và giai đoạn 2 (không báo nhầm).

---

## 2. Tổng quan Kiến trúc Hệ thống

### 2.1. Triết lý Thiết kế

> **"Thà bắt nhầm còn hơn bỏ sót"** ở Giai đoạn 1, sau đó **lọc nghiêm ngặt** ở Giai đoạn 2.

Kiến trúc TTLD-Net gồm **hai nhánh song song** hội tụ tại Implicit Topology Sampler:

- **Nhánh Cục bộ (Local Branch)**: Sinh ứng viên với độ phủ cao (High-Recall).
- **Nhánh Ngữ cảnh (Context Branch)**: Cung cấp thông tin bối cảnh cấu trúc để xác minh.

### 2.2. Luồng Dữ liệu Tổng thể

```
INPUT IMAGE (1280 × 720)
        │
        ▼
┌─────────────────────────────────────┐
│   YOLO Backbone + PANet             │
│   ─────────────────────────────     │
│   P1 (stride 2)  ─┐                 │
│   P2 (stride 4)  ─┤  Shallow feats  │
│   P3 (stride 8)  ─┘                 │
│   P4 (stride 16) ─┐  Deep feats     │
│   P5 (stride 32) ─┘                 │
└─────────────────────────────────────┘
          │                    │
          ▼                    ▼
┌──────────────────┐  ┌──────────────────┐
│  LOCAL BRANCH    │  │  CONTEXT BRANCH  │
│  P1, P2, P3      │  │  P4, P5          │
│                  │  │                  │
│  High-Recall     │  │  Semantic Context│
│  Candidate Gen.  │  │  Features        │
│                  │  │                  │
│  → C = {c1..cN} │  │  → Fctx          │
│  → fcand         │  │  (256-dim)       │
└──────────────────┘  └──────────────────┘
          │                    │
          └─────────┬──────────┘
                    ▼
      ┌─────────────────────────────┐
      │   IMPLICIT TOPOLOGY SAMPLER │
      │   (Deformable Attention)    │
      │   → Topology Embedding (zi) │
      └─────────────────────────────┘
                    │
                    ▼
      ┌─────────────────────────────┐
      │   VERIFICATION MODULE       │
      │   (MLP + InfoNCE Loss)      │
      │   → P(Valid) per candidate  │
      │   → Final Detection Set     │
      └─────────────────────────────┘
```

### 2.3. Tensor Dimensions — Tham chiếu nhanh

| Tensor | Shape | Mô tả |
|---|---|---|
| Input image | `(B, 3, 720, 1280)` | Batch ảnh đầu vào |
| P1 | `(B, C1, 360, 640)` | Stride 2, resolution cao nhất |
| P2 | `(B, C2, 180, 320)` | Stride 4 |
| P3 | `(B, C3, 90, 160)` | Stride 8 |
| P4 | `(B, C4, 45, 80)` | Stride 16 |
| P5 | `(B, C5, 23, 40)` | Stride 32, ngữ nghĩa nhất |
| fcand | `(B, N, 256)` | N ứng viên, mỗi ứng viên 256-dim |
| Fctx | `(B, 256, H4, W4)` + `(B, 256, H5, W5)` | Context features đã project |
| zi | `(B, N, 256)` | Topology Embedding mỗi ứng viên |
| vi | `(B, N, 512)` | Concat(fcand, zi) |
| P(Valid) | `(B, N, 1)` | Xác suất hợp lệ của ứng viên |

---

## 3. Module 1 — High-Recall Candidate Generator

**File**: `models/heads/tiny_generator.py`

### 3.1. Mục tiêu & Nguyên lý

Mục tiêu duy nhất của module này là **đảm bảo Recall > 95%** — không được bỏ sót bất kỳ đèn giao thông thật nào. Việc lọc False Positives được **cố tình trì hoãn** đến Module 3 và Module 4.

Module này can thiệp vào **forward pass của YOLO backbone** để trích xuất các feature map nông (shallow) thay vì chỉ dùng feature map sâu.

### 3.2. Input / Output

```python
# Input
P1: torch.Tensor  # (B, C1, H/2, W/2)   - stride 2
P2: torch.Tensor  # (B, C2, H/4, W/4)   - stride 4
P3: torch.Tensor  # (B, C3, H/8, W/8)   - stride 8

# Output
candidates: List[Dict]  # C = {c1, c2, ..., cN}
                        # mỗi ci chứa: bbox (x,y,w,h), confidence, class_id
fcand: torch.Tensor     # (B, N, 256) - local feature per candidate
```

### 3.3. Ba Kỹ thuật Quan trọng

#### (A) Hạ thấp Confidence Threshold = 0.05

```python
# Trong config file hoặc inference script
conf_threshold = 0.05   # Thay vì mặc định 0.25 ~ 0.5

# Lý do:
# - Giữ lại "hard examples" — đèn thật nhưng độ tự tin thấp
# - Bao gồm chủ động các False Positives để lọc sau
# - Đèn siêu nhỏ (8x8px) thường cho confidence thấp
```

#### (B) Soft-NMS thay thế NMS cứng

```python
# Vấn đề NMS cứng:
# - Xóa hoàn toàn box có IoU > threshold với box tốt nhất
# - Xóa nhầm đèn đứng sát nhau trên cùng 1 cột

# Soft-NMS: giảm score dần dần thay vì xóa
def soft_nms(boxes, scores, sigma=0.5, score_thresh=0.001):
    """
    Gaussian Soft-NMS
    score_new = score * exp(-IoU^2 / sigma)
    """
    # Override hàm NMS mặc định trong YOLO inference pipeline
    ...

# Cài đặt: override trong yolo26.py hoặc inference wrapper
```

#### (C) Tinh chỉnh Focal Loss — γ = 1.5, α = 0.75

```python
# losses/focal_loss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.75):
        """
        gamma = 1.5  (mặc định 2.0 — giảm xuống để phạt FN nặng hơn)
        alpha = 0.75 (tăng trọng số lớp đèn — minority class trong Bosch)

        Bosch Dataset rất mất cân bằng:
        - Số lượng background patches >> số lượng đèn giao thông
        - alpha cao giúp mạng ưu tiên bắt đúng đèn hơn là tránh báo nhầm
        """
        self.gamma = gamma
        self.alpha = alpha
```

### 3.4. Cách Can thiệp vào YOLO Forward Pass

```python
# models/backbones/yolo26.py
class YOLO26Backbone(nn.Module):
    def forward(self, x):
        # Chạy qua các lớp backbone
        p1 = self.layer1(x)      # stride 2
        p2 = self.layer2(p1)     # stride 4
        p3 = self.layer3(p2)     # stride 8
        p4 = self.layer4(p3)     # stride 16
        p5 = self.layer5(p4)     # stride 32

        # QUAN TRỌNG: trả về TẤT CẢ feature maps
        # Không chỉ (P3, P4, P5) như YOLO thông thường
        return p1, p2, p3, p4, p5
```

### 3.5. What NOT to do ở Module này

- ❌ KHÔNG áp dụng NMS cứng (hard NMS)
- ❌ KHÔNG lọc False Positives
- ❌ KHÔNG dùng confidence threshold cao (> 0.1)
- ❌ KHÔNG xóa các đèn đứng cạnh nhau (Soft-NMS giữ lại)

---

## 4. Module 2 — Semantic Context Branch (P4, P5)

**File**: `models/necks/fpn_panet.py` (trích xuất) + phần trong `models/heads/implicit_topo.py`

### 4.1. Mục tiêu & Nguyên lý

Cung cấp **"tầm nhìn rộng"** — bối cảnh không gian cấp cao để xác minh từng ứng viên. Trong khi Module 1 nhìn "cận cảnh" vào từng đối tượng nhỏ, Module 2 nhìn "toàn cảnh" để hiểu môi trường xung quanh (cột đèn, đường phố, bầu trời, ngã tư).

### 4.2. Đặc điểm P4 và P5

| Feature Map | Stride | Kích thước (với input 720×1280) | Nội dung ngữ nghĩa |
|---|---|---|---|
| P4 | 16 | 45 × 80 | Cấu trúc vùng: cột đèn, vùng trời, mặt đường |
| P5 | 32 | 23 × 40 | Toàn cảnh: layout ngã tư, geometry đường phố |

### 4.3. Feature Subspace Projection

```python
# Vấn đề: P4 và P5 có số kênh khác nhau (C4 ≠ C5 ≠ C_cand)
# Giải pháp: chiếu về không gian 256 chiều chung

class FeatureProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # Conv 1×1: không thay đổi spatial resolution, chỉ đổi số kênh

    def forward(self, x):
        return self.proj(x)

# Sử dụng:
proj_p4 = FeatureProjection(in_channels=C4, out_channels=256)
proj_p5 = FeatureProjection(in_channels=C5, out_channels=256)
proj_cand = FeatureProjection(in_channels=C_cand, out_channels=256)

Fctx_p4 = proj_p4(P4)   # (B, 256, 45, 80)
Fctx_p5 = proj_p5(P5)   # (B, 256, 23, 40)
fcand_proj = proj_cand(fcand)  # (B, N, 256)
```

---

## 5. Module 3 — Implicit Topology Sampler

**File**: `models/heads/implicit_topo.py`

### 5.1. Mục tiêu & Nguyên lý — "Vươn Xúc Tu"

Đây là **trái tim kỹ thuật** của TTLD-Net. Thay vì cắt vùng ảnh tĩnh xung quanh ứng viên (static grid sampling), module này **tự học vươn đến các vùng ảnh có giá trị xác thực cao nhất** thông qua Deformable Attention.

Ví dụ: Khi đánh giá ứng viên "đèn tại vị trí (x, y)", hệ thống không chỉ nhìn xung quanh (x, y), mà có thể học rằng "nếu phía dưới ứng viên này có cột đèn thẳng, thì đây nhiều khả năng là đèn giao thông thật". Các "xúc tu" sẽ vươn đến cột đèn bên dưới.

### 5.2. Input / Output

```python
# Input
fcand:    torch.Tensor  # (B, N, 256)       - local candidate features
Fctx_p4:  torch.Tensor  # (B, 256, 45, 80) - projected P4
Fctx_p5:  torch.Tensor  # (B, 256, 23, 40) - projected P5
pq:       torch.Tensor  # (B, N, 2)         - tọa độ trung tâm ứng viên (normalized 0~1)

# Output
zi:       torch.Tensor  # (B, N, 256)       - Topology Embedding mỗi ứng viên
```

### 5.3. Công thức Toán học

$$z_i = \sum_{k=1}^{K} A_{qk} \cdot W \cdot f_{ctx}(p_q + \Delta p_{qk})$$

| Ký hiệu | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| `pq` | `(B, N, 2)` | Tọa độ trung tâm ứng viên — điểm tham chiếu |
| `Δpqk` | `(B, N, K, 2)` | K offset lấy mẫu — được **mạng dự đoán động** |
| `Aqk` | `(B, N, K)` | Trọng số Attention cho mỗi điểm lấy mẫu |
| `W` | `(256, 256)` | Ma trận chiếu tuyến tính học được |
| `fctx(·)` | - | Bilinear interpolation tại tọa độ lẻ |
| `K` | `int` | Siêu tham số, K = 4 ~ 8 điểm |

### 5.4. Implementation Chi tiết

```python
# models/heads/implicit_topo.py
from mmcv.ops import MultiScaleDeformableAttention  # CUDA Kernel tối ưu

class ImplicitTopologySampler(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_points = n_points  # K = số điểm lấy mẫu

        # Dự đoán offsets Δpqk: (B, N, 256) → (B, N, K*2)
        self.offset_predictor = nn.Linear(d_model, n_points * 2)

        # Dự đoán attention weights Aqk: (B, N, 256) → (B, N, K)
        self.attention_weights = nn.Linear(d_model, n_points)

        # Ma trận chiếu W
        self.proj = nn.Linear(d_model, d_model)

        # Multi-scale deformable attention từ MMCV
        self.deform_attn = MultiScaleDeformableAttention(
            embed_dims=d_model,
            num_heads=n_heads,
            num_points=n_points,
        )

    def forward(self, fcand, Fctx_list, pq):
        """
        Args:
            fcand:      (B, N, 256)
            Fctx_list:  [(B, 256, H4, W4), (B, 256, H5, W5)]
            pq:         (B, N, 2) - normalized coords in [0, 1]

        Returns:
            zi:         (B, N, 256)
        """
        # Bước 1: Dự đoán offsets và attention weights từ fcand
        offsets = self.offset_predictor(fcand)   # (B, N, K*2)
        attn_w  = self.attention_weights(fcand)  # (B, N, K)
        attn_w  = attn_w.softmax(dim=-1)         # normalize

        # Bước 2: Reshape offsets → (B, N, K, 2)
        offsets = offsets.view(*offsets.shape[:2], self.n_points, 2)

        # Bước 3: Bilinear sampling tại (pq + Δpqk) trên Fctx
        # fctx(pq + Δpqk): lấy đặc trưng tại vị trí fractional
        # Gradient chạy qua bilinear interpolation → backprop học được offsets

        # Bước 4: Weighted sum → Topology Embedding
        zi = self.deform_attn(
            query=fcand,
            reference_points=pq.unsqueeze(2),  # (B, N, 1, 2)
            input_flatten=...,  # flatten Fctx
            ...
        )
        return zi  # (B, N, 256)
```

### 5.5. Điều kiện Tiên quyết

```bash
# Cài MMCV với CUDA support (BẮT BUỘC)
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cuda_version}/{torch_version}/index.html

# Kiểm tra import
python -c "from mmcv.ops import MultiScaleDeformableAttention; print('OK')"
```

### 5.6. Test Forward Pass

```python
# Test độc lập — PHẢI PASS trước khi tích hợp
def test_topology_sampler():
    B, N, K = 2, 100, 4
    sampler = ImplicitTopologySampler(d_model=256, n_points=K)

    fcand = torch.randn(B, N, 256)
    Fctx_p4 = torch.randn(B, 256, 45, 80)
    Fctx_p5 = torch.randn(B, 256, 23, 40)
    pq = torch.rand(B, N, 2)  # normalized coordinates

    zi = sampler(fcand, [Fctx_p4, Fctx_p5], pq)
    assert zi.shape == (B, N, 256), f"Expected (B,N,256), got {zi.shape}"
    assert not torch.isnan(zi).any(), "NaN in topology embedding"
    print(f"✅ ImplicitTopologySampler: zi.shape = {zi.shape}")
```

---

## 6. Module 4 — Verification & Contrastive Learning

**File**: `models/heads/verification.py` + `losses/infonce_loss.py`

### 6.1. Mục tiêu

Đây là **màng lọc cuối cùng** — phân loại mỗi ứng viên thành `Hợp lệ` hoặc `Loại bỏ` dựa trên **cả hai**: đặc trưng ngoại hình (`fcand`) và bối cảnh topology (`zi`).

### 6.2. Luồng Dữ liệu

```python
# Input
fcand:  (B, N, 256)   # Local appearance features (từ Module 1)
zi:     (B, N, 256)   # Topology Embedding (từ Module 3)

# Bước 1: Concatenation
vi = torch.cat([fcand, zi], dim=-1)   # (B, N, 512)

# Bước 2: Verification MLP
# vi (512) → Linear → ReLU → Linear → ReLU → Linear → Sigmoid → P(Valid)

# Output
p_valid:  (B, N, 1)   # P(Valid) ∈ [0, 1] cho mỗi ứng viên
```

### 6.3. Verification MLP

```python
# models/heads/verification.py
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

# Loss: Binary Cross-Entropy
L_verify = F.binary_cross_entropy(p_valid, labels)
```

### 6.4. InfoNCE Contrastive Loss — Topology Loss

**Mục tiêu**: Ép không gian tiềm ẩn phân tách rõ ràng `đèn thật` vs `đèn giả` ở mức topology embedding.

```python
# losses/infonce_loss.py
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature  # τ = 0.07

    def forward(self, zi, z_pos, z_neg_list):
        """
        Args:
            zi:          (B, N, 256) - topology embedding của ứng viên
            z_pos:       (B, N, 256) - context embedding của đèn THẬT tương ứng
            z_neg_list:  (B, N, M, 256) - M negative contexts (nhiễu, ảnh khác)

        Công thức:
            sim(u, v) = dot(u, v) / (||u|| * ||v||)   # cosine similarity
            L = -log[ exp(sim(zi, z+)/τ) / Σ_j exp(sim(zi, z-_j)/τ) ]
        """
        # Normalize
        zi_norm  = F.normalize(zi, dim=-1)
        zp_norm  = F.normalize(z_pos, dim=-1)
        zn_norm  = F.normalize(z_neg_list, dim=-1)

        # Positive similarity
        pos_sim = (zi_norm * zp_norm).sum(-1) / self.tau   # (B, N)

        # Negative similarities
        neg_sim = torch.einsum('bnd,bnmd->bnm', zi_norm, zn_norm) / self.tau  # (B, N, M)

        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (B, N, 1+M)
        labels = torch.zeros(logits.shape[:2], dtype=torch.long)       # positive = index 0
        loss = F.cross_entropy(logits.view(-1, 1+neg_sim.shape[-1]),
                               labels.view(-1))
        return loss
```

### 6.5. Hard Negative Mining

```python
# Vấn đề đặc thù của Bosch Dataset:
# - Rất nhiều đèn phanh xe tải có độ tự tin cao
# - Nếu dùng random negatives, Loss hội tụ chậm và ranh giới mờ

def mine_hard_negatives(candidates, n_hard=32):
    """
    Lọc các ứng viên nhiễu có confidence CAO NHẤT làm Hard Negatives.
    Các ứng viên này "trông giống đèn thật nhất" nhưng thực ra là nhiễu.

    Args:
        candidates: danh sách ứng viên với confidence score
        n_hard: số lượng hard negatives cần lấy

    Returns:
        hard_negatives: top-n_hard candidates sorted by confidence (descending)
                        trong số các candidates là False Positives
    """
    fp_candidates = [c for c in candidates if c['is_false_positive']]
    fp_candidates.sort(key=lambda c: c['confidence'], reverse=True)
    return fp_candidates[:n_hard]
```

---

## 7. Hàm Loss Tổng thể

$$L_{total} = L_{det} + \lambda_1 \cdot L_{topology} + \lambda_2 \cdot L_{verify}$$

| Loss component | File | Mô tả | Giá trị λ khuyến nghị |
|---|---|---|---|
| `L_det` | (YOLO built-in) | YOLO detection loss: bbox regression + state classification (Red/Yellow/Green/Off) | 1.0 (base) |
| `L_topology` | `losses/infonce_loss.py` | InfoNCE Contrastive Loss — học topology ngầm | λ1 = 0.5 ~ 1.0 (tune) |
| `L_verify` | `models/heads/verification.py` | Binary Cross-Entropy — xác minh hợp lệ/không | λ2 = 0.5 ~ 1.0 (tune) |

```python
# train.py — loss combination
loss_total = loss_det + lambda1 * loss_topology + lambda2 * loss_verify
loss_total.backward()
```

> **Lưu ý quan trọng**: Cả 3 loss phải giảm đồng thời. Nếu `L_topology` không giảm, kiểm tra lại InfoNCE implementation và Hard Negative Mining. Theo dõi riêng từng loss trên TensorBoard.

---

## 8. Dữ liệu — Bosch Small Traffic Lights Dataset

### 8.1. Tổng quan Dataset

- **Tên**: Bosch Small Traffic Lights Dataset
- **Format**: YAML label files + PNG stereo images
- **Độ phân giải ảnh**: 1280 × 720 pixels
- **Đặc điểm nổi bật**: Chứa rất nhiều đèn siêu nhỏ (< 10 pixel), mất cân bằng lớp nghiêm trọng

### 8.2. Mapping Classes

```python
# Bosch dataset có nhiều class gốc (hướng, vị trí, ...)
# TTLD-Net rút gọn về 4 class:

CLASS_MAPPING = {
    # Bosch original → TTLD class
    'Green': 0,
    'GreenLeft': 0,
    'GreenRight': 0,
    'GreenStraight': 0,
    'Yellow': 1,
    'Red': 2,
    'RedLeft': 2,
    'RedRight': 2,
    'RedStraight': 2,
    'off': 3,
}

CLASS_NAMES = ['Green', 'Yellow', 'Red', 'Off']  # 4 classes
```

### 8.3. BoschDataset Class

```python
# data/dataset.py
import yaml
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class BoschDataset(Dataset):
    """
    Dataset class cho Bosch Small Traffic Lights.

    Cấu trúc YAML label:
    - path/to/image.png:
        boxes:
          - {x_center, y_center, w, h, label, occluded}
          - ...
    """

    def __init__(self, yaml_path: str, transform=None, split='train'):
        self.samples = self._load_yaml(yaml_path)
        self.transform = transform
        self.class_mapping = CLASS_MAPPING

    def _load_yaml(self, yaml_path):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return data  # list of {image_path, boxes}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        boxes = self._parse_boxes(sample['boxes'])  # → Tensor (N, 5): x,y,w,h,class

        if self.transform:
            image, boxes = self.transform(image, boxes)

        return image, boxes

    def _parse_boxes(self, raw_boxes):
        boxes = []
        for box in raw_boxes:
            cls = self.class_mapping.get(box['label'], 3)  # default: Off
            boxes.append([box['x_center'], box['y_center'],
                          box['w'], box['h'], cls])
        return torch.tensor(boxes, dtype=torch.float32)


# DataLoader config
def create_dataloader(yaml_path, batch_size=16, num_workers=4):
    dataset = BoschDataset(yaml_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,           # Tăng tốc transfer CPU→GPU
        collate_fn=custom_collate  # Xử lý batch có số boxes khác nhau
    )

def custom_collate(batch):
    """
    Xử lý batch với số lượng boxes không đồng đều giữa các ảnh.
    Padding boxes về chiều dài tối đa trong batch.
    """
    images, targets = zip(*batch)
    images = torch.stack(images)
    # targets: list of tensors với shape (Ni, 5)
    return images, targets
```

### 8.4. EDA Script — Phân tích Dataset

```python
# scripts/eda_bosch.py
# Mục đích: Hiểu phân phối kích thước bounding box để thiết kế anchor sizes

def analyze_bbox_distribution(yaml_path):
    """
    Output cần có:
    - Histogram kích thước bbox (w, h)
    - Tỉ lệ % các đèn có w < 10px, h < 10px
    - Phân phối classes (Red/Yellow/Green/Off)
    - Số lượng đèn per image (để estimate N cho batch)
    """
    ...
```

---

## 9. Sprint Roadmap — Kế hoạch Phát triển

### Sprint 1 — Xử lý Dữ liệu & Baseline

**Goal**: Chinh phục định dạng YAML của Bosch và thiết lập điểm mốc baseline đầu tiên.

**Tasks**:
1. Viết `scripts/eda_bosch.py` — đọc `train.yaml`, thống kê phân phối kích thước bbox
2. Implement `data/dataset.py` — class `BoschDataset` kế thừa `torch.utils.data.Dataset`
3. Implement `data/transforms.py` — augmentation: random flip, color jitter, mosaic
4. Cấu hình DataLoader: `num_workers=4`, `pin_memory=True`, `collate_fn` tùy chỉnh
5. Huấn luyện YOLO baseline (không có TTLD modifications), đo APsmall và Recall

**Deliverable**:
- [ ] DataLoader chạy mượt không lỗi
- [ ] File `logs/baseline_yolo_metrics.json` với mAP, APsmall, Recall của YOLO gốc

---

### Sprint 2 — High-Recall Generator

**Goal**: Ép mạng sinh ra hàng nghìn bounding box, đảm bảo Recall > 95%.

**Tasks**:
1. Modify `models/backbones/yolo26.py` — can thiệp forward pass, trích xuất P1, P2, P3
2. Implement `models/heads/tiny_generator.py` — head nhận (P1, P2, P3), output candidates + fcand
3. Override NMS trong inference pipeline → Soft-NMS (`sigma=0.5`)
4. Sửa config: `conf_threshold = 0.05`
5. Implement `losses/focal_loss.py` với `gamma=1.5`, `alpha=0.75`

**Deliverable**:
- [ ] Recall > 95% trên validation set
- [ ] Mô hình bắt được đèn siêu nhỏ (< 10px)
- [ ] Số lượng candidates N ~ vài nghìn trên mỗi ảnh (expected)

---

### Sprint 3 — Implicit Topology Sampler

**Goal**: Implement lõi lấy mẫu ngữ cảnh bằng CUDA từ MMCV.

**Tasks**:
1. Cài đặt MMCV với CUDA support
2. Implement `FeatureProjection` (Conv 1×1) chiếu P4, P5, fcand → 256-dim
3. Implement `ImplicitTopologySampler` trong `models/heads/implicit_topo.py`
4. Thiết kế offset predictor và attention weight predictor (Linear layers)
5. Tích hợp `MultiScaleDeformableAttention` từ MMCV
6. Viết unit test: forward pass với dummy tensor, kiểm tra shape + no NaN

**Deliverable**:
- [ ] `ImplicitTopologySampler` forward pass không lỗi
- [ ] Output zi shape: `(B, N, 256)`, giá trị hợp lệ (không NaN, không overflow)
- [ ] Memory usage trong giới hạn chấp nhận được

---

### Sprint 4 — Verification & InfoNCE

**Goal**: Viết mạng phân loại và hàm Loss đối chiếu.

**Tasks**:
1. Implement `VerificationMLP` trong `models/heads/verification.py`
2. Implement `InfoNCELoss` trong `losses/infonce_loss.py` (τ = 0.07)
3. Implement `HardNegativeMiner` — lọc top-k FP confidence cao nhất
4. Viết unit test cho InfoNCE: kiểm tra loss giảm với positive pairs gần nhau

**Deliverable**:
- [ ] Loss hội tụ trên toy dataset
- [ ] Đồ thị InfoNCE Loss giảm ổn định
- [ ] Ranh giới phân tách đèn thật vs nhiễu rõ ràng trong không gian embedding

---

### Sprint 5 — End-to-End Training

**Goal**: Ghép toàn bộ kiến trúc và huấn luyện trên cụm GPU.

**Tasks**:
1. Viết `train.py` — kết nối Dataset → Backbone → Generator → Topology Sampler → Verification
2. Implement loss combination: `L_total = L_det + λ1*L_topo + λ2*L_verify`
3. Cấu hình optimizer: AdamW (lr=1e-4) + Cosine Annealing LR Scheduler
4. Setup Distributed Data Parallel (DDP) nếu multi-GPU
5. Tích hợp TensorBoard logging: loss curves, learning rate, metrics per epoch

**Cấu hình Training**:
```yaml
# configs/m4_full_ttld.yaml
model:
  backbone: yolo26
  neck: fpn_panet
  heads:
    - tiny_generator
    - implicit_topo_sampler
    - verification_mlp

training:
  epochs: 100
  batch_size: 32        # tăng lên 64 nếu đủ VRAM
  optimizer: AdamW
  lr: 1.0e-4
  scheduler: CosineAnnealing
  warmup_epochs: 5

loss:
  lambda1: 1.0          # weight cho L_topology
  lambda2: 1.0          # weight cho L_verify

data:
  conf_threshold: 0.05  # Stage 1 threshold
  soft_nms_sigma: 0.5
  focal_gamma: 1.5
  focal_alpha: 0.75
  num_workers: 4
  pin_memory: true
```

**Deliverable**:
- [ ] Mô hình TTLD-Net đầy đủ chạy end-to-end không lỗi
- [ ] Weights tốt nhất được lưu vào `checkpoints/best_model.pth`
- [ ] TensorBoard logs: cả 3 loss cùng giảm ổn định

---

### Sprint 6 — Ablation & Visualization

**Goal**: Chạy thực nghiệm cắt bỏ và trực quan hóa để viết paper.

**Tasks**:
1. Implement 5 config Ablation (xem Mục 13)
2. Script `scripts/run_ablation.py` — chạy tự động M0 → M4
3. Viết `utils/visualize.py` — vẽ bounding box, attention heatmap ("xúc tu") lên ảnh Bosch
4. Tổng hợp bảng so sánh APsmall, Precision, Recall, FPR
5. Dọn dẹp codebase, viết `README.md`

**Deliverable**:
- [ ] Bảng so sánh 5 biến thể M0–M4
- [ ] Hình ảnh heatmap xúc tu Deformable Attention
- [ ] README.md đầy đủ hướng dẫn setup/train/test

---

## 10. Cấu trúc Thư mục Dự án

```
ttld-net/
│
├── data/
│   ├── dataset.py          # BoschDataset class, DataLoader factory
│   └── transforms.py       # Augmentation pipeline (flip, jitter, mosaic)
│
├── models/
│   ├── backbones/
│   │   └── yolo26.py       # YOLO backbone, trích xuất P1–P5
│   ├── necks/
│   │   └── fpn_panet.py    # Feature Pyramid Network + PANet
│   └── heads/
│       ├── tiny_generator.py   # High-Recall Candidate Generator (Stage 1)
│       ├── implicit_topo.py    # Implicit Topology Sampler (Deformable Attn)
│       └── verification.py     # Verification MLP (Stage 2)
│
├── losses/
│   ├── focal_loss.py       # Focal Loss (γ=1.5, α=0.75)
│   └── infonce_loss.py     # InfoNCE Contrastive Loss (τ=0.07)
│
├── configs/
│   ├── m0_baseline.yaml    # Ablation: YOLO only (no TTLD)
│   ├── m1_shallow.yaml     # Ablation: + P1/P2/P3 shallow features
│   ├── m2_softNMS.yaml     # Ablation: + Soft-NMS
│   ├── m3_topology.yaml    # Ablation: + Implicit Topology Sampler
│   └── m4_full_ttld.yaml   # Full TTLD-Net (all components)
│
├── scripts/
│   ├── eda_bosch.py        # Exploratory data analysis
│   ├── run_ablation.py     # Chạy tự động 5 ablation configs
│   └── draw_attention.py   # Visualize deformable attention offsets
│
├── utils/
│   └── visualize.py        # Vẽ bbox, heatmap, attention points lên ảnh
│
├── logs/                   # TensorBoard logs
├── checkpoints/            # Model weights (.pth)
│
├── train.py                # Main training script
├── test.py                 # Evaluation script: tính mAP, APsmall, FPR
├── requirements.txt        # Python dependencies
└── README.md               # Setup, train, test instructions
```

---

## 11. Stack Công nghệ & Dependencies

```txt
# requirements.txt

# Core Deep Learning
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0

# YOLO
ultralytics>=8.0.0          # YOLOv8 base

# MMCV (BẮT BUỘC cho Deformable Attention CUDA Kernel)
mmcv-full>=2.0.0            # Phải cài đúng version CUDA của máy

# Data & Visualization
pyyaml>=6.0
pillow>=10.0.0
opencv-python>=4.8.0
matplotlib>=3.7.0

# Training utilities
tensorboard>=2.13.0
tqdm>=4.65.0

# Distributed training (optional, cho multi-GPU)
# torch.distributed (built-in)
```

### Cài đặt MMCV — Lưu ý quan trọng

```bash
# Kiểm tra CUDA version
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Cài mmcv-full đúng version (ví dụ: CUDA 11.8, PyTorch 2.0)
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html

# Xác nhận MultiScaleDeformableAttention hoạt động
python -c "from mmcv.ops import MultiScaleDeformableAttention; print('✅ MMCV OK')"
```

---

## 12. Chỉ số Đánh giá & Kết quả Kỳ vọng

### 12.1. Metrics được Dùng

| Metric | Mô tả | Relevance |
|---|---|---|
| **AP50** | Average Precision tại IoU=0.5 | Độ chính xác tổng thể |
| **APsmall** | AP cho objects < 32×32px (COCO definition) | Đánh giá đèn siêu nhỏ |
| **Recall** | Tỉ lệ đèn thật được phát hiện | Quan trọng nhất cho safety |
| **Precision** | Tỉ lệ cảnh báo chính xác | Tránh false alarms |
| **FPR** | False Positive Rate | Đo lường nhiễu |

> **Bắt buộc báo cáo thêm size-stratified breakdown** cho AP50/APsmall/Recall/FPR theo 3 bin kích thước bbox: `<8px`, `8–16px`, `16–32px`. Chỉ báo APsmall gộp (COCO <32px) là không đủ để chứng minh cải thiện thực sự ở nhóm "siêu nhỏ" (~8×8px) — mục tiêu chính của dự án. Xem `scripts/eda_bosch.py` (T1.1) để lấy phân phối kích thước làm cơ sở chia bin.

### 12.2. Kết quả Kỳ vọng so với YOLO Baseline

| Metric | Cải thiện Kỳ vọng | Ý nghĩa Thực tế |
|---|---|---|
| AP50 | +1.5% ~ +3.0% | Độ chính xác tổng thể tốt hơn |
| APsmall | **+3.0% ~ +6.0%** | Phát hiện đèn siêu nhỏ hiệu quả hơn rõ rệt |
| Recall | +2.0% ~ +5.0% | Ít bỏ sót đèn thật hơn |
| Precision | +5.0% ~ +8.0% | Cảnh báo chính xác hơn |
| False Positive Rate | **−15.0% ~ −30.0%** | Giảm đáng kể báo nhầm |

### 12.3. Script Đánh giá

```python
# test.py
def evaluate(model, dataloader, conf_threshold=0.5):
    """
    Tính toán tất cả metrics sau khi training.
    conf_threshold ở đây là STRICT (khác với 0.05 trong Stage 1).
    """
    metrics = {
        'ap50': compute_ap50(preds, targets),
        'apsmall': compute_apsmall(preds, targets),
        'recall': compute_recall(preds, targets),
        'precision': compute_precision(preds, targets),
        'fpr': compute_false_positive_rate(preds, targets),
    }
    return metrics
```

---

## 13. Ablation Study — 5 Biến thể Thực nghiệm

Ablation Study chứng minh đóng góp của từng component trong TTLD-Net.

| ID | Tên | Mô tả | Components Active |
|---|---|---|---|
| **M0** | Baseline | YOLO thuần, không có modification | YOLO only |
| **M1** | +Shallow | Thêm P1/P2/P3, Soft-NMS, conf=0.05 | M0 + Stage 1 mods |
| **M2** | +FocalTune | Thêm Focal Loss γ=1.5, α=0.75 | M1 + Focal tuning |
| **M3** | +Topology | Thêm Implicit Topology Sampler | M2 + Deformable Attn |
| **M4** | **Full TTLD** | Tất cả components + InfoNCE + Verification | **Complete TTLD-Net** |

```bash
# scripts/run_ablation.py — Chạy tự động
for config in m0_baseline m1_shallow m2_focal m3_topology m4_full_ttld; do
    python train.py --config configs/${config}.yaml --output logs/ablation/${config}
    python test.py  --config configs/${config}.yaml --weights checkpoints/${config}_best.pth \
                    --output results/ablation/${config}_metrics.json
done

# Tổng hợp kết quả
python scripts/compare_ablation.py --results_dir results/ablation/
```

---

## 14. Cross-Dataset Generalization & Robustness Evaluation

### 14.1. Động lực

Mục 1.2 đã xác định **Domain Shift** là một trong ba thách thức cốt lõi: hiệu suất suy giảm khi đổi điều kiện ngày/đêm hoặc loại camera. Toàn bộ Phase 0–7 chỉ train và test trên **một dataset duy nhất (Bosch/BSTLD)** — điều này không chứng minh được model giải quyết được Domain Shift, chỉ chứng minh model học tốt trên phân phối Bosch. Mục 14 định nghĩa một pha đánh giá bổ sung, **chạy sau khi Phase 7 hoàn thành**, không thay đổi training pipeline, không train lại model.

**Nguyên tắc quan trọng**: các dataset trong mục này **chỉ dùng để test (zero-shot inference)**, tuyệt đối không dùng để train hoặc fine-tune (xem `Context.md` — Fixed Constraints).

### 14.2. Bộ Dataset Đánh giá Ngoài (External Test Sets)

| Dataset | Nguồn gốc | Đặc điểm | Ngách kiểm chứng |
|---|---|---|---|
| **DTLD** (DriveU Traffic Light Dataset) | 11 thành phố ở Đức, ảnh độ phân giải 2MP, có stereo + GPS/vehicle data | >230,000 đèn giao thông được gán nhãn, có pictogram và trạng thái đèn vàng-đỏ | Domain shift camera/độ phân giải/quốc gia |
| **S2TLD** (SJTU Small Traffic Light Dataset) | Shanghai Jiao Tong University + Anhui University, Trung Quốc | 5,786 ảnh (~1920×1080 và 1280×720), 14,130 instance, 5 class (red/yellow/green/off/wait-on), cảnh có đèn nhấp nháy, thay đổi ánh sáng mạnh, vật thể dễ nhầm với đèn (đèn hậu xe) | Domain shift lục địa khác + Appearance Ambiguity (Mục 1.2) |
| **LISA Traffic Light Dataset** | Mỹ | Benchmark phổ biến, thường dùng cùng Bosch+DTLD trong literature cùng niche | So sánh trực tiếp với các paper công bố khác |
| **Cityscapes TL++ (CSTL)** | Đức, dựa trên Cityscapes | Mật độ đèn/ảnh khác biệt rõ so với Bosch/DTLD, dùng trong benchmark tiny-traffic-light gần đây | So sánh SOTA đã công bố trên cùng 3 dataset (Bosch/DTLD/CSTL) |

### 14.3. Protocol Đánh giá

**Protocol A — Zero-shot Cross-Dataset Transfer**
```
Train: Bosch (train split, không đổi)
Test:  DTLD / S2TLD / LISA / CSTL (toàn bộ, KHÔNG fine-tune)
Đo:    AP50, APsmall, Recall, Precision, FPR cho M0 (baseline) và M4 (full TTLD-Net)
So sánh: gap = metric(Bosch test) - metric(external test) — gap nhỏ hơn ở M4 so với M0
         chứng minh Topology Sampler học context tổng quát, không overfit Bosch.
```

**Protocol B — Size-Stratified Breakdown**
```
Với mỗi dataset ở trên, chia kết quả theo 3 bin: <8px, 8-16px, 16-32px
(áp dụng luôn cho Bosch test set nội bộ — không chỉ external)
```

**Protocol C — Condition-Stratified (nếu dataset có metadata)**
```
DTLD và LISA có thể tách theo ngày/đêm — báo riêng APsmall/FPR cho từng điều kiện.
Nếu dataset không có nhãn điều kiện, dùng heuristic độ sáng trung bình ảnh (mean pixel intensity)
để phân nhóm ngày/đêm gần đúng.
```

**Protocol D — Synthetic Corruption Robustness** (không cần dataset mới)
```
Áp corruption lên chính Bosch test set gốc theo kiểu ImageNet-C:
- Gaussian noise (severity 1-3)
- Motion blur
- Gamma thấp (giả lập điều kiện đêm)
- Fog/haze synthetic
- JPEG compression
Đo APsmall degradation (%) theo từng loại corruption, từng severity level.
File: scripts/eval_robustness.py (xem 14.5)
```

**Protocol E — Latency / Throughput**
```
Đo trên chính GPU dùng để train (RTX 4090):
- Params (M), FLOPs (G) cho M0 và M4
- ms/frame (batch_size=1), FPS
Lý do: N candidates ở Stage 1 có thể lên tới vài nghìn/ảnh (xem Phase 2), cộng thêm
Deformable Attention ở Stage 2 → compute cost có thể tăng đáng kể so với YOLO baseline.
Phải báo cáo trade-off AP vs latency trung thực trong paper.
```

**Protocol F — Statistical Significance**
```
Chạy M4 (full model) với tối thiểu 3 random seed khác nhau.
Báo cáo mean ± std cho AP50/APsmall/FPR, không chỉ 1 số run tốt nhất.
```

### 14.4. Output Kỳ vọng

```
results/cross_dataset/
├── dtld_metrics.json
├── s2tld_metrics.json
├── lisa_metrics.json
├── cstl_metrics.json
├── size_stratified_breakdown.json     # Protocol B, mọi dataset
├── condition_stratified_breakdown.json # Protocol C
├── robustness_corruption.json          # Protocol D
├── latency_benchmark.json              # Protocol E
└── multi_seed_variance.json            # Protocol F
```

### 14.5. File Cần Tạo

```
scripts/prepare_external_datasets.py   # download + convert DTLD/S2TLD/LISA/CSTL sang format
                                        # tương thích BoschDataset (class remapping bắt buộc,
                                        # vì mỗi dataset có class set khác — xem 14.6)
scripts/eval_cross_dataset.py          # Protocol A, B, C
scripts/eval_robustness.py             # Protocol D
scripts/benchmark_latency.py           # Protocol E
scripts/eval_multi_seed.py             # Protocol F
```

### 14.6. Lưu ý Quan trọng — Class Mapping Không Đồng nhất

Mỗi dataset ngoài có class set khác Bosch (ví dụ DTLD có thêm pictogram + trạng thái đèn vàng-đỏ, S2TLD có thêm "wait-on"). **Không đoán mapping** — khi implement `prepare_external_datasets.py`, phải map về đúng 4 class gốc của TTLD-Net (`Green, Yellow, Red, Off` — xem `CLASS_MAPPING` ở Mục 8), loại các class không tương thích (vd pictogram arrow) hoặc gộp về class gần nhất, và ghi rõ quy tắc mapping trong docstring để đảm bảo reproducibility.

### 14.7. Gate để coi Phase 8 hoàn thành

- [ ] Zero-shot inference chạy được không lỗi trên ít nhất 3/4 dataset ngoài
- [ ] `size_stratified_breakdown.json` có đủ 3 bin cho cả Bosch-test và external test
- [ ] `robustness_corruption.json` có đủ 5 loại corruption × 3 severity
- [ ] `latency_benchmark.json` có params/FLOPs/FPS cho M0 và M4
- [ ] `multi_seed_variance.json` có mean±std từ ≥3 seed cho M4
- [ ] Bảng tổng hợp kết quả đưa vào README.md / paper draft

---

## PHỤ LỤC A — Những Điểm Cần Chú ý Đặc biệt

### A.1. Những Lỗi Thường Gặp

```python
# ❌ LỖI: Dùng NMS cứng trong Stage 1
# → Xóa nhầm đèn đứng sát nhau → Recall giảm mạnh
# ✅ FIX: Override bằng Soft-NMS

# ❌ LỖI: Không project P4/P5/fcand về cùng 256-dim trước Deformable Attention
# → Channel mismatch error khi tính Attention
# ✅ FIX: FeatureProjection (Conv 1×1) cho từng feature map

# ❌ LỖI: Cài mmcv không đúng CUDA version
# → MultiScaleDeformableAttention báo lỗi khi chạy
# ✅ FIX: Dùng đúng CUDA version trong URL download mmcv

# ❌ LỖI: Dùng random negatives trong InfoNCE
# → Loss hội tụ chậm trên Bosch (quá nhiều easy negatives)
# ✅ FIX: Hard Negative Mining — lấy FP có confidence cao nhất

# ❌ LỖI: L_topology không giảm
# → Kiểm tra lại normalization trong InfoNCE (F.normalize trước dot product)
# ✅ FIX: Đảm bảo F.normalize(zi, dim=-1) và F.normalize(z_pos, dim=-1)
```

### A.2. Memory Management

```python
# Batch size 32 ~ 64 yêu cầu ~20-40GB VRAM (ảnh 1280×720, N~1000 candidates)
# Nếu OOM:
# 1. Giảm batch size xuống 8~16
# 2. Giảm K (n_points trong Deformable Attention) từ 8 → 4
# 3. Dùng gradient checkpointing:
model = torch.utils.checkpoint.checkpoint_wrapper(model)
```

### A.3. Monitoring Training Health

```python
# Dấu hiệu training HEALTHY:
# - L_det:      giảm đều từ epoch 1
# - L_topology: giảm chậm hơn, ổn định sau epoch 10-20
# - L_verify:   giảm từ epoch 5-10 khi topology embedding đủ tốt

# Dấu hiệu training CÓ VẤN ĐỀ:
# - L_topology không giảm → lỗi InfoNCE implementation
# - L_det tăng sau epoch 20 → learning rate quá cao
# - NaN loss → gradient explosion, thêm gradient clipping
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
optimizer.step()
```

---

*Tài liệu này được viết để làm ngữ cảnh đầy đủ cho AI Coding Agent. Mọi quyết định kiến trúc, hyperparameter, và cài đặt kỹ thuật đã được ghi chép rõ ràng kèm lý do. Khi implement, hãy đọc phần "What NOT to do" và "Lỗi Thường Gặp" trước.*
