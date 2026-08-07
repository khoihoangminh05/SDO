# TTLD-Net — Báo cáo kết quả thí nghiệm

**Dataset:** Bosch Small Traffic Lights Dataset (BSTLD / BoschTLD)  
**Cập nhật:** 2026-08  
**Trạng thái:** Phase 7 ablation — đang làm (đã có M0, M1)

---

## 1. Vì sao BoschTLD khó?

Bosch Small Traffic Lights là benchmark **rất khó** cho object detection:

| Đặc điểm | Ý nghĩa |
|----------|---------|
| Đèn rất nhỏ | Nhiều box chỉ vài pixel (thường &lt; 10 px chiều rộng) |
| IoU@0.5 khắc nghiệt | Lệch 1–2 px là mất match → AP tụt mạnh |
| Background phức tạp | Đường phố, phản chiếu, bóng, đèn mờ / off |
| Mất cân lớp | Green / Yellow / Red / Off không đều |

→ AP tuyệt đối trên BSTLD **thấp hơn nhiều** so với COCO/VOC là bình thường, kể cả với YOLO mạnh.

---

## 2. Các biến thể ablation

| ID | Tên | Nội dung |
|----|-----|----------|
| **M0** | YOLO26n baseline | Ultralytics YOLO26n — detector end-to-end |
| **M1** | Shallow + Soft-NMS | Stage-1: sinh candidate high-recall (P1–P3) |
| **M2** | + Focal | M1 + tinh chỉnh focal loss |
| **M3** | + Topology | + Deformable attention topology + InfoNCE |
| **M4** | Full TTLD-Net | + Verification MLP (lọc FP) |

Loss đầy đủ (M4):  
`L = L_det + λ₁·L_topology + λ₂·L_verify`

---

## 3. Điều kiện đo (quan trọng)

Kết quả dưới đây chạy trên **Google Colab Pro+**, profile `--proplus` (**rút gọn**, chưa phải full paper):

| Tham số | Giá trị dùng |
|---------|--------------|
| Epochs | 50 |
| Train subset | ~80% |
| Max batches / epoch | 500 |
| Image size | 640 × 1120 |
| Eval conf (M1) | **0.05** (high-recall) |
| Eval conf (M0) | **0.5** (Ultralytics mặc định) |

> **Lưu ý so sánh:** M0 và M1 **không cùng operating point** (0.5 vs 0.05).  
> M1 cố tình conf thấp để đo khả năng bắt đèn (recall), chấp nhận nhiều FP — phần lọc FP thuộc M3/M4 (chưa xong).

---

## 4. Kết quả đã có

### Bảng tổng hợp

| Variant | Vai trò | AP50 | Recall | Precision | Ghi chú |
|---------|---------|-----:|-------:|----------:|---------|
| **M0** YOLO26n | Baseline đầy đủ | **~0.54** | — | — | conf = 0.5 |
| **M1** Shallow | Stage-1 generator | **0.2412** | **0.5355** | 0.0885 | conf = 0.05 |
| M2 Focal | — | — | — | — | Chưa train |
| M3 Topology | — | — | — | — | Chưa train |
| M4 Full | — | — | — | — | Chưa train |

### Chi tiết M1 (đã validate)

| Metric | Giá trị |
|--------|--------:|
| AP50 (VOC-style) | 0.2412 |
| mAP @ 0.5:0.95 | 0.1447 |
| APsmall (proxy báo cáo) | 0.2171 |
| Recall @ conf 0.05 | 0.5355 |
| Precision @ conf 0.05 | 0.0885 |
| FPR | 0.9115 |
| TP / FP / FN | 1751 / 18027 / 1519 |
| Số GT trên val | 3270 |
| Image size | 640 × 1120 |
| max_dets / ảnh | 100 |

---

## 5. Đọc kết quả thế nào? (cho thầy)

1. **BoschTLD khó** → AP ~0.2–0.5 vẫn có ý nghĩa thực tế với đèn siêu nhỏ.  
2. **M0 YOLO26 đang tốt hơn về AP tổng** (~0.54) — đây là baseline mạnh, end-to-end.  
3. **M1 chưa “thua” theo nghĩa hỏng model**:  
   - Đã học được (trước bug technical thì AP≈0).  
   - Recall ~53% ở conf rất thấp → Stage-1 bắt được khá nhiều đèn.  
   - Precision thấp là **đúng thiết kế** high-recall; cần M3/M4 để lọc FP.  
4. **Chưa công bằng nếu kết luận “TTLD kém YOLO”** khi mới xong Stage-1 + train rút gọn.  
5. Hướng tiếp: hoàn thành **M2 → M3 → M4**, rồi so lại với M0 trên **cùng protocol** (full data nếu có GPU).

---

## 6. Tóm tắt một câu

> Trên BoschTLD (bộ khó, đèn cực nhỏ), YOLO26n (M0) đạt AP50 ~0.54; TTLD-Net mới hoàn thành Stage-1 (M1) với AP50 ~0.24 và recall ~0.54 ở conf 0.05 dưới protocol Colab rút gọn — chứng minh generator học được, nhưng chưa có verification nên precision còn thấp; ablation M2–M4 và train full vẫn đang tiếp tục.

---

## 7. File liên quan trong repo

| File | Nội dung |
|------|----------|
| `ttld-net/README.md` | Tổng quan phương pháp |
| `ttld-net/document/RESULTS.md` | Bản EN ngắn (đồng bộ số liệu) |
| `ttld-net/configs/m0_*.yaml` … `m4_*.yaml` | Cấu hình ablation |
| `document/PROJECT_STATE.md` | Tiến độ phase |
| `results/ablation/*_metrics.json` | JSON thô (khi sync từ máy train) |
