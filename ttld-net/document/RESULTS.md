# Experimental results (EN snapshot)

Vietnamese full report for advisors: **[`BAO_CAO_KET_QUA.md`](BAO_CAO_KET_QUA.md)**.

> Bosch Small Traffic Lights (BSTLD) is a **hard** benchmark: many boxes are only a few pixels wide, so IoU@0.5 is unforgiving and absolute AP is much lower than on COCO.

## Protocol (Colab `--proplus`, not full YAML)

| Setting | Value |
|---------|-------|
| Epochs | 50 |
| Train subset | ~80% |
| Max train batches / epoch | 500 |
| Image size | 640×1120 |
| M1 eval conf | 0.05 |
| M0 eval conf | 0.5 |

## Results

| Variant | Role | AP50 | Recall | Precision |
|---------|------|-----:|-------:|----------:|
| M0 YOLO26n | End-to-end baseline | ~0.54 @ 0.5 | — | — |
| M1 Shallow | Stage-1 high-recall | **0.2412** | **0.5355** | 0.0885 |
| M2–M4 | Focal / topology / full | — | — | pending |

### M1 detail

| Metric | Value |
|--------|------:|
| AP50 | 0.2412 |
| mAP50-95 | 0.1447 |
| TP / FP / FN | 1751 / 18027 / 1519 |
| num_gt | 3270 |

**Reading:** M0 is stronger on AP today. M1 shows the generator learns (recall≈0.54) but low precision at conf=0.05 is expected until M3/M4 verification. Do not treat Stage-1 alone as the final TTLD vs YOLO comparison.
