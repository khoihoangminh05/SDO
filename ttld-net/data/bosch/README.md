# Bosch / BSTLD data paths

TTLD-Net configs reference the YOLO-converted BSTLD trees under the SDO worker app.

| Split | YAML (Bosch original) | YOLO images (relative to repo root) |
|-------|------------------------|-------------------------------------|
| Train | `apps/worker/datasets/dataset_train_rgb/train.yaml` | `apps/worker/datasets/dataset_train_rgb/rgb/train` |
| Val   | `apps/worker/datasets/dataset_val_sample/val.yaml` | `apps/worker/datasets/dataset_val_sample/rgb/val` |
| Test  | `apps/worker/datasets/dataset_test_rgb/test.yaml` | `apps/worker/datasets/dataset_test_rgb/rgb/test` |

Convert labels if needed (from `apps/worker`):

```bash
python scripts/convert_bstld_to_yolo.py --yaml datasets/dataset_train_rgb/train.yaml --images-root datasets
```

Ultralytics dataset config: `apps/worker/datasets/bstld.yaml`
