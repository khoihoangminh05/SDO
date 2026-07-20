"""Phase 1 gate tests — DataLoader and EDA."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN_YAML = REPO / "apps/worker/datasets/dataset_train_rgb/train.yaml"
SKIP_NO_DATA = not TRAIN_YAML.is_file()


def test_yolo_dir_dataset_and_fallback(tmp_path: Path) -> None:
    """Val split can load from rgb/val PNG+TXT when val.yaml is absent."""
    from PIL import Image

    from data.dataset import YoloDirDataset, resolve_dataset_path

    img_dir = tmp_path / "rgb" / "val"
    img_dir.mkdir(parents=True)
    Image.new("RGB", (1280, 720), color=(0, 0, 0)).save(img_dir / "frame.png")
    (img_dir / "frame.txt").write_text("2 0.5 0.5 0.05 0.05\n", encoding="utf-8")

    resolved, mode = resolve_dataset_path(tmp_path / "val.yaml")
    assert mode == "yolo_dir"
    assert resolved == img_dir.resolve()

    dataset = YoloDirDataset(resolved)
    image, boxes = dataset[0]
    assert image.shape == (3, 720, 1280)
    assert boxes.shape[1] == 5
    assert boxes[0, 4] == 0.0  # yolo green -> TTLD Green


@pytest.mark.skipif(SKIP_NO_DATA, reason="BSTLD train.yaml not found")
def test_dataloader_smoke() -> None:
    """T1.A — DataLoader shape and box format."""
    from data.dataset import create_dataloader

    loader = create_dataloader(TRAIN_YAML, batch_size=4, num_workers=0, split="train")
    images, targets = next(iter(loader))

    assert images.shape == (4, 3, 720, 1280), f"Expected (4,3,720,1280), got {images.shape}"
    assert isinstance(targets, list) and len(targets) == 4
    assert all(t.shape[1] == 5 for t in targets if len(t) > 0), "Box format must be N×5"
    assert not torch.isnan(images).any(), "NaN in images"


@pytest.mark.skipif(SKIP_NO_DATA, reason="BSTLD train.yaml not found")
def test_eda_sanity() -> None:
    """T1.B — EDA statistics within expected ranges."""
    from scripts.eda_bosch import analyze_bbox_distribution

    out = ROOT / "logs" / "test_eda"
    summary = analyze_bbox_distribution(TRAIN_YAML, out)

    assert summary["pct_width_lt_10px"] > 30, "Expected many tiny boxes (w < 10px)"
    assert summary["avg_boxes_per_image"] > 2, "Expected > 2 boxes per image on average"
    red_pct = summary["class_counts"].get("Red", 0) / max(summary["total_boxes"], 1)
    assert red_pct > 0.2, "Red class should be common in BSTLD"

    assert (out / "eda_summary.json").is_file()
    assert (out / "bbox_distribution.png").is_file()


@pytest.mark.skipif(SKIP_NO_DATA, reason="BSTLD train.yaml not found")
def test_prepare_ultralytics_yaml() -> None:
    """Runtime bstld yaml has portable path."""
    from utils.config import resolve_path
    from utils.yolo_baseline import prepare_ultralytics_yaml

    src = resolve_path("../apps/worker/datasets/bstld.yaml")
    out = prepare_ultralytics_yaml(src, ROOT / "logs" / "test_bstld.yaml")
    import yaml

    with out.open(encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)
    assert "path" in parsed
    assert Path(parsed["path"]).is_dir()
