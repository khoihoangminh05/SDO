"""
Centralized runtime configuration for the worker detection pipeline.

All values mirror the previous hard-coded defaults so behaviour is unchanged
unless an environment variable override is supplied. This keeps tuning of the
detection / OCR pipeline in one place instead of scattered magic numbers.
"""
import os


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ── Debugging ────────────────────────────────────────────────────────────────
# When enabled, intermediate crops are written to the `runs/` directory. This is
# expensive (disk I/O on every inference) and must stay OFF in production.
DEBUG_CROPS: bool = _get_bool("DEBUG_CROPS", False)
DEBUG_DIR: str = os.environ.get("DEBUG_DIR", "runs")

# ── Base traffic-light detection (Stage 1) ───────────────────────────────────
BASE_CONF_THRESHOLD: float = _get_float("BASE_CONF_THRESHOLD", 0.10)
BASE_IMGSZ: int = _get_int("BASE_IMGSZ", 1280)
# Boxes smaller than this (pixels) are treated as noise and dropped.
MIN_BOX_WIDTH: int = _get_int("MIN_BOX_WIDTH", 20)
MIN_BOX_HEIGHT: int = _get_int("MIN_BOX_HEIGHT", 15)

# ── Lamp-box detection (Stage 3) ─────────────────────────────────────────────
# Fixed upscale factor used when adaptive upscaling is disabled. Also serves as
# the reference scale (4x) against which size-dependent thresholds are calibrated.
UPSCALE_FACTOR: float = _get_float("UPSCALE_FACTOR", 4.0)
CUSTOM_CONF_THRESHOLD: float = _get_float("CUSTOM_CONF_THRESHOLD", 0.10)

# Adaptive upscaling: tiny traffic-light crops are enlarged more aggressively so
# small lamps / digits become recoverable, while large crops are enlarged less
# to save compute. Disable to reproduce the legacy fixed-4x behaviour.
UPSCALE_ADAPTIVE: bool = _get_bool("UPSCALE_ADAPTIVE", True)
# Hard cap on the longest side of an upscaled crop (pixels) to bound memory/compute.
UPSCALE_MAX_DIM: int = _get_int("UPSCALE_MAX_DIM", 1920)


def compute_upscale_factor(box_w: float, box_h: float) -> float:
    """
    Pick an upscale factor based on the crop's shortest side.

    Smaller lamps need a larger magnification to expose enough pixels for the
    YOLO-P2 / HSV / OCR stages; large housings need little or none. The result
    is clamped so the longest upscaled side never exceeds ``UPSCALE_MAX_DIM``.
    """
    if not UPSCALE_ADAPTIVE:
        return UPSCALE_FACTOR

    short_side = min(box_w, box_h)
    long_side = max(box_w, box_h)

    if short_side < 20:
        scale = 8.0
    elif short_side < 40:
        scale = 6.0
    elif short_side < 80:
        scale = 4.0
    elif short_side < 150:
        scale = 2.0
    else:
        scale = 1.5

    # Guard against enormous crops (e.g. a tall narrow pole upscaled 8x).
    if long_side > 0 and long_side * scale > UPSCALE_MAX_DIM:
        scale = max(1.0, UPSCALE_MAX_DIM / long_side)

    return scale

# ── Weighted Box Fusion IoU thresholds ───────────────────────────────────────
WBF_IOU_POLE: float = _get_float("WBF_IOU_POLE", 0.30)
WBF_IOU_LAMP: float = _get_float("WBF_IOU_LAMP", 0.35)

# ── OCR (Stage 4) ────────────────────────────────────────────────────────────
OCR_CONF_THRESHOLD: float = _get_float("OCR_CONF_THRESHOLD", 0.15)
OCR_UPSCALE_FACTOR: float = _get_float("OCR_UPSCALE_FACTOR", 3.0)

# ── HSV colour thresholds (fallback lamp detection) ──────────────────────────
# Format: [Hue, Sat, Val]. Green hue extended to 110 to cover cyan LEDs.
HSV_RED1_LOWER = (0, 50, 130)
HSV_RED1_UPPER = (12, 255, 255)
HSV_RED2_LOWER = (165, 50, 130)
HSV_RED2_UPPER = (180, 255, 255)
HSV_YELLOW_LOWER = (13, 50, 130)
HSV_YELLOW_UPPER = (34, 255, 255)
HSV_GREEN_LOWER = (35, 35, 120)
HSV_GREEN_UPPER = (110, 255, 255)

# Minimum contour area for an HSV blob to count as a lamp. Calibrated on the
# reference 4x crop; the engine rescales it by (scale/4)^2 so the effective
# physical size filter stays constant under adaptive upscaling.
HSV_MIN_CONTOUR_AREA: float = _get_float("HSV_MIN_CONTOUR_AREA", 350.0)

# ── CLAHE contrast enhancement ───────────────────────────────────────────────
CLAHE_ENABLED: bool = _get_bool("CLAHE_ENABLED", False)
CLAHE_CLIP_LIMIT: float = _get_float("CLAHE_CLIP_LIMIT", 3.0)
CLAHE_TILE_GRID: int = _get_int("CLAHE_TILE_GRID", 8)

# ── Sign Filter thresholds ────────────────────────────────────────────────────
# Rule A — Tall-pole filter: very tall narrow detections are sign poles, not lights.
SIGN_FILTER_TALL_AR: float        = _get_float("SIGN_FILTER_TALL_AR", 1.7)
SIGN_FILTER_TALL_MAX_W: int       = _get_int("SIGN_FILTER_TALL_MAX_W", 150)
# Rule B — Square-fill filter: square-ish box where a colour sub fills most of it.
SIGN_FILTER_SQ_AR_MIN: float      = _get_float("SIGN_FILTER_SQ_AR_MIN", 0.65)
SIGN_FILTER_SQ_AR_MAX: float      = _get_float("SIGN_FILTER_SQ_AR_MAX", 1.55)
SIGN_FILTER_SQ_MIN_W: int         = _get_int("SIGN_FILTER_SQ_MIN_W", 80)
SIGN_FILTER_SQ_MAX_W: int         = _get_int("SIGN_FILTER_SQ_MAX_W", 160)
SIGN_FILTER_SQ_FILL_RATIO: float  = _get_float("SIGN_FILTER_SQ_FILL_RATIO", 0.75)
SIGN_FILTER_SQ_AREA_RATIO: float  = _get_float("SIGN_FILTER_SQ_AREA_RATIO", 0.55)
# Rule C (new) — Low-confidence empty: no colour sub + very low confidence.
GHOST_BOX_CONF_THRESHOLD: float   = _get_float("GHOST_BOX_CONF_THRESHOLD", 0.18)
# Deduplication IoU for colour sub-detections.
DEDUP_COLOR_IOU: float            = _get_float("DEDUP_COLOR_IOU", 0.40)

# ── OCR HSV pre-filter (tighter than main detector — applied on the cropped box) ──
OCR_HSV_RED1_LOWER  = (0,   25, 80)
OCR_HSV_RED1_UPPER  = (12,  255, 255)
OCR_HSV_RED2_LOWER  = (165, 25, 80)
OCR_HSV_RED2_UPPER  = (180, 255, 255)
OCR_HSV_YELLOW_LOWER = (13, 25, 80)
OCR_HSV_YELLOW_UPPER = (34, 255, 255)
OCR_HSV_GREEN_LOWER  = (35, 25, 80)
OCR_HSV_GREEN_UPPER  = (110, 255, 255)
# OCR digit validation: only keep recognised numbers in this range.
OCR_DIGIT_MIN: int = _get_int("OCR_DIGIT_MIN", 0)
OCR_DIGIT_MAX: int = _get_int("OCR_DIGIT_MAX", 99)

# ── Pipeline mode ─────────────────────────────────────────────────────────────
# "recursive"   — current crop+upscale+P2 pipeline (Phase 0-3)
# "single_pass" — YOLO26 full-frame single inference (Phase 5, requires new weights)
PIPELINE_MODE: str = os.environ.get("PIPELINE_MODE", "recursive")
