# Trigger reload to import newly installed ensemble-boxes package
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

from contextlib import asynccontextmanager
import os
import logging
import cv2
from fastapi import FastAPI, HTTPException
from app.models import ExploreRequestDto, RegionNode
from app.services.yolo_service import yolo_service

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize YOLO model on startup
    yolo_service.initialize()
    from app.services.ocr_service import ocr_service
    ocr_service.initialize()
    yield


app = FastAPI(
    title="Recursive Object Detector Worker",
    version="1.0.0",
    description="Python FastAPI worker for Ultralytics YOLO object detection",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    return {"status": "ok", "service": "worker"}


@app.post(
    "/api/v1/detect",
    response_model=RegionNode,
    summary="Perform object detection on a region",
)
def detect_objects(request: ExploreRequestDto):
    """
    Perform object detection on a cropped region (tile) of a high-resolution image.
    """
    # Securely resolve path using image_id from /app/uploads
    uploads_dir = "/app/uploads"
    matched_file = None
    if os.path.exists(uploads_dir):
        for f in os.listdir(uploads_dir):
            if f.startswith(request.image_id):
                matched_file = f
                break

    if matched_file:
        image_path = os.path.normpath(os.path.join(uploads_dir, matched_file))
    else:
        # Fallback for local running / unit tests
        image_path = os.path.normpath(request.image_path)

    # Load the high-resolution image
    image = cv2.imread(image_path)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load image at path: {image_path}",
        )

    h, w, _ = image.shape

    # Crop the image strictly based on the global_bbox provided in the request
    # Ensure coordinates are within image boundaries and rounded properly.
    # At zoom level 0, request.global_bbox is [0.0, 0.0, 1.0, 1.0] (normalized),
    # representing the entire image. We default to full image dimensions.
    if request.zoom_level == 0 or (
        request.global_bbox[0] == 0.0
        and request.global_bbox[1] == 0.0
        and request.global_bbox[2] == 1.0
        and request.global_bbox[3] == 1.0
    ):
        x_min, y_min, x_max, y_max = 0, 0, w, h
    else:
        x_min = max(0, int(round(request.global_bbox[0])))
        y_min = max(0, int(round(request.global_bbox[1])))
        x_max = min(w, int(round(request.global_bbox[2])))
        y_max = min(h, int(round(request.global_bbox[3])))

    # Handle edge case where cropped size is zero or invalid
    if x_min >= x_max or y_min >= y_max:
        return RegionNode(
            node_id=f"node_{request.image_id}_{request.zoom_level}_{x_min}_{y_min}",
            image_id=request.image_id,
            zoom_level=request.zoom_level,
            global_bbox=request.global_bbox,
            is_leaf=True,
            detections=[],
        )

    box_width = x_max - x_min
    box_height = y_max - y_min
    max_dim = max(box_width, box_height)

    # Dynamic max_depth based on zoom_level and dimension
    if request.zoom_level == 0:
        max_depth = 3
    else:
        if max_dim > 2560:
            max_depth = 3
        elif max_dim > 1280:
            max_depth = 2
        elif max_dim > 640:
            max_depth = 1
        else:
            max_depth = 0

    global_bbox_pixel = [float(x_min), float(y_min), float(x_max), float(y_max)]
    stats = {"yolo_calls": 0, "skipped_nodes": 0}

    # Call Recursion Engine for single-pass detection, avoiding partitioning/slicing
    try:
        from app.services.recursion_engine import recursion_engine
        detections = recursion_engine.process_context_aware_recursion(
            image=image,
            global_bbox=global_bbox_pixel,
        )
    except Exception as e:
        logger.error(
            f"Error during detection processing: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during detection: {str(e)}",
        )

    logger.info("Single-pass detection finished.")

    is_leaf = True
    children = None

    node_id = f"node_{request.image_id}_{request.zoom_level}_{x_min}_{y_min}"
    return RegionNode(
        node_id=node_id,
        image_id=request.image_id,
        zoom_level=request.zoom_level,
        global_bbox=request.global_bbox,
        is_leaf=is_leaf,
        detections=detections,
        children=children,
    )
