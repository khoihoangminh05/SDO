from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

BoundingBox = Tuple[float, float, float, float]


class Detection(BaseModel):
    class_id: int = Field(..., description="ID of the detected class")
    class_name: str = Field(..., description="Name of the detected class")
    confidence: float = Field(..., description="Confidence score of the detection")
    local_bbox: BoundingBox = Field(
        ...,
        description="Local bounding box [x_min, y_min, x_max, y_max]",
    )
    global_bbox: BoundingBox = Field(
        ...,
        description="Global bounding box [x_min, y_min, x_max, y_max]",
    )
    sub_detections: Optional[List["Detection"]] = Field(
        default=None,
        description="Nested sub-detections for hierarchical object data",
    )


# For recursive models in Pydantic v2
Detection.model_rebuild()


class RegionNode(BaseModel):
    node_id: str = Field(..., description="Unique identifier for this region node")
    image_id: str = Field(..., description="Target image identifier")
    zoom_level: int = Field(..., description="Zoom depth of this node")
    global_bbox: BoundingBox = Field(
        ..., description="Global coordinates bounding this node"
    )
    is_leaf: bool = Field(
        ..., description="Whether this node has children nodes zoom-ins"
    )
    detections: List[Detection] = Field(
        ..., description="Detections found at this zoom level region"
    )
    children: Optional[List["RegionNode"]] = Field(
        default=None, description="Zoom-in child nodes for recursive detection"
    )


# For recursive models in Pydantic v2
RegionNode.model_rebuild()


class ExploreRequestDto(BaseModel):
    image_id: str = Field(..., description="Target image identifier")
    image_path: str = Field(
        ..., description="Absolute file path of the image on the host"
    )
    zoom_level: int = Field(..., description="Zoom depth of the window")
    global_bbox: BoundingBox = Field(
        ..., description="Coordinates bounding the target exploration window"
    )
