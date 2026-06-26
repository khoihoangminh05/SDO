export type BoundingBox = [number, number, number, number]; // [x_min, y_min, x_max, y_max]

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  local_bbox: BoundingBox;
  global_bbox: BoundingBox;
  sub_detections?: Detection[];
}

export interface RegionNode {
  node_id: string;
  image_id: string;
  zoom_level: number;
  global_bbox: BoundingBox;
  is_leaf: boolean;
  detections: Detection[];
  children?: RegionNode[];
}

export interface ExploreRequestDto {
  image_id: string;
  zoom_level: number;
  global_bbox: BoundingBox;
}
