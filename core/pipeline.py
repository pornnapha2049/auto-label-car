import os
import json
import zipfile
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from ultralytics import YOLO, SAM

# Standard 8 vehicle categories matching Set1 / CVAT schema
DEFAULT_CATEGORIES = [
    {"id": 1, "name": "car", "supercategory": ""},
    {"id": 2, "name": "commercial_pickup", "supercategory": ""},
    {"id": 3, "name": "home_pickup", "supercategory": ""},
    {"id": 4, "name": "SUV", "supercategory": ""},
    {"id": 5, "name": "truck", "supercategory": ""},
    {"id": 6, "name": "van", "supercategory": ""},
    {"id": 7, "name": "bus", "supercategory": ""},
    {"id": 8, "name": "motorbike", "supercategory": ""}
]

# Mapping standard COCO YOLO class names to Set1 category IDs
STANDARD_YOLO_TO_CATEGORY_ID = {
    "car": 1,
    "commercial_pickup": 2,
    "home_pickup": 3,
    "suv": 4,
    "truck": 5,
    "van": 6,
    "bus": 7,
    "motorcycle": 8,
    "motorbike": 8
}

# Distinct colors for visualization (BGR)
CLASS_COLORS = {
    "car": (0, 215, 255),               # Yellow-Orange
    "commercial_pickup": (255, 191, 0), # Deep Amber
    "home_pickup": (0, 165, 255),       # Orange
    "suv": (147, 20, 255),              # Deep Pink
    "truck": (255, 69, 0),              # Red-Orange
    "van": (255, 0, 255),               # Magenta
    "bus": (50, 205, 50),               # Lime Green
    "motorbike": (255, 144, 30),        # Sky Blue
    "motorcycle": (255, 144, 30),
    "vehicle": (0, 255, 127)
}

class CarSegmenter:
    """
    Automated Car / Vehicle Segmentation Pipeline:
    1. YOLO: Fast vehicle detection & rough bounding box localization
    2. SAM: Precision zero-shot boundary / polygon mask refinement
    3. COCO Format: 100% compliant with CVAT Set1 schema (instances_default.json)
    """

    def __init__(
        self,
        yolo_model_name: str = "yolov8n.pt",
        sam_model_name: str = "mobile_sam.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        target_classes: Optional[List[str]] = None,
        approx_tolerance: float = 1.0,
        min_area: float = 25.0,
        categories: Optional[List[Dict]] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.approx_tolerance = approx_tolerance
        self.min_area = min_area

        # Active COCO categories (defaults to Set1 8 classes)
        self.coco_categories = categories or list(DEFAULT_CATEGORIES)
        self.category_name_to_id = {c["name"].lower(): c["id"] for c in self.coco_categories}
        # Add alias motorcycle -> motorbike
        if "motorbike" in self.category_name_to_id:
            self.category_name_to_id["motorcycle"] = self.category_name_to_id["motorbike"]

        if target_classes is None:
            self.target_classes = [c["name"].lower() for c in self.coco_categories] + ["motorcycle"]
        else:
            self.target_classes = [c.lower() for c in target_classes]

        self.yolo_model_name = yolo_model_name
        self.sam_model_name = sam_model_name
        self.yolo = YOLO(yolo_model_name)
        self.sam = SAM(sam_model_name)

        self._build_yolo_class_filters()

    def _build_yolo_class_filters(self):
        """Map YOLO internal model class IDs to target classes"""
        self.target_class_indices = []
        for idx, name in self.yolo.names.items():
            name_lower = name.lower()
            if name_lower in self.target_classes or (name_lower == "motorcycle" and "motorbike" in self.target_classes):
                self.target_class_indices.append(idx)

    def set_models(self, yolo_model_name: str, sam_model_name: str):
        """Update active YOLO or SAM models"""
        if yolo_model_name != self.yolo_model_name:
            self.yolo_model_name = yolo_model_name
            self.yolo = YOLO(yolo_model_name)
            self._build_yolo_class_filters()

        if sam_model_name != self.sam_model_name:
            self.sam_model_name = sam_model_name
            self.sam = SAM(sam_model_name)

    def set_target_classes(self, target_classes: List[str]):
        """Update targeted vehicle classes"""
        self.target_classes = [c.lower() for c in target_classes]
        if "motorbike" in self.target_classes and "motorcycle" not in self.target_classes:
            self.target_classes.append("motorcycle")
        self._build_yolo_class_filters()

    def load_schema_from_json(self, json_path: str):
        """Load categories directly from a sample instances_default.json"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "categories" in data:
            self.coco_categories = data["categories"]
            self.category_name_to_id = {c["name"].lower(): c["id"] for c in self.coco_categories}
            if "motorbike" in self.category_name_to_id:
                self.category_name_to_id["motorcycle"] = self.category_name_to_id["motorbike"]
            self._build_yolo_class_filters()

    def simplify_polygon(self, polygon: np.ndarray) -> np.ndarray:
        """Simplify polygon contour using Douglas-Peucker algorithm"""
        if self.approx_tolerance <= 0 or len(polygon) <= 4:
            return polygon
        poly_float = polygon.astype(np.float32)
        approx = cv2.approxPolyDP(poly_float, self.approx_tolerance, True)
        if len(approx) >= 3:
            return approx.reshape(-1, 2)
        return polygon

    def process_image(
        self,
        image_input: Union[str, np.ndarray],
        image_id: int = 1,
        file_name: str = "image.jpg"
    ) -> Dict:
        """
        Run detection with YOLO and refine boundaries with SAM.
        Returns annotations strictly formatted like Set1 instances_default.json.
        """
        if isinstance(image_input, (str, Path)):
            file_name = os.path.basename(str(image_input))
            image = cv2.imread(str(image_input))
            if image is None:
                raise ValueError(f"Could not load image: {image_input}")
        else:
            image = image_input

        height, width = image.shape[:2]

        # 1. YOLO Detection
        yolo_results = self.yolo(
            image,
            classes=self.target_class_indices if self.target_class_indices else None,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]

        boxes = []
        classes = []
        scores = []

        if len(yolo_results.boxes) > 0:
            for b in yolo_results.boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                cls_idx = int(b.cls[0].cpu().numpy())
                cls_name = self.yolo.names[cls_idx].lower()
                conf = float(b.conf[0].cpu().numpy())

                # Standardize motorcycle to motorbike if needed
                if cls_name == "motorcycle" and "motorbike" in self.category_name_to_id:
                    target_name = "motorbike"
                else:
                    target_name = cls_name

                boxes.append(xyxy)
                classes.append(target_name)
                scores.append(conf)

        annotations = []
        refined_polygons = []

        # 2. SAM Mask Refinement
        if boxes:
            try:
                sam_results = self.sam(
                    image,
                    bboxes=boxes,
                    device=self.device,
                    verbose=False
                )[0]

                if sam_results.masks and len(sam_results.masks.xy) > 0:
                    for i, poly in enumerate(sam_results.masks.xy):
                        if i >= len(boxes):
                            break
                        if len(poly) < 3:
                            continue

                        # Polygon simplification
                        simplified = self.simplify_polygon(poly)
                        area = float(cv2.contourArea(simplified.astype(np.float32)))
                        if area < self.min_area:
                            continue

                        cls_name = classes[i]
                        # Lookup category ID from Set1 schema (fallback to 1 if unknown)
                        cat_id = self.category_name_to_id.get(
                            cls_name, STANDARD_YOLO_TO_CATEGORY_ID.get(cls_name, 1)
                        )

                        # Bounding box in COCO format: [x, y, w, h] (float)
                        x_coords = simplified[:, 0]
                        y_coords = simplified[:, 1]
                        x_min = float(round(np.min(x_coords), 1))
                        y_min = float(round(np.min(y_coords), 1))
                        box_w = float(round(np.max(x_coords) - x_min, 1))
                        box_h = float(round(np.max(y_coords) - y_min, 1))

                        # Flatten coordinates as floats (e.g., [291.0, 278.0, ...])
                        poly_flat = [float(round(coord, 1)) for coord in simplified.flatten().tolist()]

                        # Annotation object matching Set1 instances_default.json EXACTLY
                        ann = {
                            "image_id": image_id,
                            "category_id": cat_id,
                            "segmentation": [poly_flat],
                            "area": float(round(area, 1)),
                            "bbox": [x_min, y_min, box_w, box_h],
                            "iscrowd": 0,
                            "attributes": {
                                "occluded": False
                            }
                        }
                        annotations.append(ann)
                        refined_polygons.append((simplified, cls_name, scores[i]))
            except Exception as e:
                print(f"[Warning] SAM refinement exception: {e}")

        # 3. Create visual overlay
        vis_image = self.draw_visualizations(image, refined_polygons, boxes, classes, scores)

        # Image info matching Set1 instances_default.json
        image_info = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": file_name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        }

        return {
            "image_info": image_info,
            "annotations": annotations,
            "visual_image": vis_image,
            "count": len(annotations)
        }

    def draw_visualizations(
        self,
        image: np.ndarray,
        refined_polygons: List[Tuple[np.ndarray, str, float]],
        raw_boxes: List[List[float]],
        raw_classes: List[str],
        raw_scores: List[float]
    ) -> np.ndarray:
        """Render crisp masks, outlines, and label tags onto image"""
        vis = image.copy()
        overlay = image.copy()

        for poly, cls_name, conf in refined_polygons:
            color = CLASS_COLORS.get(cls_name, (0, 255, 0))
            pts = poly.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)

        cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)

        for poly, cls_name, conf in refined_polygons:
            color = CLASS_COLORS.get(cls_name, (0, 255, 0))
            pts = poly.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

            x_min = int(np.min(poly[:, 0]))
            y_min = int(np.min(poly[:, 1]))
            label_text = f"{cls_name} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            badge_y1 = max(0, y_min - text_h - 8)
            badge_y2 = badge_y1 + text_h + 8
            badge_x2 = min(vis.shape[1], x_min + text_w + 10)

            cv2.rectangle(vis, (x_min, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(
                vis,
                label_text,
                (x_min + 5, badge_y2 - baseline - 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        return vis

    def create_coco_dataset(
        self,
        images_info: List[Dict],
        annotations_list: List[Dict]
    ) -> Dict:
        """
        Assemble full COCO 1.0 JSON structure matching Set1 instances_default.json 100%
        """
        coco_annotations = []
        for ann_id, ann in enumerate(annotations_list, start=1):
            ann_copy = dict(ann)
            ann_copy["id"] = ann_id
            coco_annotations.append(ann_copy)

        return {
            "licenses": [
                {
                    "name": "",
                    "id": 0,
                    "url": ""
                }
            ],
            "info": {
                "contributor": "",
                "date_created": "",
                "description": "",
                "url": "",
                "version": "",
                "year": ""
            },
            "categories": self.coco_categories,
            "images": images_info,
            "annotations": coco_annotations
        }

    @staticmethod
    def package_coco_zip(coco_json_path: str, zip_output_path: str):
        """
        Pack COCO JSON into a zip archive with paths structured for CVAT COCO 1.0 import
        """
        with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(coco_json_path, arcname="instances_default.json")
            zf.write(coco_json_path, arcname="annotations/instances_default.json")
