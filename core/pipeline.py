import os
import json
import zipfile
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from ultralytics import YOLO, SAM

# 2 Major Categories: car & motorbike (all other vehicles grouped into car)
DEFAULT_CATEGORIES = [
    {"id": 1, "name": "car", "supercategory": ""},
    {"id": 2, "name": "motorbike", "supercategory": ""}
]
DEFAULT_VEHICLE_CLASSES = DEFAULT_CATEGORIES

# Distinct colors for visualization (BGR)
CLASS_COLORS = {
    "car": (0, 215, 255),        # Yellow-Orange
    "motorbike": (255, 144, 30), # Sky Blue
    "motorcycle": (255, 144, 30),
    "vehicle": (0, 255, 127)
}

class CarSegmenter:
    """
    Automated Car / Motorbike Segmentation Pipeline:
    1. YOLO: Fast vehicle detection (detects car, truck, bus, motorcycle)
    2. Grouping: Groups truck, bus, van, pickup -> 'car' (id: 1); motorcycle -> 'motorbike' (id: 2)
    3. SAM: Precision zero-shot boundary / polygon mask refinement
    4. COCO Format: 100% compliant COCO 1.0 schema for CVAT
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

        # Active COCO categories: exactly 2 classes (car: 1, motorbike: 2)
        self.coco_categories = categories or list(DEFAULT_CATEGORIES)
        self.category_name_to_id = {c["name"].lower(): c["id"] for c in self.coco_categories}
        if "motorbike" in self.category_name_to_id:
            self.category_name_to_id["motorcycle"] = self.category_name_to_id["motorbike"]

        # Default allowed vehicle types to detect
        if target_classes is None:
            self.target_classes = ["car", "motorbike"]
        else:
            self.target_classes = [c.lower() for c in target_classes]

        self.yolo_model_name = yolo_model_name
        self.sam_model_name = sam_model_name
        self.yolo = YOLO(yolo_model_name)
        self.sam = SAM(sam_model_name)

        self._build_yolo_class_filters()

    def _build_yolo_class_filters(self):
        """Map YOLO internal vehicle class IDs (car, motorcycle, bus, truck)"""
        self.target_class_indices = []
        for idx, name in self.yolo.names.items():
            name_lower = name.lower()
            # Include motorcycle if motorbike is requested
            if "motorbike" in self.target_classes and name_lower in ["motorcycle", "motorbike"]:
                self.target_class_indices.append(idx)
            # Include car, truck, bus, van etc. if car is requested
            elif "car" in self.target_classes and name_lower in ["car", "bus", "truck", "van", "suv", "pickup"]:
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
        """Update targeted classes (car and/or motorbike)"""
        self.target_classes = [c.lower() for c in target_classes]
        self._build_yolo_class_filters()

    def map_to_target_class(self, raw_class_name: str) -> Tuple[str, int]:
        """
        Maps any vehicle class to either 'motorbike' (2) or 'car' (1)
        """
        name_lower = raw_class_name.lower()
        if name_lower in ["motorcycle", "motorbike"]:
            return "motorbike", 2
        else:
            # All other vehicles (truck, bus, van, pickup, etc.) are merged into 'car'
            return "car", 1

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
        Classifies strictly into 'car' (1) or 'motorbike' (2).
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
        cat_ids = []
        scores = []

        if len(yolo_results.boxes) > 0:
            for b in yolo_results.boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                cls_idx = int(b.cls[0].cpu().numpy())
                raw_name = self.yolo.names[cls_idx].lower()
                conf = float(b.conf[0].cpu().numpy())

                # Map to 'car' (1) or 'motorbike' (2)
                mapped_name, cat_id = self.map_to_target_class(raw_name)

                # Filter by active target classes
                if mapped_name in self.target_classes:
                    boxes.append(xyxy)
                    classes.append(mapped_name)
                    cat_ids.append(cat_id)
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

                        simplified = self.simplify_polygon(poly)
                        area = float(cv2.contourArea(simplified.astype(np.float32)))
                        if area < self.min_area:
                            continue

                        cls_name = classes[i]
                        cat_id = cat_ids[i]

                        x_coords = simplified[:, 0]
                        y_coords = simplified[:, 1]
                        x_min = float(round(np.min(x_coords), 1))
                        y_min = float(round(np.min(y_coords), 1))
                        box_w = float(round(np.max(x_coords) - x_min, 1))
                        box_h = float(round(np.max(y_coords) - y_min, 1))

                        poly_flat = [float(round(coord, 1)) for coord in simplified.flatten().tolist()]

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
            color = CLASS_COLORS.get(cls_name, (0, 215, 255))
            pts = poly.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)

        cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)

        for poly, cls_name, conf in refined_polygons:
            color = CLASS_COLORS.get(cls_name, (0, 215, 255))
            pts = poly.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

            x_min = int(np.min(poly[:, 0]))
            y_min = int(np.min(poly[:, 1]))
            label_text = f"{cls_name.capitalize()} {conf:.2f}"
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
        Assemble COCO 1.0 JSON structure with 2 categories: car & motorbike
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
