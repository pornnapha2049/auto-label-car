import os
import json
import zipfile
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from ultralytics import YOLO, SAM

# Standard COCO vehicle class mappings
DEFAULT_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Distinct colors for visualization (BGR)
CLASS_COLORS = {
    "car": (0, 215, 255),        # Gold / Yellow-Orange
    "motorcycle": (255, 144, 30), # Deep Sky Blue
    "bus": (50, 205, 50),        # Lime Green
    "truck": (255, 69, 0),       # Blue / Red-Orange
    "vehicle": (0, 255, 127)     # Spring Green
}

class CarSegmenter:
    """
    Automated Car / Vehicle Segmentation Pipeline combining:
    1. YOLO: Fast vehicle detection & classification
    2. SAM: Precision zero-shot mask refinement
    3. COCO Format: Standards-compliant export for CVAT & training
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
        min_area: float = 25.0
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.approx_tolerance = approx_tolerance
        self.min_area = min_area

        # Selected vehicle classes
        if target_classes is None:
            self.target_classes = ["car", "truck", "bus", "motorcycle"]
        else:
            self.target_classes = [c.lower() for c in target_classes]

        # Initialize models
        self.yolo_model_name = yolo_model_name
        self.sam_model_name = sam_model_name
        self.yolo = YOLO(yolo_model_name)
        self.sam = SAM(sam_model_name)

        # Build class index filters from YOLO's internal names
        self.target_class_indices = []
        self.coco_categories = []
        self.class_name_to_cat_id = {}

        cat_id = 1
        for idx, name in self.yolo.names.items():
            name_lower = name.lower()
            if name_lower in self.target_classes:
                self.target_class_indices.append(idx)
                self.coco_categories.append({
                    "id": cat_id,
                    "name": name_lower,
                    "supercategory": "vehicle"
                })
                self.class_name_to_cat_id[name_lower] = cat_id
                cat_id += 1

    def set_models(self, yolo_model_name: str, sam_model_name: str):
        """Update active YOLO or SAM models"""
        if yolo_model_name != self.yolo_model_name:
            self.yolo_model_name = yolo_model_name
            self.yolo = YOLO(yolo_model_name)
            # Rebuild class filters
            self.target_class_indices = []
            for idx, name in self.yolo.names.items():
                if name.lower() in self.target_classes:
                    self.target_class_indices.append(idx)

        if sam_model_name != self.sam_model_name:
            self.sam_model_name = sam_model_name
            self.sam = SAM(sam_model_name)

    def set_target_classes(self, target_classes: List[str]):
        """Update targeted vehicle classes"""
        self.target_classes = [c.lower() for c in target_classes]
        self.target_class_indices = []
        self.coco_categories = []
        self.class_name_to_cat_id = {}

        cat_id = 1
        for idx, name in self.yolo.names.items():
            name_lower = name.lower()
            if name_lower in self.target_classes:
                self.target_class_indices.append(idx)
                self.coco_categories.append({
                    "id": cat_id,
                    "name": name_lower,
                    "supercategory": "vehicle"
                })
                self.class_name_to_cat_id[name_lower] = cat_id
                cat_id += 1

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
        Returns:
            Dict containing annotations, visual image, and detected count
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

                if not self.target_classes or cls_name in self.target_classes:
                    boxes.append(xyxy)
                    classes.append(cls_name)
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
                        cat_id = self.class_name_to_cat_id.get(cls_name, 1)

                        # Bounding box in COCO format: [x, y, w, h]
                        x_coords = simplified[:, 0]
                        y_coords = simplified[:, 1]
                        x_min = float(np.min(x_coords))
                        y_min = float(np.min(y_coords))
                        box_w = float(np.max(x_coords) - x_min)
                        box_h = float(np.max(y_coords) - y_min)

                        poly_flat = simplified.flatten().tolist()

                        ann = {
                            "image_id": image_id,
                            "category_id": cat_id,
                            "category_name": cls_name,
                            "segmentation": [poly_flat],
                            "area": round(area, 2),
                            "bbox": [round(x_min, 2), round(y_min, 2), round(box_w, 2), round(box_h, 2)],
                            "score": round(scores[i], 4),
                            "iscrowd": 0
                        }
                        annotations.append(ann)
                        refined_polygons.append((simplified, cls_name, scores[i]))
            except Exception as e:
                print(f"[Warning] SAM refinement fallback: {e}")

        # 3. Create visual overlay
        vis_image = self.draw_visualizations(image, refined_polygons, boxes, classes, scores)

        image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
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
        """Render masks, outlines, and label tags onto image"""
        vis = image.copy()
        overlay = image.copy()

        for poly, cls_name, conf in refined_polygons:
            color = CLASS_COLORS.get(cls_name, (0, 255, 0))
            pts = poly.astype(np.int32).reshape((-1, 1, 2))

            # Filled mask
            cv2.fillPoly(overlay, [pts], color)

        # Alpha blend masks
        cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)

        # Draw crisp contour boundaries and tags
        for poly, cls_name, conf in refined_polygons:
            color = CLASS_COLORS.get(cls_name, (0, 255, 0))
            pts = poly.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

            # Label badge
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

            # Draw background tag
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
        """Assemble full COCO 1.0 JSON structure"""
        coco_annotations = []
        for ann_id, ann in enumerate(annotations_list, start=1):
            ann_copy = dict(ann)
            ann_copy["id"] = ann_id
            if "category_name" in ann_copy:
                del ann_copy["category_name"]
            coco_annotations.append(ann_copy)

        return {
            "info": {
                "description": "Auto-annotated Car Segmentation with YOLO + SAM",
                "version": "1.0",
                "year": 2026,
                "contributor": "YOLO-SAM CVAT Pipeline"
            },
            "licenses": [],
            "images": images_info,
            "annotations": coco_annotations,
            "categories": self.coco_categories
        }

    @staticmethod
    def package_coco_zip(coco_json_path: str, zip_output_path: str):
        """
        Pack COCO JSON into a zip archive with paths structured for CVAT COCO 1.0 import
        """
        with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(coco_json_path, arcname="instances_default.json")
            zf.write(coco_json_path, arcname="annotations/instances_default.json")
