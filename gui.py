import os
import sys
import json
import time
import shutil
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QCheckBox, QSlider,
    QProgressBar, QTextEdit, QFileDialog, QGroupBox, QTabWidget,
    QMessageBox, QSplitter, QDialog, QFormLayout, QRadioButton,
    QButtonGroup, QFrame, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QIcon

from core.pipeline import CarSegmenter, DEFAULT_VEHICLE_CLASSES, CLASS_COLORS
from core.cvat_manager import CVATManager

# Modern Dark Theme Stylesheet
MODERN_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #313244;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 15px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
}
QPushButton {
    background-color: #89b4fa;
    color: #11111b;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
    border: none;
}
QPushButton:hover {
    background-color: #b4befe;
}
QPushButton:pressed {
    background-color: #74c7ec;
}
QPushButton#secondaryBtn {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
}
QPushButton#secondaryBtn:hover {
    background-color: #45475a;
}
QPushButton#actionBtn {
    background-color: #a6e3a1;
    color: #11111b;
    font-size: 14px;
    padding: 12px 20px;
}
QPushButton#actionBtn:hover {
    background-color: #94e2d5;
}
QLineEdit, QComboBox {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 6px;
    padding: 6px 10px;
    color: #cdd6f4;
    selection-background-color: #89b4fa;
}
QLineEdit:focus, QComboBox:focus {
    border: 1px solid #89b4fa;
}
QComboBox QAbstractItemView {
    background-color: #181825;
    color: #cdd6f4;
    selection-background-color: #313244;
}
QCheckBox {
    spacing: 8px;
    color: #cdd6f4;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #45475a;
    background-color: #181825;
}
QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
QProgressBar {
    border: 1px solid #313244;
    border-radius: 6px;
    text-align: center;
    background-color: #181825;
    color: #cdd6f4;
    font-weight: bold;
}
QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 5px;
}
QTextEdit {
    background-color: #11111b;
    border: 1px solid #313244;
    border-radius: 6px;
    padding: 8px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}
QTabWidget::pane {
    border: 1px solid #313244;
    border-radius: 6px;
}
QTabBar::tab {
    background-color: #181825;
    border: 1px solid #313244;
    padding: 8px 16px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: #a6adc8;
}
QTabBar::tab:selected {
    background-color: #313244;
    color: #89b4fa;
    font-weight: bold;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #313244;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #89b4fa;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #cdd6f4;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}
"""

class WorkerThread(QThread):
    progress_signal = pyqtSignal(str, int)
    image_processed_signal = pyqtSignal(dict)
    log_signal = pyqtSignal(str, str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(
        self,
        segmenter: CarSegmenter,
        cvat_mgr: CVATManager,
        mode: str,
        input_paths: list,
        project_id: int,
        task_id: int,
        task_name: str,
        output_dir: str
    ):
        super().__init__()
        self.segmenter = segmenter
        self.cvat_mgr = cvat_mgr
        self.mode = mode
        self.input_paths = input_paths
        self.project_id = project_id
        self.task_id = task_id
        self.task_name = task_name
        self.output_dir = output_dir
        self.is_running = True

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            vis_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)

            images_info = []
            all_annotations = []
            local_processed_images = []

            # -------------------------------------------------------------
            # MODE 1: Local Images -> Segment -> Upload New Task / Annotations
            # -------------------------------------------------------------
            if self.mode == "local_upload":
                if not self.input_paths:
                    self.finished_signal.emit(False, "No input images selected")
                    return

                total = len(self.input_paths)
                self.log_signal.emit(f"🚀 Starting segmentation on {total} local images...", "info")

                for i, img_path in enumerate(self.input_paths):
                    if not self.is_running:
                        break

                    file_name = os.path.basename(img_path)
                    self.progress_signal.emit(f"Segmenting {file_name} ({i+1}/{total})...", int(((i+1)/total) * 60))

                    res = self.segmenter.process_image(img_path, image_id=i+1, file_name=file_name)
                    images_info.append(res["image_info"])
                    all_annotations.extend(res["annotations"])
                    local_processed_images.append(img_path)

                    # Save visual overlay
                    vis_path = os.path.join(vis_dir, f"vis_{file_name}")
                    cv2.imwrite(vis_path, res["visual_image"])

                    self.log_signal.emit(
                        f"Image [{file_name}]: Detected {res['count']} vehicle(s)", "info"
                    )

                    # Send preview
                    self.image_processed_signal.emit({
                        "vis_path": vis_path,
                        "file_name": file_name,
                        "count": res["count"],
                        "current": i + 1,
                        "total": total
                    })

                # Assemble COCO Dataset
                coco_data = self.segmenter.create_coco_dataset(images_info, all_annotations)
                json_path = os.path.join(self.output_dir, "instances_default.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(coco_data, f, indent=2)

                zip_path = os.path.join(self.output_dir, "coco_annotations.zip")
                self.segmenter.package_coco_zip(json_path, zip_path)
                self.log_signal.emit(f"COCO Dataset saved: {len(all_annotations)} annotations", "success")

                # Upload to CVAT
                if self.cvat_mgr.is_connected:
                    self.progress_signal.emit("Uploading to CVAT server...", 75)
                    self.log_signal.emit("Uploading images & annotations to CVAT...", "info")

                    if self.task_id and self.task_id > 0:
                        # Upload annotations to existing task
                        ok, msg = self.cvat_mgr.upload_annotations_to_task(self.task_id, zip_path)
                        if ok:
                            self.log_signal.emit(msg, "success")
                        else:
                            self.finished_signal.emit(False, msg)
                            return
                    else:
                        # Create new task with images and annotations
                        t_name = self.task_name or f"Car_Seg_{time.strftime('%Y%m%d_%H%M%S')}"
                        created = self.cvat_mgr.create_task_with_data(
                            name=t_name,
                            image_paths=local_processed_images,
                            project_id=self.project_id if self.project_id > 0 else None,
                            annotation_zip_path=zip_path
                        )
                        self.log_signal.emit(
                            f"CVAT Task #{created['id']} '{created['name']}' created successfully!", "success"
                        )
                else:
                    self.log_signal.emit("Skipped CVAT upload (Not connected). Output saved locally.", "warning")

                self.progress_signal.emit("Complete!", 100)
                self.finished_signal.emit(True, f"Successfully processed {total} images with {len(all_annotations)} annotations!")

            # -------------------------------------------------------------
            # MODE 2: Existing CVAT Task -> Download Frames -> Segment -> Upload
            # -------------------------------------------------------------
            elif self.mode == "cvat_task":
                if not self.cvat_mgr.is_connected:
                    self.finished_signal.emit(False, "Not connected to CVAT server")
                    return
                if not self.task_id or self.task_id <= 0:
                    self.finished_signal.emit(False, "Please select an existing CVAT Task")
                    return

                self.log_signal.emit(f"Fetching frames from CVAT Task #{self.task_id}...", "info")
                frames_info = self.cvat_mgr.get_task_frames_info(self.task_id)
                total = len(frames_info)

                if total == 0:
                    self.finished_signal.emit(False, f"Task #{self.task_id} has no frames")
                    return

                self.log_signal.emit(f"Processing {total} frames from Task #{self.task_id}...", "info")

                for i, frame in enumerate(frames_info):
                    if not self.is_running:
                        break

                    f_id = frame["frame_id"]
                    f_name = frame["name"]
                    self.progress_signal.emit(f"Downloading & Segmenting {f_name} ({i+1}/{total})...", int(((i+1)/total) * 70))

                    img = self.cvat_mgr.get_task_frame_image(self.task_id, f_id)
                    if img is None:
                        continue

                    res = self.segmenter.process_image(img, image_id=i+1, file_name=f_name)
                    images_info.append(res["image_info"])
                    all_annotations.extend(res["annotations"])

                    vis_path = os.path.join(vis_dir, f"vis_task_{self.task_id}_{f_id}.jpg")
                    cv2.imwrite(vis_path, res["visual_image"])

                    self.log_signal.emit(f"Frame [{f_name}]: Detected {res['count']} vehicle(s)", "info")

                    self.image_processed_signal.emit({
                        "vis_path": vis_path,
                        "file_name": f_name,
                        "count": res["count"],
                        "current": i + 1,
                        "total": total
                    })

                # Assemble & Upload
                coco_data = self.segmenter.create_coco_dataset(images_info, all_annotations)
                json_path = os.path.join(self.output_dir, f"instances_task_{self.task_id}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(coco_data, f, indent=2)

                zip_path = os.path.join(self.output_dir, f"coco_task_{self.task_id}.zip")
                self.segmenter.package_coco_zip(json_path, zip_path)

                self.progress_signal.emit("Uploading refined COCO annotations to CVAT...", 85)
                ok, msg = self.cvat_mgr.upload_annotations_to_task(self.task_id, zip_path)

                if ok:
                    self.log_signal.emit(msg, "success")
                    self.progress_signal.emit("Complete!", 100)
                    self.finished_signal.emit(True, f"Annotated {total} frames with {len(all_annotations)} objects!")
                else:
                    self.finished_signal.emit(False, msg)

            # -------------------------------------------------------------
            # MODE 3: Local Images -> Offline Export
            # -------------------------------------------------------------
            elif self.mode == "offline":
                if not self.input_paths:
                    self.finished_signal.emit(False, "No input images selected")
                    return

                total = len(self.input_paths)
                for i, img_path in enumerate(self.input_paths):
                    if not self.is_running:
                        break

                    file_name = os.path.basename(img_path)
                    self.progress_signal.emit(f"Segmenting {file_name} ({i+1}/{total})...", int(((i+1)/total) * 90))

                    res = self.segmenter.process_image(img_path, image_id=i+1, file_name=file_name)
                    images_info.append(res["image_info"])
                    all_annotations.extend(res["annotations"])

                    vis_path = os.path.join(vis_dir, f"vis_{file_name}")
                    cv2.imwrite(vis_path, res["visual_image"])

                    self.image_processed_signal.emit({
                        "vis_path": vis_path,
                        "file_name": file_name,
                        "count": res["count"],
                        "current": i + 1,
                        "total": total
                    })

                coco_data = self.segmenter.create_coco_dataset(images_info, all_annotations)
                json_path = os.path.join(self.output_dir, "instances_default.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(coco_data, f, indent=2)

                zip_path = os.path.join(self.output_dir, "coco_annotations.zip")
                self.segmenter.package_coco_zip(json_path, zip_path)

                self.progress_signal.emit("Complete!", 100)
                self.finished_signal.emit(True, f"Offline COCO dataset saved to {self.output_dir}")

        except Exception as e:
            self.log_signal.emit(f"Error during execution: {str(e)}", "error")
            self.finished_signal.emit(False, str(e))

    def stop(self):
        self.is_running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚗 Car Segmentation & CVAT Auto-Annotator (YOLO + SAM)")
        self.resize(1300, 850)

        # Core Components
        self.cvat_mgr = CVATManager(host="http://1.10.131.222:8080")
        self.segmenter = CarSegmenter(
            yolo_model_name="yolov8n.pt",
            sam_model_name="mobile_sam.pt",
            conf_threshold=0.35,
            target_classes=["car", "truck", "bus", "motorcycle"]
        )

        self.input_image_paths = []
        self.preview_history = []
        self.current_preview_idx = 0
        self.worker: Optional[WorkerThread] = None

        self.init_ui()
        self.test_cvat_ping()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Horizontal Splitter between Left Control Panel and Right Preview
        splitter = QSplitter(Qt.Horizontal)

        # =================================================================
        # LEFT PANEL: Controls & Settings
        # =================================================================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # Tabs for Configuration
        tabs = QTabWidget()

        # ---------------- TAB 1: CVAT Server & Projects ----------------
        tab_cvat = QWidget()
        tab_cvat_layout = QVBoxLayout(tab_cvat)

        # Server Connection Group
        conn_group = QGroupBox("1. CVAT Connection (http://1.10.131.222:8080)")
        conn_layout = QVBoxLayout(conn_group)

        self.server_url_input = QLineEdit("http://1.10.131.222:8080")
        conn_layout.addWidget(QLabel("Server URL:"))
        conn_layout.addWidget(self.server_url_input)

        # Credentials inputs
        cred_layout = QHBoxLayout()
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        cred_layout.addWidget(self.username_input)
        cred_layout.addWidget(self.password_input)
        conn_layout.addLayout(cred_layout)

        # Token input
        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("Or API Token (Optional)")
        conn_layout.addWidget(self.token_input)

        # Connect button & status
        btn_conn_layout = QHBoxLayout()
        self.btn_connect = QPushButton("🔗 Connect to CVAT")
        self.btn_connect.clicked.connect(self.on_connect_cvat)
        self.lbl_conn_status = QLabel("⚪ Not connected")
        btn_conn_layout.addWidget(self.btn_connect)
        btn_conn_layout.addWidget(self.lbl_conn_status)
        conn_layout.addLayout(btn_conn_layout)

        tab_cvat_layout.addWidget(conn_group)

        # Project & Task Selection Group
        proj_group = QGroupBox("2. CVAT Project & Task Management")
        proj_layout = QVBoxLayout(proj_group)

        # Project Row
        proj_layout.addWidget(QLabel("Select Project:"))
        proj_row = QHBoxLayout()
        self.combo_projects = QComboBox()
        self.combo_projects.addItem("-- None (Standalone Tasks) --", 0)
        self.combo_projects.currentIndexChanged.connect(self.on_project_changed)
        self.btn_new_project = QPushButton("+ New Project")
        self.btn_new_project.setObjectName("secondaryBtn")
        self.btn_new_project.clicked.connect(self.on_create_project_dialog)
        proj_row.addWidget(self.combo_projects, 1)
        proj_row.addWidget(self.btn_new_project)
        proj_layout.addLayout(proj_row)

        # Task Row
        proj_layout.addWidget(QLabel("Select Existing Task:"))
        task_row = QHBoxLayout()
        self.combo_tasks = QComboBox()
        self.combo_tasks.addItem("-- Create New Task --", 0)
        self.btn_new_task = QPushButton("+ New Task")
        self.btn_new_task.setObjectName("secondaryBtn")
        self.btn_new_task.clicked.connect(self.on_create_task_dialog)
        self.btn_refresh = QPushButton("🔄 Refresh")
        self.btn_refresh.setObjectName("secondaryBtn")
        self.btn_refresh.clicked.connect(self.refresh_projects_and_tasks)
        task_row.addWidget(self.combo_tasks, 1)
        task_row.addWidget(self.btn_new_task)
        task_row.addWidget(self.btn_refresh)
        proj_layout.addLayout(task_row)

        # New Task Name Field
        proj_layout.addWidget(QLabel("New Task Name (if creating new):"))
        self.new_task_name_input = QLineEdit()
        self.new_task_name_input.setPlaceholderText("e.g. Car_Dataset_Batch_01")
        proj_layout.addWidget(self.new_task_name_input)

        tab_cvat_layout.addWidget(proj_group)
        tab_cvat_layout.addStretch()
        tabs.addTab(tab_cvat, "🌐 CVAT Server")

        # ---------------- TAB 2: Model Configuration ----------------
        tab_model = QWidget()
        tab_model_layout = QVBoxLayout(tab_model)

        model_group = QGroupBox("Model Size & Refinement Settings")
        m_layout = QVBoxLayout(model_group)

        # YOLO Model Selection
        m_layout.addWidget(QLabel("YOLO Model (Detection & Rough Localization):"))
        self.combo_yolo = QComboBox()
        self.combo_yolo.addItems([
            "yolov8n.pt (Nano - Light & Fast)",
            "yolov8s.pt (Small - Balanced)",
            "yolov8m.pt (Medium - High Accuracy)",
            "yolov8l.pt (Large - Maximum Recall)",
            "yolo11n.pt (YOLO11 Nano - Next-Gen Fast)",
            "yolo11s.pt (YOLO11 Small - Accurate)"
        ])
        m_layout.addWidget(self.combo_yolo)

        # SAM Model Selection
        m_layout.addWidget(QLabel("SAM Model (Boundary Refinement):"))
        self.combo_sam = QComboBox()
        self.combo_sam.addItems([
            "mobile_sam.pt (MobileSAM - ~39MB, Fast / CPU Ideal)",
            "sam2.1_t.pt (SAM 2 Tiny - Fast & Crisp)",
            "sam2_b.pt (SAM 2 Base - Very Crisp)",
            "sam_b.pt (SAM ViT-Base - Standard ~375MB)",
            "sam_l.pt (SAM ViT-Large - Maximum Precision)"
        ])
        m_layout.addWidget(self.combo_sam)

        # Vehicle Classes Filter
        m_layout.addWidget(QLabel("Vehicle Classes to Segment (Matching Set1 Schema):"))
        cls_grid = QVBoxLayout()
        cls_row1 = QHBoxLayout()
        self.chk_car = QCheckBox("Car (1)")
        self.chk_car.setChecked(True)
        self.chk_comm_pickup = QCheckBox("Commercial Pickup (2)")
        self.chk_comm_pickup.setChecked(True)
        self.chk_home_pickup = QCheckBox("Home Pickup (3)")
        self.chk_home_pickup.setChecked(True)
        self.chk_suv = QCheckBox("SUV (4)")
        self.chk_suv.setChecked(True)
        cls_row1.addWidget(self.chk_car)
        cls_row1.addWidget(self.chk_comm_pickup)
        cls_row1.addWidget(self.chk_home_pickup)
        cls_row1.addWidget(self.chk_suv)

        cls_row2 = QHBoxLayout()
        self.chk_truck = QCheckBox("Truck (5)")
        self.chk_truck.setChecked(True)
        self.chk_van = QCheckBox("Van (6)")
        self.chk_van.setChecked(True)
        self.chk_bus = QCheckBox("Bus (7)")
        self.chk_bus.setChecked(True)
        self.chk_moto = QCheckBox("Motorbike (8)")
        self.chk_moto.setChecked(True)
        cls_row2.addWidget(self.chk_truck)
        cls_row2.addWidget(self.chk_van)
        cls_row2.addWidget(self.chk_bus)
        cls_row2.addWidget(self.chk_moto)

        cls_grid.addLayout(cls_row1)
        cls_grid.addLayout(cls_row2)
        m_layout.addLayout(cls_grid)

        # Confidence Threshold Slider
        conf_row = QHBoxLayout()
        self.lbl_conf = QLabel("Confidence: 0.35")
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(10, 95)
        self.slider_conf.setValue(35)
        self.slider_conf.valueChanged.connect(lambda v: self.lbl_conf.setText(f"Confidence: {v/100:.2f}"))
        conf_row.addWidget(self.lbl_conf)
        conf_row.addWidget(self.slider_conf)
        m_layout.addLayout(conf_row)

        # Polygon Approx Tolerance
        poly_row = QHBoxLayout()
        self.lbl_poly = QLabel("Polygon Simplification: 1.0 px")
        self.slider_poly = QSlider(Qt.Horizontal)
        self.slider_poly.setRange(0, 30)
        self.slider_poly.setValue(10)
        self.slider_poly.valueChanged.connect(lambda v: self.lbl_poly.setText(f"Polygon Simplification: {v/10:.1f} px"))
        poly_row.addWidget(self.lbl_poly)
        poly_row.addWidget(self.slider_poly)
        m_layout.addLayout(poly_row)

        tab_model_layout.addWidget(model_group)
        tab_model_layout.addStretch()
        tabs.addTab(tab_model, "⚙️ Models & Classes")

        # ---------------- TAB 3: Data Source ----------------
        tab_data = QWidget()
        tab_data_layout = QVBoxLayout(tab_data)

        data_group = QGroupBox("Select Execution Mode & Data")
        d_layout = QVBoxLayout(data_group)

        self.radio_local_upload = QRadioButton("Local Images ➔ YOLO+SAM ➔ Upload to CVAT")
        self.radio_local_upload.setChecked(True)
        self.radio_cvat_task = QRadioButton("Annotate Existing CVAT Task Frames (Server Mode)")
        self.radio_offline = QRadioButton("Offline Mode (Local Images ➔ Export COCO JSON Only)")

        self.btn_group_mode = QButtonGroup()
        self.btn_group_mode.addButton(self.radio_local_upload)
        self.btn_group_mode.addButton(self.radio_cvat_task)
        self.btn_group_mode.addButton(self.radio_offline)

        d_layout.addWidget(self.radio_local_upload)
        d_layout.addWidget(self.radio_cvat_task)
        d_layout.addWidget(self.radio_offline)

        # File / Folder Picker
        d_layout.addWidget(QLabel("Select Local Image Folder or Files:"))
        folder_row = QHBoxLayout()
        self.btn_browse_folder = QPushButton("📁 Select Folder")
        self.btn_browse_folder.setObjectName("secondaryBtn")
        self.btn_browse_folder.clicked.connect(self.on_browse_folder)
        self.btn_browse_files = QPushButton("🖼️ Select Files")
        self.btn_browse_files.setObjectName("secondaryBtn")
        self.btn_browse_files.clicked.connect(self.on_browse_files)
        folder_row.addWidget(self.btn_browse_folder)
        folder_row.addWidget(self.btn_browse_files)
        d_layout.addLayout(folder_row)

        self.lbl_selected_images = QLabel("Selected: 0 images")
        d_layout.addWidget(self.lbl_selected_images)

        # Output Dir
        d_layout.addWidget(QLabel("Local Output Directory:"))
        out_row = QHBoxLayout()
        self.output_dir_input = QLineEdit(os.path.abspath("./output_coco"))
        self.btn_browse_out = QPushButton("Browse...")
        self.btn_browse_out.setObjectName("secondaryBtn")
        self.btn_browse_out.clicked.connect(self.on_browse_output)
        out_row.addWidget(self.output_dir_input, 1)
        out_row.addWidget(self.btn_browse_out)
        d_layout.addLayout(out_row)

        tab_data_layout.addWidget(data_group)
        tab_data_layout.addStretch()
        tabs.addTab(tab_data, "📁 Data & Output")

        left_layout.addWidget(tabs)

        # Big Action Button
        self.btn_start = QPushButton("🚀 Start Car Segmentation & Upload")
        self.btn_start.setObjectName("actionBtn")
        self.btn_start.clicked.connect(self.on_start_processing)
        left_layout.addWidget(self.btn_start)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        self.lbl_progress_text = QLabel("Ready")
        self.lbl_progress_text.setStyleSheet("color: #a6adc8; font-size: 11px;")
        left_layout.addWidget(self.lbl_progress_text)

        splitter.addWidget(left_panel)

        # =================================================================
        # RIGHT PANEL: Preview & Logs
        # =================================================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Image Preview Card
        preview_group = QGroupBox("Visual Preview (Crisp SAM Masks & Bounding Boxes)")
        prev_layout = QVBoxLayout(preview_group)

        self.lbl_preview = QLabel("Process an image or select sample to preview...")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setStyleSheet("background-color: #11111b; border: 1px dashed #45475a; border-radius: 6px;")
        self.lbl_preview.setMinimumSize(500, 380)
        prev_layout.addWidget(self.lbl_preview, 1)

        # Preview Navigation
        nav_row = QHBoxLayout()
        self.btn_prev_img = QPushButton("◀ Prev")
        self.btn_prev_img.setObjectName("secondaryBtn")
        self.btn_prev_img.clicked.connect(self.on_prev_preview)
        self.lbl_preview_info = QLabel("0 / 0")
        self.btn_next_img = QPushButton("Next ▶")
        self.btn_next_img.setObjectName("secondaryBtn")
        self.btn_next_img.clicked.connect(self.on_next_preview)
        nav_row.addWidget(self.btn_prev_img)
        nav_row.addWidget(self.lbl_preview_info)
        nav_row.addWidget(self.btn_next_img)
        prev_layout.addLayout(nav_row)

        right_layout.addWidget(preview_group, 2)

        # Log Console Card
        log_group = QGroupBox("Activity & Detection Logs")
        log_layout = QVBoxLayout(log_group)
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        log_layout.addWidget(self.log_console)

        right_layout.addWidget(log_group, 1)

        splitter.addWidget(right_panel)
        splitter.setSizes([550, 750])

        main_layout.addWidget(splitter)
        self.setStyleSheet(MODERN_STYLE)

    def log(self, text: str, level: str = "info"):
        timestamp = time.strftime("%H:%M:%S")
        color = "#cdd6f4"
        if level == "success":
            color = "#a6e3a1"
        elif level == "warning":
            color = "#f9e2af"
        elif level == "error":
            color = "#f38ba8"
        elif level == "highlight":
            color = "#89b4fa"

        html = f"<span style='color: #6c7086;'>[{timestamp}]</span> <span style='color: {color};'>{text}</span>"
        self.log_console.append(html)

    def test_cvat_ping(self):
        ok, msg = self.cvat_mgr.test_connection()
        if ok:
            self.lbl_conn_status.setText("🟢 Server Reachable (Ready to Login)")
            self.lbl_conn_status.setStyleSheet("color: #a6e3a1;")
            self.log(msg, "success")
        else:
            self.lbl_conn_status.setText("🔴 Server Unreachable")
            self.lbl_conn_status.setStyleSheet("color: #f38ba8;")
            self.log(msg, "warning")

    def on_connect_cvat(self):
        url = self.server_url_input.text().strip()
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()
        token = self.token_input.text().strip()

        self.cvat_mgr.host = url
        self.lbl_conn_status.setText("⏳ Authenticating...")
        QApplication.processEvents()

        ok, msg = self.cvat_mgr.login(username=username, password=password, token=token)
        if ok:
            self.lbl_conn_status.setText("🟢 Authenticated & Ready")
            self.lbl_conn_status.setStyleSheet("color: #a6e3a1; font-weight: bold;")
            self.log(msg, "success")
            self.refresh_projects_and_tasks()
        else:
            self.lbl_conn_status.setText("🔴 Login Failed")
            self.lbl_conn_status.setStyleSheet("color: #f38ba8;")
            self.log(msg, "error")
            QMessageBox.warning(self, "CVAT Authentication Failed", msg)

    def refresh_projects_and_tasks(self):
        if not self.cvat_mgr.is_connected:
            self.log("Cannot refresh: Not logged in to CVAT", "warning")
            return

        try:
            self.combo_projects.clear()
            self.combo_projects.addItem("-- None (Standalone Tasks) --", 0)

            projects = self.cvat_mgr.list_projects()
            for p in projects:
                self.combo_projects.addItem(f"📁 #{p['id']}: {p['name']}", p['id'])

            self.log(f"Loaded {len(projects)} project(s) from CVAT", "info")
            self.refresh_tasks()
        except Exception as e:
            self.log(f"Error loading projects: {e}", "error")

    def on_project_changed(self, idx):
        self.refresh_tasks()

    def refresh_tasks(self):
        if not self.cvat_mgr.is_connected:
            return

        try:
            self.combo_tasks.clear()
            self.combo_tasks.addItem("-- Create New Task --", 0)

            project_id = self.combo_projects.currentData()
            proj_filter = project_id if project_id and project_id > 0 else None

            tasks = self.cvat_mgr.list_tasks(project_id=proj_filter)
            for t in tasks:
                self.combo_tasks.addItem(f"📋 #{t['id']}: {t['name']} ({t['size']} frames)", t['id'])

            self.log(f"Loaded {len(tasks)} task(s)", "info")
        except Exception as e:
            self.log(f"Error loading tasks: {e}", "error")

    def on_create_project_dialog(self):
        if not self.cvat_mgr.is_connected:
            QMessageBox.warning(self, "Not Connected", "Please connect and log in to CVAT first.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Create New CVAT Project")
        dialog.resize(360, 160)
        d_layout = QFormLayout(dialog)

        name_input = QLineEdit()
        name_input.setPlaceholderText("e.g. Bangkok_Traffic_Cars")
        d_layout.addRow("Project Name:", name_input)

        lbl_desc = QLabel("Labels: car, commercial_pickup, home_pickup, SUV, truck, van, bus, motorbike")
        lbl_desc.setStyleSheet("color: #a6adc8; font-size: 11px;")
        d_layout.addRow(lbl_desc)

        btn_box = QHBoxLayout()
        btn_ok = QPushButton("Create Project")
        btn_ok.clicked.connect(dialog.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setObjectName("secondaryBtn")
        btn_cancel.clicked.connect(dialog.reject)
        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        d_layout.addRow(btn_box)

        if dialog.exec_() == QDialog.Accepted:
            p_name = name_input.text().strip()
            if not p_name:
                QMessageBox.warning(self, "Invalid Name", "Project name cannot be empty")
                return

            try:
                created = self.cvat_mgr.create_project(name=p_name)
                self.log(f"Created project #{created['id']}: {created['name']}", "success")
                self.refresh_projects_and_tasks()
                # Select the created project
                for i in range(self.combo_projects.count()):
                    if self.combo_projects.itemData(i) == created['id']:
                        self.combo_projects.setCurrentIndex(i)
                        break
            except Exception as e:
                self.log(f"Failed to create project: {e}", "error")
                QMessageBox.critical(self, "Error", str(e))

    def on_create_task_dialog(self):
        if not self.cvat_mgr.is_connected:
            QMessageBox.warning(self, "Not Connected", "Please connect and log in to CVAT first.")
            return

        t_name = self.new_task_name_input.text().strip()
        if not t_name:
            t_name = f"Car_Seg_{time.strftime('%Y%m%d_%H%M%S')}"
            self.new_task_name_input.setText(t_name)

        self.log(f"Ready to create task '{t_name}' upon starting segmentation.", "info")

    def on_browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Images")
        if folder:
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            self.input_image_paths = [
                os.path.join(folder, f) for f in os.listdir(folder)
                if Path(f).suffix.lower() in valid_exts
            ]
            self.lbl_selected_images.setText(f"Selected: {len(self.input_image_paths)} images from {folder}")
            self.log(f"Selected {len(self.input_image_paths)} images from {folder}", "info")

    def on_browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if files:
            self.input_image_paths = files
            self.lbl_selected_images.setText(f"Selected: {len(self.input_image_paths)} files")
            self.log(f"Selected {len(self.input_image_paths)} files", "info")

    def on_browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_dir_input.setText(folder)

    def on_start_processing(self):
        # 1. Gather configuration
        yolo_choice = self.combo_yolo.currentText().split()[0]
        sam_choice = self.combo_sam.currentText().split()[0]
        conf_thresh = self.slider_conf.value() / 100.0
        poly_tolerance = self.slider_poly.value() / 10.0

        target_classes = []
        if self.chk_car.isChecked():
            target_classes.append("car")
        if self.chk_comm_pickup.isChecked():
            target_classes.append("commercial_pickup")
        if self.chk_home_pickup.isChecked():
            target_classes.append("home_pickup")
        if self.chk_suv.isChecked():
            target_classes.append("suv")
        if self.chk_truck.isChecked():
            target_classes.append("truck")
        if self.chk_van.isChecked():
            target_classes.append("van")
        if self.chk_bus.isChecked():
            target_classes.append("bus")
        if self.chk_moto.isChecked():
            target_classes.append("motorbike")
            target_classes.append("motorcycle")

        if not target_classes:
            QMessageBox.warning(self, "No Classes", "Please select at least one vehicle class.")
            return

        self.segmenter.set_models(yolo_choice, sam_choice)
        self.segmenter.set_target_classes(target_classes)
        self.segmenter.conf_threshold = conf_thresh
        self.segmenter.approx_tolerance = poly_tolerance

        # 2. Determine mode
        if self.radio_local_upload.isChecked():
            mode = "local_upload"
        elif self.radio_cvat_task.isChecked():
            mode = "cvat_task"
        else:
            mode = "offline"

        project_id = self.combo_projects.currentData() or 0
        task_id = self.combo_tasks.currentData() or 0
        task_name = self.new_task_name_input.text().strip()
        output_dir = self.output_dir_input.text().strip()

        # Validation
        if mode in ("local_upload", "offline") and not self.input_paths_valid():
            return
        if mode == "cvat_task" and task_id == 0:
            QMessageBox.warning(self, "No Task Selected", "Please select an existing CVAT Task from the dropdown.")
            return

        # 3. Start background thread
        self.btn_start.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_progress_text.setText("Starting pipeline...")
        self.preview_history = []
        self.current_preview_idx = 0

        self.worker = WorkerThread(
            segmenter=self.segmenter,
            cvat_mgr=self.cvat_mgr,
            mode=mode,
            input_paths=self.input_image_paths,
            project_id=project_id,
            task_id=task_id,
            task_name=task_name,
            output_dir=output_dir
        )
        self.worker.progress_signal.connect(self.on_worker_progress)
        self.worker.image_processed_signal.connect(self.on_worker_image_processed)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_worker_finished)
        self.worker.start()

    def input_paths_valid(self) -> bool:
        if not self.input_image_paths:
            # Check if bus.jpg exists in workspace as sample
            sample_path = os.path.abspath("bus.jpg")
            if os.path.exists(sample_path):
                self.input_image_paths = [sample_path]
                self.lbl_selected_images.setText("Selected: 1 image (bus.jpg sample)")
                self.log("Auto-selected sample 'bus.jpg'", "info")
                return True
            else:
                QMessageBox.warning(self, "No Images", "Please select images or a folder to process.")
                return False
        return True

    def on_worker_progress(self, text: str, percent: int):
        self.progress_bar.setValue(percent)
        self.lbl_progress_text.setText(text)

    def on_worker_image_processed(self, data: dict):
        self.preview_history.append(data)
        self.current_preview_idx = len(self.preview_history) - 1
        self.display_preview(self.current_preview_idx)

    def display_preview(self, idx: int):
        if not self.preview_history or idx < 0 or idx >= len(self.preview_history):
            return

        item = self.preview_history[idx]
        vis_path = item["vis_path"]
        if os.path.exists(vis_path):
            pixmap = QPixmap(vis_path)
            scaled = pixmap.scaled(
                self.lbl_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lbl_preview.setPixmap(scaled)
            self.lbl_preview_info.setText(
                f"{idx + 1} / {len(self.preview_history)} | {item['file_name']} ({item['count']} vehicles)"
            )

    def on_prev_preview(self):
        if self.current_preview_idx > 0:
            self.current_preview_idx -= 1
            self.display_preview(self.current_preview_idx)

    def on_next_preview(self):
        if self.current_preview_idx < len(self.preview_history) - 1:
            self.current_preview_idx += 1
            self.display_preview(self.current_preview_idx)

    def on_worker_finished(self, success: bool, message: str):
        self.btn_start.setEnabled(True)
        if success:
            self.log(f"🎉 Pipeline Completed: {message}", "success")
            QMessageBox.information(self, "Done", message)
            if self.cvat_mgr.is_connected:
                self.refresh_projects_and_tasks()
        else:
            self.log(f"❌ Execution failed: {message}", "error")
            QMessageBox.critical(self, "Execution Failed", message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
