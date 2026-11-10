import os
import sys
import time
import json
import threading
import cv2
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, send_file
from core.pipeline import CarSegmenter, DEFAULT_CATEGORIES, DEFAULT_VEHICLE_CLASSES
from core.cvat_manager import CVATManager

app = Flask(__name__)

# State
cvat_mgr = CVATManager(host="http://1.10.131.222:8080")
segmenter = CarSegmenter(
    yolo_model_name="yolov8n.pt",
    sam_model_name="mobile_sam.pt",
    conf_threshold=0.35,
    target_classes=["car", "motorbike"]
)

job_state = {
    "is_running": False,
    "progress": 0,
    "status_text": "Ready",
    "logs": [],
    "last_vis_path": None,
    "total_detected": 0
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚗 Car Segmentation & CVAT Auto-Annotator (YOLO + SAM)</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #12131c; color: #cdd6f4; font-family: 'Segoe UI', system-ui, sans-serif; }
        .card { background-color: #1e1f2e; border: 1px solid #2e3048; border-radius: 12px; margin-bottom: 20px; }
        .card-header { background-color: #26283d; border-bottom: 1px solid #363854; font-weight: bold; color: #89b4fa; }
        .form-control, .form-select { background-color: #141522; border: 1px solid #363854; color: #cdd6f4; }
        .form-control:focus, .form-select:focus { background-color: #161726; border-color: #89b4fa; color: #fff; box-shadow: none; }
        .btn-primary { background-color: #89b4fa; border: none; color: #11111b; font-weight: bold; }
        .btn-primary:hover { background-color: #b4befe; color: #11111b; }
        .btn-success { background-color: #a6e3a1; border: none; color: #11111b; font-weight: bold; }
        .btn-success:hover { background-color: #94e2d5; color: #11111b; }
        .log-box { background-color: #0b0c13; border: 1px solid #2e3048; border-radius: 8px; height: 260px; overflow-y: auto; padding: 10px; font-family: monospace; font-size: 12px; }
        .badge-status { font-size: 0.85rem; padding: 6px 12px; border-radius: 20px; }
        .preview-img { max-height: 420px; width: auto; max-width: 100%; border-radius: 8px; border: 1px solid #363854; }
    </style>
</head>
<body class="p-4">
    <div class="container-fluid">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <div>
                <h3 class="mb-0">🚗 Car Segmentation & CVAT Auto-Annotator</h3>
                <small class="text-secondary">YOLO Bounding Boxes ➔ SAM Crisp Edge Refinement ➔ COCO 1.0 ➔ CVAT Server</small>
            </div>
            <div id="serverBadge" class="badge-status bg-secondary">Checking Server...</div>
        </div>

        <div class="row">
            <!-- Left Panel -->
            <div class="col-lg-5">
                <!-- CVAT Connection Card -->
                <div class="card shadow-sm">
                    <div class="card-header">1. CVAT Server Connection</div>
                    <div class="card-body">
                        <div class="mb-2">
                            <label class="form-label small">Server URL</label>
                            <input type="text" id="cvatUrl" class="form-control" value="http://1.10.131.222:8080">
                        </div>
                        <div class="row mb-2">
                            <div class="col">
                                <input type="text" id="cvatUser" class="form-control" placeholder="Username">
                            </div>
                            <div class="col">
                                <input type="password" id="cvatPass" class="form-control" placeholder="Password">
                            </div>
                        </div>
                        <div class="mb-3">
                            <input type="text" id="cvatToken" class="form-control" placeholder="Or API Token (Optional)">
                        </div>
                        <button class="btn btn-primary w-100" onclick="connectCVAT()">🔗 Connect to CVAT</button>
                    </div>
                </div>

                <!-- Projects and Tasks Card -->
                <div class="card shadow-sm">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <span>2. Project & Task Selection</span>
                        <button class="btn btn-sm btn-outline-light" onclick="loadProjectsAndTasks()">🔄 Refresh</button>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label class="form-label small">Project</label>
                            <div class="input-group">
                                <select id="projectSelect" class="form-select" onchange="filterTasks()">
                                    <option value="0">-- None (Standalone Tasks) --</option>
                                </select>
                                <button class="btn btn-outline-secondary" onclick="createProjectPrompt()">+ New</button>
                            </div>
                        </div>
                        <div class="mb-3">
                            <label class="form-label small">Task</label>
                            <select id="taskSelect" class="form-select">
                                <option value="0">-- Create New Task --</option>
                            </select>
                        </div>
                        <div class="mb-2">
                            <label class="form-label small">New Task Name (if creating)</label>
                            <input type="text" id="newTaskName" class="form-control" placeholder="Car_Segmentation_Batch_01">
                        </div>
                    </div>
                </div>

                <!-- Models & Settings Card -->
                <div class="card shadow-sm">
                    <div class="card-header">3. Model Configuration</div>
                    <div class="card-body">
                        <div class="mb-2">
                            <label class="form-label small">YOLO Model (Size: Nano, Small, Medium, Large, XLarge)</label>
                            <select id="yoloModel" class="form-select">
                                <option value="yolo11x.pt">yolo11x.pt (YOLO11 XLarge - Maximum Accuracy)</option>
                                <option value="yolo11l.pt">yolo11l.pt (YOLO11 Large - High Accuracy)</option>
                                <option value="yolo11m.pt">yolo11m.pt (YOLO11 Medium - Balanced)</option>
                                <option value="yolo11s.pt">yolo11s.pt (YOLO11 Small)</option>
                                <option value="yolo11n.pt">yolo11n.pt (YOLO11 Nano - Next Gen)</option>
                                <option value="yolov8x.pt">yolov8x.pt (YOLOv8 XLarge - Maximum Accuracy)</option>
                                <option value="yolov8l.pt">yolov8l.pt (YOLOv8 Large - High Accuracy)</option>
                                <option value="yolov8m.pt">yolov8m.pt (YOLOv8 Medium - High Accuracy)</option>
                                <option value="yolov8s.pt">yolov8s.pt (YOLOv8 Small - Balanced)</option>
                                <option value="yolov8n.pt">yolov8n.pt (YOLOv8 Nano - Light & Fast)</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label small">SAM Model (Size: Mobile, Tiny, Small, Base, Large)</label>
                            <select id="samModel" class="form-select">
                                <option value="sam_l.pt">sam_l.pt (SAM ViT-Large - Maximum Precision, ~2.5GB)</option>
                                <option value="sam_b.pt">sam_b.pt (SAM ViT-Base - Standard High Precision, ~375MB)</option>
                                <option value="sam2.1_l.pt">sam2.1_l.pt (SAM 2.1 Large - State-of-the-Art Crisp)</option>
                                <option value="sam2.1_b.pt">sam2.1_b.pt (SAM 2.1 Base - Very Crisp)</option>
                                <option value="sam2.1_s.pt">sam2.1_s.pt (SAM 2.1 Small - Balanced)</option>
                                <option value="sam2.1_t.pt">sam2.1_t.pt (SAM 2.1 Tiny - Fast & Crisp)</option>
                                <option value="mobile_sam.pt">mobile_sam.pt (~39MB - Fast / CPU Friendly)</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label small">Target Classes (2 Major Classes)</label>
                            <div class="d-flex gap-3">
                                <div class="form-check"><input class="form-check-input" type="checkbox" id="clsCar" checked><label class="form-check-label">1. Car (รวมรถยนต์, บรรทุก, บัส, กระบะ, ตู้)</label></div>
                                <div class="form-check"><input class="form-check-input" type="checkbox" id="clsMoto" checked><label class="form-check-label">2. Motorbike (มอเตอร์ไซค์)</label></div>
                            </div>
                        </div>
                        <div class="mb-3">
                            <label class="form-label small">Confidence Threshold: <span id="confVal">0.35</span></label>
                            <input type="range" class="form-range" id="confSlider" min="10" max="95" value="35" oninput="document.getElementById('confVal').innerText = (this.value/100).toFixed(2)">
                        </div>
                        <div class="mb-3">
                            <label class="form-label small">Local Image Folder</label>
                            <input type="text" id="imageFolder" class="form-control" value="." placeholder="Folder path containing car images">
                        </div>
                        <button class="btn btn-success w-100 py-2 fs-6" id="btnStart" onclick="startSegmentation()">
                            🚀 Start Car Segmentation & Upload to CVAT
                        </button>
                    </div>
                </div>
            </div>

            <!-- Right Panel -->
            <div class="col-lg-7">
                <!-- Preview Card -->
                <div class="card shadow-sm">
                    <div class="card-header">Visual Preview (Original + YOLO Bbox + Refined SAM Mask)</div>
                    <div class="card-body text-center p-3">
                        <img id="previewImg" src="/api/preview_image" class="preview-img" alt="Preview Image" onerror="this.style.display='none'">
                        <div id="noPreviewText" class="text-secondary py-5">No image processed yet</div>
                    </div>
                </div>

                <!-- Progress & Logs -->
                <div class="card shadow-sm">
                    <div class="card-header d-flex justify-content-between">
                        <span>Process Status & Activity Log</span>
                        <span id="statusText" class="text-info">Ready</span>
                    </div>
                    <div class="card-body">
                        <div class="progress mb-3" style="height: 18px;">
                            <div id="progressBar" class="progress-bar progress-bar-striped progress-bar-animated bg-info" role="progressbar" style="width: 0%">0%</div>
                        </div>
                        <div id="logConsole" class="log-box"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function checkServer() {
            try {
                let res = await fetch('/api/test_cvat');
                let data = await res.json();
                let b = document.getElementById('serverBadge');
                if (data.ok) {
                    b.className = 'badge-status bg-success';
                    b.innerText = data.msg;
                } else {
                    b.className = 'badge-status bg-danger';
                    b.innerText = 'CVAT Unreachable';
                }
            } catch(e) {
                console.error(e);
            }
        }

        async function connectCVAT() {
            let url = document.getElementById('cvatUrl').value;
            let user = document.getElementById('cvatUser').value;
            let pass = document.getElementById('cvatPass').value;
            let token = document.getElementById('cvatToken').value;

            let res = await fetch('/api/login_cvat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({url, user, pass, token})
            });
            let data = await res.json();
            alert(data.msg);
            if (data.ok) {
                loadProjectsAndTasks();
            }
        }

        async function loadProjectsAndTasks() {
            let resP = await fetch('/api/projects');
            let dataP = await resP.json();
            let pSelect = document.getElementById('projectSelect');
            pSelect.innerHTML = '<option value="0">-- None (Standalone Tasks) --</option>';
            if (dataP.projects) {
                dataP.projects.forEach(p => {
                    pSelect.innerHTML += `<option value="${p.id}">📁 #${p.id}: ${p.name}</option>`;
                });
            }

            let resT = await fetch('/api/tasks');
            let dataT = await resT.json();
            let tSelect = document.getElementById('taskSelect');
            tSelect.innerHTML = '<option value="0">-- Create New Task --</option>';
            if (dataT.tasks) {
                dataT.tasks.forEach(t => {
                    tSelect.innerHTML += `<option value="${t.id}">📋 #${t.id}: ${t.name} (${t.size} frames)</option>`;
                });
            }
        }

        async function filterTasks() {
            let pId = document.getElementById('projectSelect').value;
            let resT = await fetch(`/api/tasks?project_id=${pId}`);
            let dataT = await resT.json();
            let tSelect = document.getElementById('taskSelect');
            tSelect.innerHTML = '<option value="0">-- Create New Task --</option>';
            if (dataT.tasks) {
                dataT.tasks.forEach(t => {
                    tSelect.innerHTML += `<option value="${t.id}">📋 #${t.id}: ${t.name} (${t.size} frames)</option>`;
                });
            }
        }

        async function createProjectPrompt() {
            let name = prompt("Enter new CVAT Project Name:");
            if (name) {
                let res = await fetch('/api/create_project', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name})
                });
                let data = await res.json();
                alert(data.msg);
                loadProjectsAndTasks();
            }
        }

        async function startSegmentation() {
            let yolo = document.getElementById('yoloModel').value;
            let sam = document.getElementById('samModel').value;
            let conf = document.getElementById('confSlider').value / 100.0;
            let folder = document.getElementById('imageFolder').value;
            let project_id = document.getElementById('projectSelect').value;
            let task_id = document.getElementById('taskSelect').value;
            let task_name = document.getElementById('newTaskName').value;

            let classes = [];
            if (document.getElementById('clsCar').checked) classes.push("car");
            if (document.getElementById('clsMoto').checked) classes.push("motorbike");

            let res = await fetch('/api/start_segmentation', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    yolo, sam, conf, folder, project_id, task_id, task_name, classes
                })
            });
            let data = await res.json();
            if (data.ok) {
                document.getElementById('btnStart').disabled = true;
                pollStatus();
            } else {
                alert(data.msg);
            }
        }

        function pollStatus() {
            let interval = setInterval(async () => {
                let res = await fetch('/api/status');
                let data = await res.json();

                document.getElementById('progressBar').style.width = data.progress + '%';
                document.getElementById('progressBar').innerText = data.progress + '%';
                document.getElementById('statusText').innerText = data.status_text;

                let consoleDiv = document.getElementById('logConsole');
                consoleDiv.innerHTML = data.logs.map(l => `<div>${l}</div>`).join('');
                consoleDiv.scrollTop = consoleDiv.scrollHeight;

                if (data.has_image) {
                    let img = document.getElementById('previewImg');
                    img.src = '/api/preview_image?' + new Date().getTime();
                    img.style.display = 'block';
                    document.getElementById('noPreviewText').style.display = 'none';
                }

                if (!data.is_running) {
                    clearInterval(interval);
                    document.getElementById('btnStart').disabled = false;
                }
            }, 1000);
        }

        window.onload = () => {
            checkServer();
        };
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/test_cvat")
def api_test_cvat():
    ok, msg = cvat_mgr.test_connection()
    return jsonify({"ok": ok, "msg": msg})

@app.route("/api/login_cvat", methods=["POST"])
def api_login_cvat():
    data = request.json or {}
    url = data.get("url", "http://1.10.131.222:8080")
    user = data.get("user")
    pwd = data.get("pass")
    token = data.get("token")

    cvat_mgr.host = url
    ok, msg = cvat_mgr.login(username=user, password=pwd, token=token)
    return jsonify({"ok": ok, "msg": msg})

@app.route("/api/projects")
def api_projects():
    if not cvat_mgr.is_connected:
        return jsonify({"projects": []})
    return jsonify({"projects": cvat_mgr.list_projects()})

@app.route("/api/tasks")
def api_tasks():
    if not cvat_mgr.is_connected:
        return jsonify({"tasks": []})
    p_id = request.args.get("project_id", type=int)
    p_filter = p_id if p_id and p_id > 0 else None
    return jsonify({"tasks": cvat_mgr.list_tasks(project_id=p_filter)})

@app.route("/api/create_project", methods=["POST"])
def api_create_project():
    if not cvat_mgr.is_connected:
        return jsonify({"ok": False, "msg": "Not connected to CVAT"})
    name = (request.json or {}).get("name")
    if not name:
        return jsonify({"ok": False, "msg": "Project name cannot be empty"})
    try:
        proj = cvat_mgr.create_project(name=name)
        return jsonify({"ok": True, "msg": f"Created Project #{proj['id']}: {proj['name']}"})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})

@app.route("/api/start_segmentation", methods=["POST"])
def api_start_segmentation():
    if job_state["is_running"]:
        return jsonify({"ok": False, "msg": "A segmentation job is already running"})

    req_data = request.json or {}
    yolo_model = req_data.get("yolo", "yolov8n.pt")
    sam_model = req_data.get("sam", "mobile_sam.pt")
    conf = float(req_data.get("conf", 0.35))
    folder = req_data.get("folder", ".")
    project_id = int(req_data.get("project_id", 0))
    task_id = int(req_data.get("task_id", 0))
    task_name = req_data.get("task_name", f"Car_Seg_{time.strftime('%Y%m%d_%H%M%S')}")
    classes = req_data.get("classes", ["car", "motorbike"])

    segmenter.set_models(yolo_model, sam_model)
    segmenter.set_target_classes(classes)
    segmenter.conf_threshold = conf

    # Resolve image files
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    if os.path.isdir(folder):
        images = [os.path.join(folder, f) for f in os.listdir(folder) if Path(f).suffix.lower() in valid_exts]

    if not images:
        sample = os.path.abspath("bus.jpg")
        if os.path.exists(sample):
            images = [sample]

    if not images:
        return jsonify({"ok": False, "msg": "No images found in the specified folder"})

    def run_worker():
        job_state["is_running"] = True
        job_state["progress"] = 0
        job_state["logs"] = [f"🚀 Starting Car Segmentation on {len(images)} images..."]

        out_dir = os.path.abspath("./output_coco")
        vis_dir = os.path.join(out_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        images_info = []
        all_anns = []
        local_imgs = []

        total = len(images)
        for i, img_path in enumerate(images):
            fname = os.path.basename(img_path)
            job_state["progress"] = int(((i+1)/total) * 60)
            job_state["status_text"] = f"Processing {fname} ({i+1}/{total})..."

            res = segmenter.process_image(img_path, image_id=i+1, file_name=fname)
            images_info.append(res["image_info"])
            all_anns.extend(res["annotations"])
            local_imgs.append(img_path)

            vis_path = os.path.join(vis_dir, f"vis_{fname}")
            cv2.imwrite(vis_path, res["visual_image"])
            job_state["last_vis_path"] = vis_path
            job_state["logs"].append(f"[{fname}] Detected {res['count']} vehicle(s)")

        # Save COCO
        coco_data = segmenter.create_coco_dataset(images_info, all_anns)
        json_path = os.path.join(out_dir, "instances_default.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2)

        zip_path = os.path.join(out_dir, "coco_annotations.zip")
        segmenter.package_coco_zip(json_path, zip_path)
        job_state["logs"].append(f"💾 COCO Dataset saved locally: {len(all_anns)} objects")

        # Upload to CVAT if connected
        if cvat_mgr.is_connected:
            job_state["progress"] = 80
            job_state["status_text"] = "Uploading to CVAT..."
            job_state["logs"].append("Uploading to CVAT server...")

            if task_id > 0:
                ok, msg = cvat_mgr.upload_annotations_to_task(task_id, zip_path)
                job_state["logs"].append(msg)
            else:
                created = cvat_mgr.create_task_with_data(
                    name=task_name,
                    image_paths=local_imgs,
                    project_id=project_id if project_id > 0 else None,
                    annotation_zip_path=zip_path
                )
                job_state["logs"].append(f"CVAT Task #{created['id']} '{created['name']}' created!")

        job_state["progress"] = 100
        job_state["status_text"] = "Completed!"
        job_state["logs"].append(f"🎉 Completed! Processed {total} images with {len(all_anns)} vehicles.")
        job_state["is_running"] = False

    t = threading.Thread(target=run_worker, daemon=True)
    t.start()

    return jsonify({"ok": True, "msg": "Job started"})

@app.route("/api/status")
def api_status():
    return jsonify({
        "is_running": job_state["is_running"],
        "progress": job_state["progress"],
        "status_text": job_state["status_text"],
        "logs": job_state["logs"],
        "has_image": bool(job_state["last_vis_path"] and os.path.exists(job_state["last_vis_path"]))
    })

@app.route("/api/preview_image")
def api_preview_image():
    if job_state["last_vis_path"] and os.path.exists(job_state["last_vis_path"]):
        return send_file(job_state["last_vis_path"], mimetype="image/jpeg")
    return send_file("bus.jpg", mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
