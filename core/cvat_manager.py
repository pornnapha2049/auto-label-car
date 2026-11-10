import os
import io
import time
import requests
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable, Any
from cvat_sdk import Client
from cvat_sdk.api_client import models
from cvat_sdk.core.helpers import get_paginated_collection

class CVATManager:
    """
    Manages connection and operations with CVAT Server (http://1.10.131.222:8080)
    Supports:
    - Login via Username/Password or API Token
    - Projects listing and creation
    - Tasks listing and creation
    - Data & Annotation upload (COCO 1.0)
    - Frame extraction for existing CVAT tasks
    """

    def __init__(self, host: str = "http://1.10.131.222:8080"):
        self.host = host.rstrip("/")
        self.client: Optional[Client] = None
        self.is_connected: bool = False
        self.user_info: Dict[str, Any] = {}

    def test_connection(self) -> Tuple[bool, str]:
        """Test if CVAT server is reachable"""
        try:
            resp = requests.get(f"{self.host}/api/server/about", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                version = data.get("version", "Unknown")
                return True, f"CVAT Server is reachable (Version {version})"
            return False, f"Server responded with status code: {resp.status_code}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def login(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Authenticate with CVAT using credentials or token"""
        try:
            self.client = Client(url=self.host, check_server_version=False)

            if token and token.strip():
                # Token Authentication
                token_clean = token.strip()
                self.client.api_client.configuration.api_key['tokenAuth'] = token_clean
                self.client.api_client.set_default_header("Authorization", f"Token {token_clean}")
                self.is_connected = True
                return True, "Authenticated successfully via Token"

            elif username and password:
                # Session Authentication
                self.client.login((username.strip(), password))
                self.is_connected = True
                return True, f"Logged in successfully as '{username}'"

            else:
                return False, "Please provide either Username/Password or Token"

        except Exception as e:
            self.is_connected = False
            self.client = None
            err_msg = str(e)
            if "401" in err_msg or "credential" in err_msg.lower():
                return False, "Authentication failed: Invalid username, password, or token"
            return False, f"Login error: {err_msg}"

    def list_projects(self) -> List[Dict]:
        """Fetch list of all projects"""
        if not self.is_connected or not self.client:
            raise ConnectionError("Not connected to CVAT server")

        projects = []
        try:
            for p in self.client.projects.list():
                labels = [l.name for l in p.get_labels()]
                projects.append({
                    "id": p.id,
                    "name": p.name,
                    "labels": labels,
                    "created_date": str(getattr(p, "created_date", ""))
                })
        except Exception as e:
            print(f"[CVAT] Error listing projects: {e}")
        return projects

    def create_project(
        self,
        name: str,
        labels: Optional[List[str]] = None
    ) -> Dict:
        """Create a new CVAT project with vehicle labels"""
        if not self.is_connected or not self.client:
            raise ConnectionError("Not connected to CVAT server")

        if labels is None:
            labels = ["car", "motorbike"]

        label_specs = [{"name": label_name} for label_name in labels]

        spec = {
            "name": name,
            "labels": label_specs
        }

        created = self.client.projects.create(spec=spec)
        return {
            "id": created.id,
            "name": created.name,
            "labels": [l.name for l in created.get_labels()]
        }

    def list_tasks(self, project_id: Optional[int] = None) -> List[Dict]:
        """Fetch list of tasks (optionally filtered by project)"""
        if not self.is_connected or not self.client:
            raise ConnectionError("Not connected to CVAT server")

        tasks = []
        try:
            for t in self.client.tasks.list():
                t_proj_id = getattr(t, "project_id", None)
                if project_id is not None and t_proj_id != project_id:
                    continue

                size = getattr(t, "size", 0)
                status = getattr(t, "status", "unknown")

                tasks.append({
                    "id": t.id,
                    "name": t.name,
                    "project_id": t_proj_id,
                    "size": size,
                    "status": status
                })
        except Exception as e:
            print(f"[CVAT] Error listing tasks: {e}")
        return tasks

    def create_task_with_data(
        self,
        name: str,
        image_paths: List[str],
        project_id: Optional[int] = None,
        labels: Optional[List[str]] = None,
        annotation_zip_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict:
        """
        Create a new task, upload image files, and import COCO annotations
        """
        if not self.is_connected or not self.client:
            raise ConnectionError("Not connected to CVAT server")

        if progress_callback:
            progress_callback("Creating task specification on CVAT...", 10)

        spec = {"name": name}
        if project_id:
            spec["project_id"] = int(project_id)
        else:
            if labels is None:
                labels = ["car", "motorbike"]
            spec["labels"] = [{"name": l} for l in labels]

        if progress_callback:
            progress_callback(f"Uploading {len(image_paths)} image(s)...", 30)

        task = self.client.tasks.create_from_data(
            spec=spec,
            resources=image_paths,
            annotation_path=annotation_zip_path if annotation_zip_path else "",
            annotation_format="COCO 1.0" if annotation_zip_path else "CVAT XML 1.1"
        )

        if progress_callback:
            progress_callback("Task successfully created and data synchronized!", 100)

        return {
            "id": task.id,
            "name": task.name,
            "project_id": getattr(task, "project_id", None),
            "size": getattr(task, "size", len(image_paths))
        }

    def upload_annotations_to_task(
        self,
        task_id: int,
        annotation_zip_path: str,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> Tuple[bool, str]:
        """Upload COCO 1.0 annotations zip to an existing task"""
        if not self.is_connected or not self.client:
            raise ConnectionError("Not connected to CVAT server")

        try:
            if progress_callback:
                progress_callback(f"Retrieving Task #{task_id}...", 20)

            task = self.client.tasks.retrieve(task_id)

            if progress_callback:
                progress_callback(f"Importing COCO 1.0 annotations to Task #{task_id}...", 50)

            task.import_annotations(
                format_name="COCO 1.0",
                filename=annotation_zip_path
            )

            if progress_callback:
                progress_callback("Annotations successfully uploaded!", 100)

            return True, f"Successfully uploaded annotations to Task #{task_id}"

        except Exception as e:
            return False, f"Failed to upload annotations to Task #{task_id}: {str(e)}"

    def get_task_frames_info(self, task_id: int) -> List[Dict]:
        """Get metadata for frames belonging to a task"""
        if not self.is_connected or not self.client:
            raise ConnectionError("Not connected to CVAT server")

        task = self.client.tasks.retrieve(task_id)
        meta = task.get_meta()
        frames_info = []
        for idx, frame in enumerate(meta.frames):
            frames_info.append({
                "frame_id": idx,
                "name": frame.name,
                "width": frame.width,
                "height": frame.height
            })
        return frames_info

    def get_task_frame_image(self, task_id: int, frame_id: int) -> np.ndarray:
        """Download a single frame image as OpenCV BGR format"""
        if not self.is_connected or not self.client:
            raise ConnectionError("Not connected to CVAT server")

        task = self.client.tasks.retrieve(task_id)
        raw_stream = task.get_frame(frame_id, quality="original")
        img_bytes = raw_stream.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
