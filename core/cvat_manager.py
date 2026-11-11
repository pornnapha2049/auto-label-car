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
        self._org_slug_cache: Dict[int, str] = {}
        self._project_org_cache: Dict[int, Optional[str]] = {}
        self._task_org_cache: Dict[int, Optional[str]] = {}

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
                self._preload_organizations()
                return True, "Authenticated successfully via Token"

            elif username and password:
                # Session Authentication
                self.client.login((username.strip(), password))
                self.is_connected = True
                self._preload_organizations()
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

    def _preload_organizations(self):
        """Fetch and cache organizations the user has access to"""
        if not self.client:
            return
        try:
            for org in self.client.organizations.list():
                self._org_slug_cache[org.id] = org.slug
        except Exception as e:
            print(f"[CVAT] Note: Preloading organizations: {e}")

    def _get_or_fetch_org_slug(self, org_id: Optional[int]) -> Optional[str]:
        if not org_id or not self.client:
            return None
        if org_id in self._org_slug_cache:
            return self._org_slug_cache[org_id]
        try:
            org = self.client.organizations.retrieve(int(org_id))
            slug = getattr(org, "slug", None)
            if slug:
                self._org_slug_cache[org_id] = slug
            return slug
        except Exception as e:
            print(f"[CVAT] Error retrieving organization #{org_id}: {e}")
            return None

    def _resolve_project_org_slug(self, project_id: Optional[int]) -> Optional[str]:
        if not project_id or not self.client:
            return None
        p_id = int(project_id)
        if p_id in self._project_org_cache:
            return self._project_org_cache[p_id]

        org_slug = None
        try:
            proj = self.client.projects.retrieve(p_id)
            org_id = getattr(proj, "organization_id", None) or getattr(proj, "organization", None)
            if org_id:
                org_slug = self._get_or_fetch_org_slug(int(org_id))
        except Exception:
            try:
                for org in self.client.organizations.list():
                    self._org_slug_cache[org.id] = org.slug
                    with self.client.organization_context(org.slug):
                        try:
                            proj = self.client.projects.retrieve(p_id)
                            org_slug = org.slug
                            break
                        except Exception:
                            continue
            except Exception:
                pass

        self._project_org_cache[p_id] = org_slug
        return org_slug

    def _resolve_task_org_slug(self, task_id: int) -> Optional[str]:
        if not task_id or not self.client:
            return None
        t_id = int(task_id)
        if t_id in self._task_org_cache:
            return self._task_org_cache[t_id]

        org_slug = None
        try:
            task = self.client.tasks.retrieve(t_id)
            org_id = getattr(task, "organization_id", None) or getattr(task, "organization", None)
            if org_id:
                org_slug = self._get_or_fetch_org_slug(int(org_id))
            elif getattr(task, "project_id", None):
                org_slug = self._resolve_project_org_slug(task.project_id)
        except Exception:
            try:
                for org in self.client.organizations.list():
                    self._org_slug_cache[org.id] = org.slug
                    with self.client.organization_context(org.slug):
                        try:
                            task = self.client.tasks.retrieve(t_id)
                            org_slug = org.slug
                            break
                        except Exception:
                            continue
            except Exception:
                pass

        self._task_org_cache[t_id] = org_slug
        return org_slug

    def list_projects(self) -> List[Dict]:
        """Fetch list of all projects across personal and organizations"""
        if not self.is_connected or not self.client:
            raise ConnectionError("Not connected to CVAT server")

        projects = []
        seen_ids = set()

        def add_p(p):
            if p.id in seen_ids:
                return
            seen_ids.add(p.id)
            labels = [l.name for l in p.get_labels()]
            org_id = getattr(p, "organization_id", None) or getattr(p, "organization", None)
            org_slug = None
            if org_id:
                org_slug = self._get_or_fetch_org_slug(int(org_id))
            self._project_org_cache[p.id] = org_slug
            projects.append({
                "id": p.id,
                "name": p.name,
                "labels": labels,
                "organization": org_slug,
                "created_date": str(getattr(p, "created_date", ""))
            })

        try:
            # 1. Projects in default context
            for p in self.client.projects.list():
                add_p(p)

            # 2. Projects in each organization
            try:
                for org in self.client.organizations.list():
                    self._org_slug_cache[org.id] = org.slug
                    with self.client.organization_context(org.slug):
                        for p in self.client.projects.list():
                            add_p(p)
            except Exception:
                pass
        except Exception as e:
            print(f"[CVAT] Error listing projects: {e}")
        return projects

    def create_project(
        self,
        name: str,
        labels: Optional[List[str]] = None,
        organization_slug: Optional[str] = None
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

        prev_org_slug = self.client.organization_slug
        try:
            if organization_slug:
                self.client.organization_slug = organization_slug
            created = self.client.projects.create(spec=spec)
            org_id = getattr(created, "organization_id", None) or getattr(created, "organization", None)
            org_slug = self._get_or_fetch_org_slug(org_id) if org_id else organization_slug
            self._project_org_cache[created.id] = org_slug

            return {
                "id": created.id,
                "name": created.name,
                "organization": org_slug,
                "labels": [l.name for l in created.get_labels()]
            }
        finally:
            self.client.organization_slug = prev_org_slug

    def list_tasks(self, project_id: Optional[int] = None) -> List[Dict]:
        """Fetch list of tasks (optionally filtered by project)"""
        if not self.is_connected or not self.client:
            raise ConnectionError("Not connected to CVAT server")

        tasks = []
        seen_ids = set()

        def add_t(t):
            if t.id in seen_ids:
                return
            t_proj_id = getattr(t, "project_id", None)
            if project_id is not None and t_proj_id != project_id:
                return
            seen_ids.add(t.id)
            org_id = getattr(t, "organization_id", None) or getattr(t, "organization", None)
            if org_id and org_id in self._org_slug_cache:
                self._task_org_cache[t.id] = self._org_slug_cache[org_id]
            tasks.append({
                "id": t.id,
                "name": t.name,
                "project_id": t_proj_id,
                "size": getattr(t, "size", 0),
                "status": getattr(t, "status", "unknown")
            })

        target_org_slug = None
        if project_id and int(project_id) > 0:
            target_org_slug = self._resolve_project_org_slug(int(project_id))

        if target_org_slug:
            with self.client.organization_context(target_org_slug):
                try:
                    for t in self.client.tasks.list():
                        add_t(t)
                except Exception as e:
                    print(f"[CVAT] Error listing tasks in org '{target_org_slug}': {e}")
        else:
            try:
                for t in self.client.tasks.list():
                    add_t(t)
            except Exception:
                pass
            try:
                for org in self.client.organizations.list():
                    self._org_slug_cache[org.id] = org.slug
                    with self.client.organization_context(org.slug):
                        for t in self.client.tasks.list():
                            add_t(t)
            except Exception:
                pass

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
        Automatically handles and matches the target project's Organization context!
        """
        if not self.is_connected or not self.client:
            raise ConnectionError("Not connected to CVAT server")

        if progress_callback:
            progress_callback("Resolving project and organization settings...", 5)

        spec = {"name": name}
        target_org_slug = None
        if project_id and int(project_id) > 0:
            spec["project_id"] = int(project_id)
            target_org_slug = self._resolve_project_org_slug(int(project_id))
        else:
            if labels is None:
                labels = ["car", "motorbike"]
            spec["labels"] = [{"name": l} for l in labels]

        if progress_callback:
            progress_callback(f"Uploading {len(image_paths)} image(s)...", 30)

        prev_org_slug = self.client.organization_slug
        try:
            self.client.organization_slug = target_org_slug
            task = self.client.tasks.create_from_data(
                spec=spec,
                resources=image_paths,
                annotation_path=annotation_zip_path if annotation_zip_path else "",
                annotation_format="COCO 1.0" if annotation_zip_path else "CVAT XML 1.1"
            )
            self._task_org_cache[task.id] = target_org_slug
        finally:
            self.client.organization_slug = prev_org_slug

        if progress_callback:
            progress_callback("Task successfully created and data synchronized!", 100)

        return {
            "id": task.id,
            "name": task.name,
            "project_id": getattr(task, "project_id", None),
            "organization": target_org_slug,
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

        target_org_slug = self._resolve_task_org_slug(task_id)
        prev_org_slug = self.client.organization_slug
        try:
            self.client.organization_slug = target_org_slug

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
        finally:
            self.client.organization_slug = prev_org_slug

    def get_task_size(self, task_id: int) -> int:
        """Get the number of frames currently in a task"""
        if not self.is_connected or not self.client:
            raise ConnectionError("Not connected to CVAT server")
        target_org_slug = self._resolve_task_org_slug(task_id)
        prev_org_slug = self.client.organization_slug
        try:
            self.client.organization_slug = target_org_slug
            task = self.client.tasks.retrieve(task_id)
            return getattr(task, "size", 0)
        finally:
            self.client.organization_slug = prev_org_slug

    def upload_images_to_task(
        self,
        task_id: int,
        image_paths: List[str],
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> None:
        """Upload image files to an existing empty or active task"""
        if not self.is_connected or not self.client:
            raise ConnectionError("Not connected to CVAT server")
        target_org_slug = self._resolve_task_org_slug(task_id)
        prev_org_slug = self.client.organization_slug
        try:
            self.client.organization_slug = target_org_slug
            if progress_callback:
                progress_callback(f"Uploading {len(image_paths)} image(s) to Task #{task_id}...", 30)
            task = self.client.tasks.retrieve(task_id)
            task.upload_data(resources=image_paths)
        finally:
            self.client.organization_slug = prev_org_slug

    def get_task_frames_info(self, task_id: int) -> List[Dict]:
        """Get metadata for frames belonging to a task"""
        if not self.is_connected or not self.client:
            raise ConnectionError("Not connected to CVAT server")

        target_org_slug = self._resolve_task_org_slug(task_id)
        prev_org_slug = self.client.organization_slug
        try:
            self.client.organization_slug = target_org_slug
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
        finally:
            self.client.organization_slug = prev_org_slug

    def get_task_frame_image(self, task_id: int, frame_id: int) -> np.ndarray:
        """Download a single frame image as OpenCV BGR format"""
        if not self.is_connected or not self.client:
            raise ConnectionError("Not connected to CVAT server")

        target_org_slug = self._resolve_task_org_slug(task_id)
        prev_org_slug = self.client.organization_slug
        try:
            self.client.organization_slug = target_org_slug
            task = self.client.tasks.retrieve(task_id)
            raw_stream = task.get_frame(frame_id, quality="original")
            img_bytes = raw_stream.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        finally:
            self.client.organization_slug = prev_org_slug
