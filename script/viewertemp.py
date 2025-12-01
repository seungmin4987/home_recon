import os
import json
import numpy as np
import trimesh
from trimesh.geometry import plane_transform
from trimesh.transformations import rotation_matrix

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk


# ============================================================
# 메타 + GLB 로딩
# ============================================================
def load_meta_and_glb():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "received_glb")

    meta_path = os.path.join(save_dir, "received_model_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    glb_path = meta.get("glb_path") or os.path.join(save_dir, "received_model.glb")
    if not os.path.isabs(glb_path):
        glb_path = os.path.join(save_dir, os.path.basename(glb_path))
    if not os.path.exists(glb_path):
        raise FileNotFoundError(glb_path)

    plane_eq = meta.get("plane_equation")
    if plane_eq is None or len(plane_eq) < 4:
        raise ValueError("plane_equation 이 메타파일에 없습니다.")

    return glb_path, tuple(plane_eq)


def add_floor_plane(scene: trimesh.Scene):
    """
    평면 정렬 후 z=0 근처에 얇은 바닥 plane 추가 (시각화용)
    """
    try:
        bmin, bmax = scene.bounds
        size = bmax - bmin
        if not np.all(np.isfinite(size)):
            size = np.array([1.0, 1.0, 1.0])

        sx, sy = max(size[0], 1.0), max(size[1], 1.0)
        px, py = sx * 1.2, sy * 1.2
        thickness = max(sx, sy) * 0.01

        cx = (bmin[0] + bmax[0]) * 0.5
        cy = (bmin[1] + bmax[1]) * 0.5

        plane_mesh = trimesh.creation.box(extents=(px, py, thickness))
        plane_mesh.apply_translation([cx, cy, -thickness / 2.0])
        plane_mesh.visual.vertex_colors = np.array([180, 230, 200, 150], np.uint8)
        scene.add_geometry(plane_mesh)
    except Exception as e:
        print("[경고] floor plane 추가 실패:", e)


# ============================================================
# GUI
# ============================================================
class SnapshotFloorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("스냅샷 탑뷰 기반 2D 가구 배치 + 3D 보기 (회전 포함)")
        self.geometry("1400x800")

        # 평면 좌표계로 정렬된 씬
        self.scene_plane = None

        # 카메라 파라미터 (스냅샷 + 좌표변환용)
        self.cam_center = None   # (cx, cy)
        self.cam_z = None        # z_cam
        self.fov_x_rad = None
        self.fov_y_rad = None
        self.img_w = None
        self.img_h = None

        self.snapshot_img = None   # ImageTk.PhotoImage

        # 2D 가구(폴리곤) + 월드 좌표 정보
        # 각 요소: info = {
        #   "canvas_id": int,
        #   "world_x": float,
        #   "world_y": float,
        #   "w_world": float,
        #   "h_world": float,
        #   "height_world": float,
        #   "yaw_deg": float,
        # }
        self.furniture_items = []

        # 2D 기준 가구 크기 (방 크기 대비 비율)
        self.furniture_scale_2d = 0.15#25  # 방의 25% 정도

        # 🔸 3D에서만 적용되는 월드 스케일 (튜닝용)
        # 1.0이면 2D footprint 크기 그대로, 0.5면 절반, 2.0이면 두 배
        self.furniture_world_scale_3d = 0.95#8#7

        # 드래그/선택 상태
        self.placement_mode = False
        self.drag_target = None     # canvas item id
        self.drag_index = None      # furniture_items index
        self.drag_last_px = None    # (x, y)

        self.selected_index = None  # 회전 대상 선택 가구 index

        self._build_ui()
        self._load_and_align_scene()
        self.after(200, self._initial_render)

    # ----------------- UI -----------------
    def _build_ui(self):
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # 왼쪽: 스냅샷 + 가구 배치용 캔버스
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(left, bg="black")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # 오른쪽: 버튼들
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        right.columnconfigure(0, weight=1)

        self.btn_place = ttk.Button(
            right, text="가구 배치 모드: OFF", command=self.toggle_placement_mode
        )
        self.btn_place.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        self.btn_show3d = ttk.Button(
            right, text="3D 보기 (팝업)", command=self.show_3d_popup
        )
        self.btn_show3d.grid(row=1, column=0, sticky="ew", pady=(0, 5))

        # 회전 버튼
        rotate_frame = ttk.LabelFrame(right, text="회전")
        rotate_frame.grid(row=2, column=0, sticky="ew", pady=(10, 5))
        rotate_frame.columnconfigure(0, weight=1)
        rotate_frame.columnconfigure(1, weight=1)

        self.btn_rot_left = ttk.Button(
            rotate_frame, text="⟲ 좌회전 (-2°)", command=lambda: self.rotate_selected(-2.0)
        )
        self.btn_rot_right = ttk.Button(
            rotate_frame, text="⟲ 우회전 (+2°)", command=lambda: self.rotate_selected(+2.0)
        )
        self.btn_rot_left.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        self.btn_rot_right.grid(row=0, column=1, sticky="ew", padx=2, pady=2)

    # ----------------- 데이터 로딩 + 평면 정렬 -----------------
    def _load_and_align_scene(self):
        glb_path, plane_eq = load_meta_and_glb()
        print("[INFO] GLB:", glb_path)
        print("[INFO] Plane eq:", plane_eq)

        mesh_or_scene = trimesh.load(glb_path)
        if isinstance(mesh_or_scene, trimesh.Scene):
            scene = mesh_or_scene
        else:
            scene = trimesh.Scene(mesh_or_scene)

        # plane_eq: ax + by + cz + d = 0
        a, b, c, d = plane_eq
        n = np.array([a, b, c], float)
        n_norm2 = np.dot(n, n)
        if n_norm2 < 1e-8:
            raise ValueError("평면 법선이 너무 작음")

        # 평면 위 한 점 p0 = -d * n / ||n||^2
        p0 = -d * n / n_norm2
        # plane_transform(p0, n): 이 평면이 z=0, normal=+z 가 되도록
        T = plane_transform(p0, n)
        scene.apply_transform(T)

        # 시각화용 바닥 plane
        add_floor_plane(scene)

        self.scene_plane = scene

    # ----------------- 초기 렌더링 -----------------
    def _initial_render(self):
        self._render_snapshot_and_set_camera()
        self._create_default_furniture()

    # ----------------- 카메라 설정 + 스냅샷 -----------------
    def _render_snapshot_and_set_camera(self):
        if self.scene_plane is None:
            return

        self.update_idletasks()
        w = max(self.canvas.winfo_width(), 400)
        h = max(self.canvas.winfo_height(), 400)
        self.img_w, self.img_h = w, h

        bmin, bmax = self.scene_plane.bounds

        # 평면 좌표계에서의 중심 (바라보는 타겟)
        cx = (bmin[0] + bmax[0]) * 0.5
        cy = (bmin[1] + bmax[1]) * 0.5
        self.cam_center = (cx, cy)

        # 카메라 높이: 씬의 최대 z보다 충분히 위
        z_top = float(bmax[2])
        z_range = float(bmax[2] - bmin[2])
        z_cam = z_top + z_range * 2.0 + 1.0
        self.cam_z = z_cam

        # FOV 설정 (라디안)
        fov_x_deg = 60.0
        fov_x_rad = np.deg2rad(fov_x_deg)
        aspect = h / float(w)
        fov_y_rad = 2.0 * np.arctan(np.tan(fov_x_rad / 2.0) * aspect)

        self.fov_x_rad = fov_x_rad
        self.fov_y_rad = fov_y_rad

        # 카메라 생성
        camera = trimesh.scene.cameras.Camera(
            resolution=(w, h),
            fov=(np.rad2deg(fov_x_rad), np.rad2deg(fov_y_rad)),
        )

        scene = self.scene_plane.copy()
        scene.camera = camera

        # 카메라 위치: (cx, cy, z_cam), 방향: -z
        cam_T = np.eye(4)
        cam_T[:3, 3] = [cx, cy, z_cam]
        scene.camera_transform = cam_T

        try:
            png_bytes = scene.save_image(resolution=(w, h))
            if not png_bytes:
                raise RuntimeError("save_image() returned empty")

            from io import BytesIO
            img = Image.open(BytesIO(png_bytes))
            self.snapshot_img = ImageTk.PhotoImage(img)

            self.canvas.delete("all")
            self.canvas.create_image(w // 2, h // 2, image=self.snapshot_img)
        except Exception as e:
            print("[경고] 스냅샷 렌더링 실패:", e)
            self.canvas.delete("all")
            self.canvas.create_text(
                w // 2, h // 2,
                text=f"스냅샷 렌더링 실패:\n{e}",
                fill="white", font=("Arial", 12), justify="center"
            )

    # ----------------- 월드(x,y,z=0) ↔ 캔버스(px,py) -----------------
    def world_to_canvas(self, x, y):
        """
        평면 좌표계의 (x,y,z=0)을 스냅샷 이미지 픽셀 좌표로 투영
        (정확히 위에서 내려다보는 카메라 기준)
        """
        if self.cam_center is None or self.cam_z is None:
            return x, y

        cx0, cy0 = self.cam_center
        z_cam = self.cam_z

        fx = np.tan(self.fov_x_rad / 2.0) * z_cam
        fy = np.tan(self.fov_y_rad / 2.0) * z_cam

        x_cam = x - cx0
        y_cam = y - cy0

        x_ndc = x_cam / fx
        y_ndc = y_cam / fy

        px = (x_ndc * 0.5 + 0.5) * self.img_w
        py = (1.0 - (y_ndc * 0.5 + 0.5)) * self.img_h
        return px, py

    def canvas_to_world(self, px, py):
        """
        world_to_canvas 의 역변환 (z=0 평면 가정)
        """
        if self.cam_center is None or self.cam_z is None:
            return px, py

        cx0, cy0 = self.cam_center
        z_cam = self.cam_z

        fx = np.tan(self.fov_x_rad / 2.0) * z_cam
        fy = np.tan(self.fov_y_rad / 2.0) * z_cam

        x_ndc = (px / self.img_w - 0.5) * 2.0
        y_ndc = ((self.img_h - py) / self.img_h - 0.5) * 2.0

        x_cam = x_ndc * fx
        y_cam = y_ndc * fy

        x = x_cam + cx0
        y = y_cam + cy0
        return x, y

    # ----------------- 가구 그리기/재그리기 -----------------
    def _compute_polygon_points_px(self, info):
        """
        info(월드 중심/크기/각도)를 이용해
        캔버스 상 폴리곤 좌표 리스트 [x1,y1,x2,y2,...] 계산
        """
        cx_w = info["world_x"]
        cy_w = info["world_y"]
        w_w = info["w_world"]
        h_w = info["h_world"]
        yaw = np.deg2rad(info.get("yaw_deg", 0.0))

        # 로컬 코너(월드 좌표 기준, 평면 위)
        hw = w_w / 2.0
        hh = h_w / 2.0
        corners_local = np.array([
            [-hw, -hh],
            [ hw, -hh],
            [ hw,  hh],
            [-hw,  hh],
        ], dtype=np.float64)

        R = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)],
        ], dtype=np.float64)

        corners_world = corners_local @ R.T + np.array([cx_w, cy_w])

        pts = []
        for x_w, y_w in corners_world:
            px, py = self.world_to_canvas(x_w, y_w)
            pts.extend([px, py])

        return pts

    def _create_furniture_on_canvas(self, info):
        pts = self._compute_polygon_points_px(info)
        canvas_id = self.canvas.create_polygon(
            pts,
            fill="#ffcc66",
            outline="#cc8800",
            width=2,
            tags=("furniture",)
        )
        info["canvas_id"] = canvas_id

    def _redraw_furniture(self, index):
        info = self.furniture_items[index]
        pts = self._compute_polygon_points_px(info)
        self.canvas.coords(info["canvas_id"], *pts)
        # 선택 상태라면 outline 업데이트
        self._update_furniture_outline()

    def _update_furniture_outline(self):
        for idx, info in enumerate(self.furniture_items):
            cid = info["canvas_id"]
            if idx == self.selected_index:
                self.canvas.itemconfigure(cid, outline="#ff0000", width=3)
            else:
                self.canvas.itemconfigure(cid, outline="#cc8800", width=2)

    # ----------------- 기본 가구 생성 -----------------
    def _create_default_furniture(self):
        """
        방(바닥) 중앙에 기본 정육면체 가구 하나 생성
        """
        if self.scene_plane is None:
            return

        bmin, bmax = self.scene_plane.bounds
        cx = (bmin[0] + bmax[0]) * 0.5
        cy = (bmin[1] + bmax[1]) * 0.5
        room_w = bmax[0] - bmin[0]
        room_h = bmax[1] - bmin[1]

        scale_2d = self.furniture_scale_2d
        w_world = room_w * scale_2d
        h_world = room_h * scale_2d
        height_world = min(w_world, h_world) * 0.7

        info = dict(
            world_x=cx,
            world_y=cy,
            w_world=w_world,
            h_world=h_world,
            height_world=height_world,
            yaw_deg=0.0,
            canvas_id=None,
        )
        self._create_furniture_on_canvas(info)
        self.furniture_items.append(info)
        self.selected_index = 0
        self._update_furniture_outline()

    # ----------------- 배치 모드/드래그 -----------------
    def toggle_placement_mode(self):
        self.placement_mode = not self.placement_mode
        self.btn_place.config(
            text="가구 배치 모드: ON" if self.placement_mode else "가구 배치 모드: OFF"
        )
        self.drag_target = None
        self.drag_index = None
        self.drag_last_px = None

    def _find_furniture_index_by_canvas_id(self, cid):
        for i, info in enumerate(self.furniture_items):
            if info["canvas_id"] == cid:
                return i
        return None

    def on_mouse_down(self, event):
        # 가구 배치 모드일 때만 드래그/선택
        if not self.placement_mode:
            return

        x, y = event.x, event.y
        items = self.canvas.find_overlapping(x, y, x, y)
        for item in items:
            if "furniture" in self.canvas.gettags(item):
                self.drag_target = item
                idx = self._find_furniture_index_by_canvas_id(item)
                self.drag_index = idx
                self.drag_last_px = (x, y)
                # 선택 가구 업데이트
                self.selected_index = idx
                self._update_furniture_outline()
                break

    def on_mouse_move(self, event):
        if not self.placement_mode or self.drag_target is None or self.drag_last_px is None:
            return

        x, y = event.x, event.y
        last_x, last_y = self.drag_last_px
        dx = x - last_x
        dy = y - last_y
        self.drag_last_px = (x, y)

        cid = self.drag_target
        coords = self.canvas.coords(cid)
        if not coords:
            return

        new_coords = []
        for i in range(0, len(coords), 2):
            new_coords.append(coords[i] + dx)
            new_coords.append(coords[i + 1] + dy)

        self.canvas.coords(cid, *new_coords)

    def on_mouse_up(self, event):
        if not self.placement_mode or self.drag_target is None or self.drag_index is None:
            self.drag_target = None
            self.drag_index = None
            self.drag_last_px = None
            return

        cid = self.drag_target
        coords = self.canvas.coords(cid)
        if not coords:
            self.drag_target = None
            self.drag_index = None
            self.drag_last_px = None
            return

        # 폴리곤 중심 → 월드 좌표 업데이트
        xs = coords[0::2]
        ys = coords[1::2]
        cx_px = sum(xs) / len(xs)
        cy_px = sum(ys) / len(ys)
        wx, wy = self.canvas_to_world(cx_px, cy_px)

        info = self.furniture_items[self.drag_index]
        info["world_x"] = wx
        info["world_y"] = wy
        # 폴리곤은 이미 움직였으므로 coords는 그대로 두고, yaw/크기는 유지
        # (필요하면 world 기준으로 재계산해도 됨)

        self.drag_target = None
        self.drag_index = None
        self.drag_last_px = None

    # ----------------- 회전 -----------------
    def rotate_selected(self, delta_deg):
        """
        선택된 가구를 delta_deg 만큼 회전 (평면 위 yaw)
        """
        if self.selected_index is None:
            return
        if self.selected_index < 0 or self.selected_index >= len(self.furniture_items):
            return

        info = self.furniture_items[self.selected_index]
        info["yaw_deg"] = (info.get("yaw_deg", 0.0) + delta_deg) % 360.0
        self._redraw_furniture(self.selected_index)

    # ----------------- 3D 팝업 (버튼 누를 때만) -----------------
    def show_3d_popup(self):
        """
        "3D 보기 (팝업)" 버튼을 눌렀을 때만 3D 뷰어를 띄움.
        """
        if self.scene_plane is None:
            return
        scene = self.scene_plane.copy()

        scale_3d = self.furniture_world_scale_3d  # 🔸 3D에서만 적용되는 스케일

        for info in self.furniture_items:
            wx = info["world_x"]
            wy = info["world_y"]
            ww = info["w_world"] * scale_3d
            wh = info["h_world"] * scale_3d
            hh = info["height_world"] * scale_3d
            yaw_deg = info.get("yaw_deg", 0.0)

            box = trimesh.creation.box(extents=(ww, wh, hh))

            # yaw 회전 (z축 기준)
            R = rotation_matrix(np.deg2rad(yaw_deg), [0, 0, 1])
            T = np.eye(4)
            T[:3, 3] = [wx, wy, hh / 2.0]
            M = T @ R
            box.apply_transform(M)

            color = np.array([255, 200, 100, 230], np.uint8)
            box.visual.vertex_colors = color
            scene.add_geometry(box)

        try:
            scene.show()
        except Exception as e:
            messagebox.showerror("3D 보기 오류", str(e))


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    try:
        app = SnapshotFloorApp()
        app.mainloop()
    except Exception as e:
        messagebox.showerror("오류", str(e))

