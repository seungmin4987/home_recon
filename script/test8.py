import os
import json
import threading
import queue
import numpy as np
import trimesh
from trimesh.geometry import plane_transform
from trimesh.transformations import rotation_matrix

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk


# ============================================================
# 🔧 템플릿 Up-Axis 보정 각도
# ============================================================
TEMPLATE_ROT_X_DEG = 90.0   # 예: Y-up -> Z-up
TEMPLATE_ROT_Y_DEG = 0.0
TEMPLATE_ROT_Z_DEG = 0.0

# ----------------- 천장 클리핑 파라미터 -----------------
# 이 값은 이제 "기본값" 정도로만 쓰고,
# 실제 천장 plane 높이는 z 분포의 95% 지점으로 자동 계산해서 사용.
CEILING_LIFT_DIST = 2.5  # fallback 용 기본값


# ============================================================
# 메타 + GLB 로딩 (방 전체 + 평면 방정식)
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


# ============================================================
# 정렬된 Scene 에서 점 + 색상 모으기
#   (scene.dump(concatenate=True) 사용: graph transform 이 반영된 좌표)
# ============================================================
def scene_to_points_with_colors(scene: trimesh.Scene):
    merged = scene.dump(concatenate=True)  # world 좌표로 합쳐진 geometry

    if isinstance(merged, trimesh.Trimesh):
        pts = np.asarray(merged.vertices, dtype=np.float64)

        vc = None
        if hasattr(merged, "visual") and hasattr(merged.visual, "vertex_colors"):
            vc = np.asarray(merged.visual.vertex_colors)
        if vc is None or vc.shape[0] != pts.shape[0]:
            vc = np.tile(
                np.array([[200, 200, 200, 255]], dtype=np.uint8),
                (pts.shape[0], 1)
            )
        else:
            vc = vc.astype(np.uint8)
            # (N,3) 이면 alpha 채워주기
            if vc.ndim == 2 and vc.shape[1] == 3:
                alpha = np.full((vc.shape[0], 1), 255, dtype=np.uint8)
                vc = np.concatenate([vc, alpha], axis=1)

        return pts, vc

    elif isinstance(merged, trimesh.points.PointCloud):
        pts = np.asarray(merged.vertices, dtype=np.float64)
        vc = None
        if hasattr(merged, "colors") and merged.colors is not None:
            vc = np.asarray(merged.colors)
        if vc is None or vc.shape[0] != pts.shape[0]:
            vc = np.tile(
                np.array([[200, 200, 200, 255]], dtype=np.uint8),
                (pts.shape[0], 1)
            )
        else:
            vc = vc.astype(np.uint8)
            if vc.ndim == 2 and vc.shape[1] == 3:
                alpha = np.full((vc.shape[0], 1), 255, dtype=np.uint8)
                vc = np.concatenate([vc, alpha], axis=1)
        return pts, vc

    else:
        raise TypeError(f"scene_to_points_with_colors: 알 수 없는 타입: {type(merged)}")


# ============================================================
# 바닥 / 천장 / 서브천장 plane 시각화용 박스
# ============================================================
def add_floor_plane(scene: trimesh.Scene, bounds=None):
    """
    평면 정렬 후 z=0 근처에 얇은 바닥 plane 추가 (시각화용)
    """
    try:
        if bounds is None:
            bmin, bmax = scene.bounds
        else:
            bmin, bmax = bounds

        size = bmax - bmin
        if not np.all(np.isfinite(size)):
            size = np.array([1.0, 1.0, 1.0])

        sx, sy = max(size[0], 1.0), max(size[1], 1.0)
        px, py = sx * 1.2, sy * 1.2
        thickness = max(sx, sy) * 0.01

        cx = (bmin[0] + bmax[0]) * 0.5
        cy = (bmin[1] + bmax[1]) * 0.5
        plane_mesh = trimesh.creation.box(extents=(px, py, thickness))
        # z=0 이 바닥이 되도록, 박스의 윗면이 z=0에 오게 배치
        plane_mesh.apply_translation([cx, cy, -thickness / 2.0])
        plane_mesh.visual.vertex_colors = np.array([130, 118, 95, 150], np.uint8)
        scene.add_geometry(plane_mesh)
    except Exception as e:
        print("[경고] floor plane 추가 실패:", e)


def add_ceiling_plane(scene: trimesh.Scene,
                      ceiling_height: float,
                      color_rgba=None,
                      bounds=None):
    """
    z축 기준 ceiling_height 위치에 얇은 천장 plane 추가 (시각화용)
    """
    if color_rgba is None:
        color_rgba = np.array([200, 80, 80, 130], np.uint8)  # 기본: 빨간색 계열

    try:
        if bounds is None:
            bmin, bmax = scene.bounds
        else:
            bmin, bmax = bounds

        size = bmax - bmin
        if not np.all(np.isfinite(size)):
            size = np.array([1.0, 1.0, 1.0])

        sx, sy = max(size[0], 1.0), max(size[1], 1.0)
        px, py = sx * 1.2, sy * 1.2
        thickness = max(sx, sy) * 0.01

        cx = (bmin[0] + bmax[0]) * 0.5
        cy = (bmin[1] + bmax[1]) * 0.5

        plane_mesh = trimesh.creation.box(extents=(px, py, thickness))
        plane_mesh.apply_translation([cx, cy, ceiling_height - thickness / 2.0])
        plane_mesh.visual.vertex_colors = np.array(color_rgba, np.uint8)
        scene.add_geometry(plane_mesh)
    except Exception as e:
        print("[경고] ceiling plane 추가 실패:", e)


def add_sub_ceiling_plane(scene: trimesh.Scene,
                          ceiling_height: float,
                          ratio: float = 0.95,
                          bounds=None):
    """
    천장보다 약간 아래 평면 추가.
    - ceiling_height 가 H라면, 서브천장 높이는 H * ratio (기본 0.95)
    """
    sub_height = ceiling_height * ratio
    print(f"[INFO] sub-ceiling height = {sub_height:.3f} (ratio={ratio})")

    color = np.array([80, 200, 80, 130], np.uint8)  # 초록색
    add_ceiling_plane(scene,
                      sub_height,
                      color_rgba=color,
                      bounds=bounds)
    return sub_height


# ============================================================
# 가상 자(virtual ruler) 생성: 바닥 위에 길이 L인 박스 생성 (3D용)
# ============================================================
def add_virtual_ruler(scene: trimesh.Scene,
                      p1: np.ndarray,
                      p2: np.ndarray,
                      thickness: float = 0.02,
                      color_rgba=None):
    """
    p1 -> p2 방향으로 길이 |p2-p1| 인 박스(자)를 바닥 위에 생성
    """
    try:
        v = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
        length = float(np.linalg.norm(v))
        if length < 1e-6:
            print("[WARN] 가상 자 길이가 너무 짧아서 생성 스킵")
            return

        direction = v / length
        center = (np.asarray(p1, dtype=float) + np.asarray(p2, dtype=float)) * 0.5

        if color_rgba is None:
            color_rgba = np.array([255, 255, 0, 255], np.uint8)  # 노란색

        # 기본 박스: x축 방향으로 length, y/z 는 thickness
        box = trimesh.creation.box(extents=(length, thickness, thickness))

        # 방향 정렬용 회전 행렬 (local x -> direction)
        x_axis = direction
        up_ref = np.array([0.0, 0.0, 1.0], dtype=float)
        # direction 이 z축과 거의 평행이면 다른 축 사용
        if abs(np.dot(x_axis, up_ref)) > 0.99:
            up_ref = np.array([0.0, 1.0, 0.0], dtype=float)

        y_axis = np.cross(up_ref, x_axis)
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-6:
            print("[WARN] 가상 자 회전축 계산 실패")
            return
        y_axis /= y_norm
        z_axis = np.cross(x_axis, y_axis)

        R = np.eye(4)
        R[:3, 0] = x_axis
        R[:3, 1] = y_axis
        R[:3, 2] = z_axis
        R[:3, 3] = center

        box.apply_transform(R)
        box.visual.vertex_colors = color_rgba
        scene.add_geometry(box)
    except Exception as e:
        print("[WARN] 가상 자 생성 중 오류:", e)


# ============================================================
# 1) 씬 정렬 + z=0 floor snap + auto ceiling 높이 계산
#    (공용 유틸: 평면 기준 정렬 + 천장 높이 추정)
# ============================================================
def align_scene_and_compute_ceiling(plane_flipped: bool = True):
    """
    1) GLB 로드 + 평면 정렬
    2) z≈0 으로 floor snap
    3) z 분포(95% quantile)로 천장 높이 자동 추정
    4) 정렬된 scene, auto_ceiling_height, aligned_bounds 반환
    """
    glb_path, plane_eq = load_meta_and_glb()
    print("[INFO] GLB:", glb_path)
    print("[INFO] Plane eq (raw):", plane_eq)

    # GLB 로드
    mesh_or_scene = trimesh.load(glb_path)
    if isinstance(mesh_or_scene, trimesh.Scene):
        scene = mesh_or_scene
    else:
        scene = trimesh.Scene(mesh_or_scene)

    # 1) 바닥 평면 기준으로 법선 방향 정리 (방 안쪽이 +dist 되도록)
    a, b, c, d = plane_eq
    n = np.array([a, b, c], dtype=float)
    nn = float(np.dot(n, n))
    if nn < 1e-8:
        raise ValueError("평면 법선이 너무 작음")
    n_len = np.sqrt(nn)

    try:
        merged = scene.dump(concatenate=True)
        if hasattr(merged, "vertices") and merged.vertices.shape[0] > 0:
            verts = merged.vertices
            signed = (verts @ n + d) / n_len
            median_dist = float(np.median(signed))
            # 방 안쪽(천장 방향)이 + 가 되도록 방향 자동 결정
            if median_dist < 0.0:
                n = -n
                d = -d
                print("[INFO] plane orientation flipped so that inside is +dist")
        else:
            print("[WARN] verts 가 비어서 plane orientation 추정 생략.")
    except Exception as e:
        print("[WARN] plane orientation 추정 중 오류:", e)

    # 2) plane_flipped 옵션 적용 + plane_transform 로 정렬
    a2, b2, c2 = n[0], n[1], n[2]
    d2 = d

    if plane_flipped:
        a2, b2, c2, d2 = -a2, -b2, -c2, -d2
        print("[INFO] plane_flipped 적용됨 (transform 용).")

    n2 = np.array([a2, b2, c2], dtype=float)
    nn2 = float(np.dot(n2, n2))
    if nn2 < 1e-8:
        raise ValueError("평면 법선이 너무 작음 (after orient/flip)")

    # 새 평면(ax+by+cz+d=0)의 한 점 p0
    p0 = -d2 * n2 / nn2

    # plane_transform(p0, n2): p0를 원점으로 옮기고, n2를 +Z축으로 보정
    T = plane_transform(p0, n2)
    scene.apply_transform(T)

    # 3) 바닥을 z ≈ 0 으로 스냅
    try:
        merged2 = scene.dump(concatenate=True)
        z_all = merged2.vertices[:, 2]
    except Exception:
        z_all = None

    if z_all is not None and z_all.size > 0:
        z_floor = float(np.quantile(z_all, 0.01))
        scene.apply_translation([0.0, 0.0, -z_floor])
        print(f"[INFO] floor snapped: ~{z_floor:.4f} → 0.0")
    else:
        print("[WARN] floor snap 생략 (z_all 없음)")

    # 4) z 분포 기준으로 천장 높이 자동 선택 (95% quantile)
    auto_ceiling_height = None
    try:
        merged3 = scene.dump(concatenate=True)
        z_all2 = merged3.vertices[:, 2]
        if z_all2.size > 0:
            z_min = float(np.quantile(z_all2, 0.01))
            z_max = float(np.quantile(z_all2, 0.99))
            z_95 = float(np.quantile(z_all2, 0.95))
            auto_ceiling_height = max(z_95, 0.0)
            print(
                f"[INFO] z-range≈[{z_min:.3f}, {z_max:.3f}], "
                f"auto ceiling(95%)≈{auto_ceiling_height:.3f}"
            )
        else:
            print("[WARN] z_all2 가 비어 있어 auto ceiling 계산 불가.")
    except Exception as e:
        print("[WARN] auto ceiling 계산 중 오류:", e)

    if auto_ceiling_height is None:
        print("[WARN] auto ceiling 높이 계산에 실패하여 fallback 사용")
        auto_ceiling_height = CEILING_LIFT_DIST

    # 정렬 + 스냅된 상태의 bounds (전체 방 기준)
    aligned_bounds = scene.bounds

    # 디버깅용 최종 z-range
    try:
        merged4 = scene.dump(concatenate=True)
        z_all4 = merged4.vertices[:, 2]
        z_min4 = float(np.quantile(z_all4, 0.01))
        z_max4 = float(np.quantile(z_all4, 0.99))
        print(
            f"[DEBUG] aligned scene final z-range≈[{z_min4:.3f}, {z_max4:.3f}]"
        )
    except Exception:
        pass

    return scene, auto_ceiling_height, aligned_bounds


# ============================================================
# GUI
# ============================================================
class SnapshotFloorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("가구배치 시뮬레이터")
        # 화면 절반을 오른쪽에 배치
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        win_w = max(int(sw * 0.48), 1000)
        win_h = max(int(sh * 0.9), 700)
        self.geometry(f"{win_w}x{win_h}+{sw - win_w}+0")

        # 평면 좌표계로 정렬된 씬 (방 전체)
        self.scene_plane = None

        # 카메라 파라미터 (스냅샷 + 좌표변환용)
        self.cam_center = None   # (cx, cy)
        self.cam_z = None        # z_cam
        self.fov_x_rad = None
        self.fov_y_rad = None
        self.img_w = None
        self.img_h = None

        self.snapshot_img = None   # ImageTk.PhotoImage

        # 템플릿: name -> { mesh, footprint_w, footprint_d, height }
        self.templates = {}        # glb 템플릿들
        self.template_var = tk.StringVar()

        # 2D/3D 공통 가구 인스턴스 정보 리스트
        # {
        #   "template": str,
        #   "world_x": float,
        #   "world_y": float,
        #   "yaw_deg": float,
        #   "scale": float,        # 전체 비율
        #   "target_w": float,     # 기준 가로 (m)
        #   "target_d": float,     # 기준 세로 (m)
        #   "canvas_id": int,
        # }
        self.furniture_items = []

        # 🔸 3D에서만 적용되는 전역 스케일 (튜닝용)
        self.furniture_world_scale_3d = 1.0

        # 드래그/선택 상태
        self.placement_mode = False
        self.drag_target = None
        self.drag_index = None
        self.drag_last_px = None

        self.selected_index = None
        self.scale_var = tk.DoubleVar(value=1.0)

        # 가구 크기 지정용 입력 값 (m 단위)
        self.width_entry_var = tk.StringVar()
        self.depth_entry_var = tk.StringVar()

        # 🔁 평면 뒤집기 상태 플래그 (False: 원본, True: n,d에 -를 곱한 상태)
        self.plane_flipped = True

        # 🔍 거리 설정 관련 상태
        self.measure_mode = False
        self.measure_points_canvas = []  # [(px,py), ...]
        self.measure_points_world = []   # [(wx,wy), ...]
        self.measure_point_ids = []      # 캔버스 위 포인트 표시용 아이템 id
        self.measure_line_id = None      # 캔버스 위 선 표시용 아이템 id
        self.measure_real_dist_m = None  # 사용자가 입력한 실제 거리(m)

        # ⭐ 메트릭 스케일 (기본 1.0) – 평면 뒤집기 후에도 유지
        self.metric_scale = 1.0

        # 🎥 실시간 3D 미리보기 상태
        self.live_preview_on = False
        self.live_preview_job = None
        self.live_preview_delay_ms = 250  # 디바운스 렌더 주기
        self.live_view_thread = None
        self.live_view_queue = None
        self.live_viewer = None
        self.live_view_init_loc = None

        # 회전 연속 입력 상태
        self._rotate_job = None
        self._rotate_delta = 0.0
        self._rotate_delay_job = None

        # 🔹 가상 자 상태
        self.ruler_enabled = False
        self.ruler_length_m = 1.0
        self.ruler_world_p1 = None   # np.array([x,y,z])
        self.ruler_world_p2 = None   # np.array([x,y,z])
        self.ruler_p1_id = None
        self.ruler_p2_id = None
        self.ruler_line_id = None
        self.ruler_label_bg_id = None
        self.ruler_label_text_id = None
        self.ruler_dragging_point = None  # "p1" / "p2" / None

        self._build_ui()
        self._load_and_align_scene()
        self._load_templates_from_dir()
        self.after(200, self._initial_render)

    # ----------------- UI -----------------
    def _build_ui(self):
        style = ttk.Style()
        style.configure("TButton", padding=(4, 2))

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=8)
        self.rowconfigure(1, weight=1)

        # 상단: 스냅샷 + 가구 배치 캔버스
        canvas_wrap = ttk.Frame(self)
        canvas_wrap.grid(row=0, column=0, sticky="nsew", padx=5, pady=(5, 2))
        canvas_wrap.rowconfigure(0, weight=1)
        canvas_wrap.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_wrap, bg="#dcdcdc")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # 하단: 컨트롤 패널
        right = ttk.Frame(self)
        right.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
        right.columnconfigure(0, weight=1)
        right.columnconfigure(1, weight=1)

        # 상단 주요 버튼을 2열 그리드로 배치
        top_btns = ttk.Frame(right)
        top_btns.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        top_btns.columnconfigure(0, weight=1)
        top_btns.columnconfigure(1, weight=1)

        self.btn_place = ttk.Button(
            top_btns, text="가구 배치 모드: OFF", command=self.toggle_placement_mode
        )
        self.btn_place.grid(row=0, column=0, sticky="ew", padx=2, pady=(0, 4))

        self.btn_measure = ttk.Button(
            top_btns, text="거리 설정 모드: OFF", command=self.toggle_measure_mode
        )
        self.btn_measure.grid(row=0, column=1, sticky="ew", padx=2, pady=(0, 4))

        self.btn_live3d = ttk.Button(
            top_btns, text="실시간 3D 미리보기: OFF", command=self.toggle_live_preview
        )
        self.btn_live3d.grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=(0, 4))

        self.btn_flip_plane = ttk.Button(
            top_btns, text="평면 뒤집기", command=self.flip_plane_and_reload
        )
        self.btn_flip_plane.grid(row=2, column=0, columnspan=2, sticky="ew", padx=2, pady=(0, 4))

        # 🔹 가상 자 ON/OFF 버튼
        self.btn_ruler = ttk.Button(
            top_btns, text="가상 자: OFF", command=self.toggle_ruler
        )
        self.btn_ruler.grid(row=3, column=0, columnspan=2, sticky="ew", padx=2, pady=(0, 6))

        # 템플릿 선택 + 가구 추가/삭제
        tpl_frame = ttk.LabelFrame(right, text="가구 템플릿")
        tpl_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 4))
        tpl_frame.columnconfigure(0, weight=1)
        tpl_frame.columnconfigure(1, weight=1)

        self.tpl_combo = ttk.Combobox(
            tpl_frame,
            textvariable=self.template_var,
            state="readonly"
        )
        self.tpl_combo.grid(row=0, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        self.btn_add_furniture = ttk.Button(
            tpl_frame, text="가구 추가", command=self.add_furniture_instance
        )
        self.btn_add_furniture.grid(row=1, column=0, sticky="ew", padx=2, pady=2)

        self.btn_delete_furniture = ttk.Button(
            tpl_frame, text="선택 가구 삭제", command=self.delete_selected_furniture
        )
        self.btn_delete_furniture.grid(row=1, column=1, sticky="ew", padx=2, pady=2)

        # 회전 버튼
        rotate_frame = ttk.LabelFrame(right, text="회전")
        rotate_frame.grid(row=2, column=0, sticky="ew", padx=(0, 2), pady=(4, 4))
        rotate_frame.columnconfigure(0, weight=1)
        rotate_frame.columnconfigure(1, weight=1)

        self.btn_rot_left = ttk.Button(
            rotate_frame, text="⟲ -1°", command=lambda: self.rotate_selected(+1.0)
        )
        self.btn_rot_right = ttk.Button(
            rotate_frame, text="⟳ +1°", command=lambda: self.rotate_selected(-1.0)
        )
        self.btn_rot_left.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        self.btn_rot_right.grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        # 길게 누르면 연속 회전
        self.btn_rot_left.bind("<ButtonPress-1>", lambda e: self.start_rotate(+1.0))
        self.btn_rot_right.bind("<ButtonPress-1>", lambda e: self.start_rotate(-1.0))
        self.btn_rot_left.bind("<ButtonRelease-1>", lambda e: self.stop_rotate())
        self.btn_rot_right.bind("<ButtonRelease-1>", lambda e: self.stop_rotate())

        # 스케일 슬라이더
        scale_frame = ttk.LabelFrame(right, text="선택 가구 스케일 (비율)")
        scale_frame.grid(row=2, column=1, sticky="ew", padx=(2, 0), pady=(4, 4))
        scale_frame.columnconfigure(0, weight=1)

        self.scale_var.set(1.0)
        scale_widget = ttk.Scale(
            scale_frame,
            from_=0.2,
            to=3.0,
            orient="horizontal",
            variable=self.scale_var,
            command=self.on_scale_change,
        )
        scale_widget.grid(row=0, column=0, sticky="ew", padx=2, pady=4)

        # 🔧 가구 크기 직접 지정 (m 단위 느낌)
        size_frame = ttk.LabelFrame(right, text="가구 크기 지정 (m)")
        size_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 2))
        size_frame.columnconfigure(1, weight=1)

        ttk.Label(size_frame, text="가로 (폭):").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        width_entry = ttk.Entry(size_frame, textvariable=self.width_entry_var)
        width_entry.grid(row=0, column=1, sticky="ew", padx=2, pady=2)

        ttk.Label(size_frame, text="세로 (깊이):").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        depth_entry = ttk.Entry(size_frame, textvariable=self.depth_entry_var)
        depth_entry.grid(row=1, column=1, sticky="ew", padx=2, pady=2)

        btn_apply_size = ttk.Button(
            size_frame, text="크기 적용", command=self.apply_furniture_size
        )
        btn_apply_size.grid(row=2, column=0, columnspan=2, sticky="ew", padx=2, pady=4)

    # ----------------- 평면 뒤집기 버튼 콜백 -----------------
    def flip_plane_and_reload(self):
        # 평면 뒤집기 토글
        self.plane_flipped = not self.plane_flipped
        print(f"[INFO] plane_flipped = {self.plane_flipped}")

        # 평면 뒤집을 때, 가상 자는 다시 기본 위치에서 시작하도록 초기화
        self.ruler_world_p1 = None
        self.ruler_world_p2 = None
        self.ruler_dragging_point = None

        # GLB 다시 로딩 + 평면 정렬 + (⭐ 메트릭 스케일 재적용)
        self._load_and_align_scene()
        self._render_snapshot_and_set_camera()
        for idx in range(len(self.furniture_items)):
            self._redraw_furniture(idx)

    # ----------------- 거리 설정 모드 토글 -----------------
    def toggle_measure_mode(self):
        self.measure_mode = not self.measure_mode
        self.ruler_dragging_point = None  # 측정 모드 전환 시 자 드래그 종료
        if self.measure_mode:
            # 거리 측정 켜면 가구 배치 모드 끔
            self.placement_mode = False
            self.btn_place.config(text="가구 배치 모드: OFF")
            self.btn_measure.config(text="거리 설정 모드: ON")

            # 이전 측정 결과 초기화 (캔버스 상 표시만)
            for cid in self.measure_point_ids:
                self.canvas.delete(cid)
            self.measure_point_ids.clear()
            if self.measure_line_id is not None:
                self.canvas.delete(self.measure_line_id)
                self.measure_line_id = None
            self.measure_points_canvas.clear()
            # measure_points_world / metric_scale / 가구 등은 유지
        else:
            self.btn_measure.config(text="거리 설정 모드: OFF")

    # ----------------- 가상 자 기본 위치 보장 -----------------
    def _ensure_ruler_world_default(self, bounds):
        """
        self.ruler_world_p1/p2 가 없으면, 방 앞쪽에 길이 L 짜리 자를 기본 생성.
        """
        if self.ruler_world_p1 is not None and self.ruler_world_p2 is not None:
            return

        bmin, bmax = bounds
        length_m = self.measure_real_dist_m if (self.measure_real_dist_m is not None and self.measure_real_dist_m > 0) else 1.0

        cx = (bmin[0] + bmax[0]) * 0.5
        y_front = bmin[1] + (bmax[1] - bmin[1]) * 0.05
        z_r = 0.02  # 바닥에서 살짝 띄움

        p1 = np.array([cx - length_m / 2.0, y_front, z_r], dtype=float)
        p2 = np.array([cx + length_m / 2.0, y_front, z_r], dtype=float)

        self.ruler_world_p1 = p1
        self.ruler_world_p2 = p2
        self.ruler_length_m = float(length_m)

        print(f"[INFO] 기본 가상 자 설정: length={self.ruler_length_m:.3f}m")

    # ----------------- 데이터 로딩 + 평면 정렬 + 천장 클리핑 + 가상자 -----------------
    def _load_and_align_scene(self):
        """
        1) GLB 로드 + 평면 정렬 (align_scene_and_compute_ceiling 사용)
        2) floor snap
        3) metric_scale 재적용
        4) z 분포의 95% 지점으로 천장 높이 자동 선택
        5) 서브천장(높이의 5% 아래) 기준으로 점들을 위/아래로 나눈 뒤,
           서브천장 **아래(z <= sub_height)** 점들만 남김
        6) 바닥 plane + (옵션) 가상 자 3D 박스 생성
        """
        # 1) 정렬 + auto ceiling 계산 (여기서는 metric_scale 적용 전)
        scene_aligned, auto_ceiling_height, aligned_bounds = align_scene_and_compute_ceiling(
            self.plane_flipped
        )

        # 2) ⭐ 메트릭 스케일 재적용
        if self.metric_scale != 1.0:
            scene_aligned.apply_scale(self.metric_scale)
            print(f"[INFO] re-apply metric_scale = {self.metric_scale:.6f}")
            try:
                scaled_bounds = scene_aligned.bounds
            except Exception:
                scaled_bounds = aligned_bounds
            auto_ceiling_height_scaled = auto_ceiling_height * self.metric_scale
        else:
            scaled_bounds = aligned_bounds
            auto_ceiling_height_scaled = auto_ceiling_height

        # 3) 서브천장 높이
        ratio = 0.95
        sub_height = auto_ceiling_height_scaled * ratio
        print(f"[INFO] sub-ceiling height = {sub_height:.3f} (ratio={ratio})")

        # 4) 정렬된 좌표계에서 점 + 색 추출 후, 서브천장 기준으로 분할
        pts, cols = scene_to_points_with_colors(scene_aligned)
        z_pts = pts[:, 2]

        mask_keep = z_pts <= sub_height   # 🔥 항상 '아래' 점만 사용
        num_keep = int(mask_keep.sum())
        num_drop = int((~mask_keep).sum())

        print(f"[INFO] points kept (below)={num_keep}, dropped (above)={num_drop}")

        if num_keep == 0:
            print("[WARN] 서브천장 아래 점이 없습니다. 전체 점을 그대로 사용합니다.")
            pts_sel = pts
            cols_sel = cols
        else:
            pts_sel = pts[mask_keep]
            cols_sel = cols[mask_keep]

        # 5) 최종 Scene: 선택된 점(PointCloud) + 바닥 plane
        final_scene = trimesh.Scene()

        pc = trimesh.points.PointCloud(pts_sel, colors=cols_sel)
        final_scene.add_geometry(pc)

        # 바닥 plane 은 "정렬된 전체 방 bounds" 기준으로 생성
        add_floor_plane(final_scene, bounds=scaled_bounds)

        # 6) 가상 자 월드 좌표 보장 + 3D 박스 추가(ON 일 때만)
        try:
            self._ensure_ruler_world_default(scaled_bounds)
            if self.ruler_enabled and self.ruler_world_p1 is not None and self.ruler_world_p2 is not None:
                add_virtual_ruler(
                    final_scene,
                    self.ruler_world_p1,
                    self.ruler_world_p2,
                    thickness=0.03
                )
        except Exception as e:
            print("[WARN] 가상 자 처리 중 오류:", e)

        # 디버깅용: 선택된 점들의 z-range
        try:
            z_min_sel = float(np.quantile(pts_sel[:, 2], 0.01))
            z_max_sel = float(np.quantile(pts_sel[:, 2], 0.99))
            print(
                f"[DEBUG] selected points final z-range≈[{z_min_sel:.3f}, {z_max_sel:.3f}]"
            )
        except Exception:
            pass

        self.scene_plane = final_scene

    # ----------------- 템플릿 로딩 -----------------
    def _load_templates_from_dir(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        tpl_dir = os.path.join(base_dir, "furniture_templates")
        os.makedirs(tpl_dir, exist_ok=True)

        templates = {}
        for fname in os.listdir(tpl_dir):
            if not fname.lower().endswith(".glb"):
                continue
            path = os.path.join(tpl_dir, fname)
            name = os.path.splitext(fname)[0]

            try:
                mesh_or_scene = trimesh.load(path)
                if isinstance(mesh_or_scene, trimesh.Scene):
                    mesh = mesh_or_scene.dump(concatenate=True)
                else:
                    mesh = mesh_or_scene

                eps = 1e-6
                if abs(TEMPLATE_ROT_X_DEG) > eps:
                    mesh.apply_transform(
                        rotation_matrix(np.deg2rad(TEMPLATE_ROT_X_DEG), [1, 0, 0])
                    )
                if abs(TEMPLATE_ROT_Y_DEG) > eps:
                    mesh.apply_transform(
                        rotation_matrix(np.deg2rad(TEMPLATE_ROT_Y_DEG), [0, 1, 0])
                    )
                if abs(TEMPLATE_ROT_Z_DEG) > eps:
                    mesh.apply_transform(
                        rotation_matrix(np.deg2rad(TEMPLATE_ROT_Z_DEG), [0, 0, 1])
                    )

                bmin, bmax = mesh.bounds
                center_x = (bmin[0] + bmax[0]) * 0.5
                center_y = (bmin[1] + bmax[1]) * 0.5
                min_z = bmin[2]

                T = np.eye(4)
                T[:3, 3] = [-center_x, -center_y, -min_z]
                mesh.apply_transform(T)

                bmin2, bmax2 = mesh.bounds
                footprint_w = float(bmax2[0] - bmin2[0])
                footprint_d = float(bmax2[1] - bmin2[1])
                height = float(bmax2[2] - bmin2[2])

                templates[name] = {
                    "mesh": mesh,
                    "footprint_w": footprint_w,
                    "footprint_d": footprint_d,
                    "height": height,
                }
                print(f"[INFO] 템플릿 로드: {name} (w={footprint_w:.3f}, d={footprint_d:.3f}, h={height:.3f})")
            except Exception as e:
                print(f"[경고] 템플릿 로딩 실패: {fname} -> {e}")

        self.templates = templates
        self._update_template_combo()

        if not self.templates:
            messagebox.showwarning(
                "템플릿 없음",
                "furniture_templates 디렉터리에 .glb 가구 템플릿 파일을 넣어주세요.",
            )

    def _update_template_combo(self):
        names = sorted(self.templates.keys())
        self.tpl_combo["values"] = names
        if names:
            self.template_var.set(names[0])
        else:
            self.template_var.set("")

    # ----------------- 초기 렌더링 -----------------
    def _initial_render(self):
        self._render_snapshot_and_set_camera()

    # ----------------- 가상 자 2D 오버레이 삭제/그리기 -----------------
    def _clear_ruler_overlay(self):
        for attr in [
            "ruler_p1_id",
            "ruler_p2_id",
            "ruler_line_id",
            "ruler_label_bg_id",
            "ruler_label_text_id",
        ]:
            cid = getattr(self, attr, None)
            if cid is not None:
                try:
                    self.canvas.delete(cid)
                except Exception:
                    pass
                setattr(self, attr, None)

    def _draw_ruler_overlay(self):
        """
        스냅샷 위에 2D 가상 자(라인 + 양 끝점 + 길이 텍스트)를 그림.
        """
        self._clear_ruler_overlay()

        if not self.ruler_enabled:
            return
        if self.ruler_world_p1 is None or self.ruler_world_p2 is None:
            return
        if self.snapshot_img is None:
            return

        px1, py1 = self.world_to_canvas(self.ruler_world_p1[0], self.ruler_world_p1[1])
        px2, py2 = self.world_to_canvas(self.ruler_world_p2[0], self.ruler_world_p2[1])

        # 라인
        self.ruler_line_id = self.canvas.create_line(
            px1, py1, px2, py2, fill="yellow", width=4
        )

        # 양 끝점(드래그용)
        r = 6
        self.ruler_p1_id = self.canvas.create_oval(
            px1 - r, py1 - r, px1 + r, py1 + r,
            fill="yellow", outline="black", width=1
        )
        self.ruler_p2_id = self.canvas.create_oval(
            px2 - r, py2 - r, px2 + r, py2 + r,
            fill="yellow", outline="black", width=1
        )

        # 길이 텍스트
        mx = (px1 + px2) * 0.5
        my = (py1 + py2) * 0.5 - 15
        label = f"{self.ruler_length_m:.2f} m"

        self.ruler_label_bg_id = self.canvas.create_rectangle(
            mx - 35, my - 10, mx + 35, my + 4,
            fill="black", outline=""
        )
        self.ruler_label_text_id = self.canvas.create_text(
            mx, my,
            text=label,
            fill="white",
            font=("Arial", 9)
        )

    # ----------------- 카메라 설정 + 스냅샷 -----------------
    def _render_snapshot_and_set_camera(self):
        if self.scene_plane is None:
            return

        self.update_idletasks()
        w = max(self.canvas.winfo_width(), 400)
        h = max(self.canvas.winfo_height(), 400)
        self.img_w, self.img_h = w, h

        bmin, bmax = self.scene_plane.bounds
        cx = (bmin[0] + bmax[0]) * 0.5
        cy = (bmin[1] + bmax[1]) * 0.5
        self.cam_center = (cx, cy)

        z_top = float(bmax[2])
        z_range = float(bmax[2] - bmin[2])
        z_cam = z_top + z_range * 2.0 + 1.0
        self.cam_z = z_cam

        fov_x_deg = 60.0
        fov_x_rad = np.deg2rad(fov_x_deg)
        aspect = h / float(w)
        fov_y_rad = 2.0 * np.arctan(np.tan(fov_x_rad / 2.0) * aspect)

        self.fov_x_rad = fov_x_rad
        self.fov_y_rad = fov_y_rad

        camera = trimesh.scene.cameras.Camera(
            resolution=(w, h),
            fov=(np.rad2deg(fov_x_rad), np.rad2deg(fov_y_rad)),
        )

        scene = self.scene_plane.copy()
        scene.camera = camera

        cam_T = np.eye(4)
        cam_T[:3, 3] = [cx, cy, z_cam]
        scene.camera_transform = cam_T

        try:
            png_bytes = scene.save_image(resolution=(w, h), visible=False)
            if not png_bytes:
                raise RuntimeError("save_image() returned empty")

            from io import BytesIO
            img = Image.open(BytesIO(png_bytes))
            self.snapshot_img = ImageTk.PhotoImage(img)

            self.canvas.delete("all")
            self.canvas.create_image(w // 2, h // 2, image=self.snapshot_img)

            # 🔹 가상 자 2D 오버레이
            self._draw_ruler_overlay()
        except Exception as e:
            print("[경고] 스냅샷 렌더링 실패:", e)
            self.canvas.delete("all")
            self.canvas.create_text(
                w // 2, h // 2,
                text=f"스냅샷 렌더링 실패:\n{e}",
                fill="white", font=("Arial", 12), justify="center"
            )
        finally:
            self._schedule_live_preview_refresh()

    # ----------------- 월드(x,y,z=0) ↔ 캔버스(px,py) -----------------
    def world_to_canvas(self, x, y):
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

    # ----------------- 가상 자 히트 테스트 -----------------
    def _hit_test_ruler_point(self, px, py, thresh=10):
        """
        캔버스 좌표(px,py)가 가상 자 끝점 근처인지 판정.
        근처면 "p1" 또는 "p2" 반환, 아니면 None.
        """
        if not self.ruler_enabled:
            return None
        if self.ruler_world_p1 is None or self.ruler_world_p2 is None:
            return None

        p1x, p1y = self.world_to_canvas(self.ruler_world_p1[0], self.ruler_world_p1[1])
        p2x, p2y = self.world_to_canvas(self.ruler_world_p2[0], self.ruler_world_p2[1])

        if (px - p1x) ** 2 + (py - p1y) ** 2 <= thresh ** 2:
            return "p1"
        if (px - p2x) ** 2 + (py - p2y) ** 2 <= thresh ** 2:
            return "p2"
        return None

    # ----------------- 가구 footprint 폴리곤 계산 -----------------
    def _ensure_target_size(self, info):
        """target_w, target_d 없으면 템플릿 기본 크기로 초기화"""
        if info.get("target_w") is not None and info.get("target_d") is not None:
            return

        tpl = self.templates.get(info["template"])
        if tpl is None:
            info["target_w"] = 1.0
            info["target_d"] = 1.0
            return

        info["target_w"] = tpl["footprint_w"]
        info["target_d"] = tpl["footprint_d"]

    def _compute_polygon_points_px(self, info):
        template_name = info["template"]
        if template_name not in self.templates:
            return []

        self._ensure_target_size(info)

        # 최종 가구 크기 = target * scale
        w_w = info["target_w"] * info["scale"]
        h_w = info["target_d"] * info["scale"]

        cx_w = info["world_x"]
        cy_w = info["world_y"]
        yaw = np.deg2rad(info.get("yaw_deg", 0.0))

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
        if not pts:
            return
        canvas_id = self.canvas.create_polygon(
            pts,
            fill="#ffcc66",
            outline="#cc8800",
            width=2,
            tags=("furniture",)
        )
        info["canvas_id"] = canvas_id
        self._update_furniture_outline()

    def _redraw_furniture(self, index):
        info = self.furniture_items[index]
        pts = self._compute_polygon_points_px(info)
        if not pts:
            return
        self.canvas.coords(info["canvas_id"], *pts)
        self._update_furniture_outline()

    def _update_furniture_outline(self):
        for idx, info in enumerate(self.furniture_items):
            cid = info["canvas_id"]
            if idx == self.selected_index:
                self.canvas.itemconfigure(cid, outline="#ff0000", width=3)
            else:
                self.canvas.itemconfigure(cid, outline="#cc8800", width=2)

        self._update_size_entries_for_selected()

    def _update_size_entries_for_selected(self):
        """선택된 가구의 현재 가로/세로 값을 입력창에 표시"""
        if self.selected_index is None:
            self.width_entry_var.set("")
            self.depth_entry_var.set("")
            return
        if self.selected_index < 0 or self.selected_index >= len(self.furniture_items):
            self.width_entry_var.set("")
            self.depth_entry_var.set("")
            return

        info = self.furniture_items[self.selected_index]
        self._ensure_target_size(info)

        final_w = info["target_w"] * info["scale"]
        final_d = info["target_d"] * info["scale"]

        self.width_entry_var.set(f"{final_w:.3f}")
        self.depth_entry_var.set(f"{final_d:.3f}")

    # ----------------- 템플릿으로 가구 추가 -----------------
    def add_furniture_instance(self):
        if not self.templates:
            messagebox.showwarning("템플릿 없음", "먼저 furniture_templates 폴더에 glb 템플릿을 넣어주세요.")
            return
        tpl_name = self.template_var.get()
        if not tpl_name or tpl_name not in self.templates:
            messagebox.showwarning("템플릿 선택", "템플릿을 먼저 선택하세요.")
            return
        if self.scene_plane is None:
            return

        tpl = self.templates[tpl_name]

        bmin, bmax = self.scene_plane.bounds
        cx = (bmin[0] + bmax[0]) * 0.5
        cy = (bmin[1] + bmax[1]) * 0.5

        info = dict(
            template=tpl_name,
            world_x=cx,
            world_y=cy,
            yaw_deg=0.0,
            scale=1.0,
            target_w=tpl["footprint_w"],
            target_d=tpl["footprint_d"],
            canvas_id=None,
        )
        self.furniture_items.append(info)
        self._create_furniture_on_canvas(info)
        self.selected_index = len(self.furniture_items) - 1
        self.scale_var.set(1.0)
        self._update_furniture_outline()
        self._schedule_live_preview_refresh()

    # ----------------- 선택 가구 삭제 -----------------
    def delete_selected_furniture(self):
        if self.selected_index is None:
            return
        if self.selected_index < 0 or self.selected_index >= len(self.furniture_items):
            return

        info = self.furniture_items[self.selected_index]
        cid = info["canvas_id"]
        self.canvas.delete(cid)

        del self.furniture_items[self.selected_index]

        if not self.furniture_items:
            self.selected_index = None
            self.scale_var.set(1.0)
            self.width_entry_var.set("")
            self.depth_entry_var.set("")
        else:
            new_idx = min(self.selected_index, len(self.furniture_items) - 1)
            self.selected_index = new_idx
            self.scale_var.set(self.furniture_items[new_idx]["scale"])

        self._update_furniture_outline()
        self._schedule_live_preview_refresh()

    # ----------------- 배치 모드/드래그 -----------------
    def toggle_placement_mode(self):
        if self.measure_mode:
            messagebox.showinfo("알림", "거리 설정 모드에서는 가구를 움직일 수 없습니다.")
            return

        self.placement_mode = not self.placement_mode
        self.btn_place.config(
            text="가구 배치 모드: ON" if self.placement_mode else "가구 배치 모드: OFF"
        )
        self.drag_target = None
        self.drag_index = None
        self.drag_last_px = None
        self.ruler_dragging_point = None

    # ----------------- 가상 자 ON/OFF -----------------
    def toggle_ruler(self):
        """
        ✅ 수정 포인트:
        - 예전에는 여기서 _render_snapshot_and_set_camera() 를 다시 불러서
          Trimesh가 새로 이미지를 뽑다가 흰 화면 나오는 문제가 있었음.
        - 이제는 '기존 스냅샷은 그대로 유지'하고,
          오버레이(라인/동그라미/텍스트)만 다시 그린다.
        """
        self.ruler_enabled = not self.ruler_enabled

        if self.ruler_enabled:
            self.btn_ruler.config(text="가상 자: ON")
            if self.scene_plane is not None:
                bmin, bmax = self.scene_plane.bounds
                self._ensure_ruler_world_default((bmin, bmax))
        else:
            self.btn_ruler.config(text="가상 자: OFF")

        # 스냅샷은 그대로 두고, 2D 오버레이만 갱신
        self._draw_ruler_overlay()

        # 가구 외곽선은 그대로 유지
        self._update_furniture_outline()

        # 3D 뷰어에는 새 scene(자 포함/제외)을 보내도록 예약
        self._schedule_live_preview_refresh()

    def _find_furniture_index_by_canvas_id(self, cid):
        for i, info in enumerate(self.furniture_items):
            if info["canvas_id"] == cid:
                return i
        return None

    def on_mouse_down(self, event):
        # 🔹 거리 설정 모드일 때는 거리 측정 처리
        if self.measure_mode:
            self._handle_measure_click(event)
            return

        # 🔹 가상 자 끝점 드래그 시작 체크 (측정 모드가 아닐 때)
        if self.ruler_enabled:
            hit = self._hit_test_ruler_point(event.x, event.y)
            if hit is not None:
                self.ruler_dragging_point = hit
                return

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
                self.selected_index = idx
                if idx is not None:
                    self.scale_var.set(self.furniture_items[idx]["scale"])
                self._update_furniture_outline()
                break

    def on_mouse_move(self, event):
        # 🔹 가상 자 드래그 중이면 우선 처리
        if (
            self.ruler_dragging_point is not None
            and not self.measure_mode
            and self.ruler_enabled
        ):
            wx, wy = self.canvas_to_world(event.x, event.y)
            if self.ruler_dragging_point == "p1" and self.ruler_world_p1 is not None:
                z = float(self.ruler_world_p1[2]) if self.ruler_world_p1.shape[0] > 2 else 0.0
                self.ruler_world_p1 = np.array([wx, wy, z], dtype=float)
            elif self.ruler_dragging_point == "p2" and self.ruler_world_p2 is not None:
                z = float(self.ruler_world_p2[2]) if self.ruler_world_p2.shape[0] > 2 else 0.0
                self.ruler_world_p2 = np.array([wx, wy, z], dtype=float)

            # 길이 재계산
            if self.ruler_world_p1 is not None and self.ruler_world_p2 is not None:
                v = self.ruler_world_p2 - self.ruler_world_p1
                self.ruler_length_m = float(np.linalg.norm(v))

            # 2D 오버레이 갱신 + 3D 미리보기 갱신 예약
            self._draw_ruler_overlay()
            self._schedule_live_preview_refresh()
            return

        if self.measure_mode:
            return

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
        self._schedule_live_preview_refresh()

    def on_mouse_up(self, event):
        # 가상 자 드래그 끝
        if self.ruler_dragging_point is not None:
            self.ruler_dragging_point = None
            return

        if self.measure_mode:
            return

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

        xs = coords[0::2]
        ys = coords[1::2]
        cx_px = sum(xs) / len(xs)
        cy_px = sum(ys) / len(ys)
        wx, wy = self.canvas_to_world(cx_px, cy_px)

        info = self.furniture_items[self.drag_index]
        info["world_x"] = wx
        info["world_y"] = wy

        self.drag_target = None
        self.drag_index = None
        self.drag_last_px = None
        self._schedule_live_preview_refresh()

    # ----------------- 거리 측정용 클릭 처리 -----------------
    def _handle_measure_click(self, event):
        px, py = event.x, event.y
        wx, wy = self.canvas_to_world(px, py)

        # 캔버스에 포인트 찍기
        r = 5
        pid = self.canvas.create_oval(
            px - r, py - r, px + r, py + r,
            fill="cyan", outline="white", width=2
        )
        self.measure_point_ids.append(pid)
        self.measure_points_canvas.append((px, py))
        self.measure_points_world.append((wx, wy))

        # 두 번째 포인트면 선 그리기 + 거리 계산 + 스케일 설정
        if len(self.measure_points_world) == 2:
            (px1, py1), (px2, py2) = self.measure_points_canvas
            if self.measure_line_id is not None:
                self.canvas.delete(self.measure_line_id)
            self.measure_line_id = self.canvas.create_line(
                px1, py1, px2, py2, fill="lime", width=3
            )

            p1 = np.array(self.measure_points_world[0])
            p2 = np.array(self.measure_points_world[1])
            dist_scene = float(np.linalg.norm(p2 - p1))

            messagebox.showinfo(
                "거리 측정",
                f"현재 포인트클라우드 상 거리: {dist_scene:.3f} (scene 단위)"
            )

            real_m = simpledialog.askfloat(
                "실제 거리 입력",
                f"두 점 사이의 실제 거리를 미터(m) 단위로 입력하세요.\n"
                f"(scene 거리 = {dist_scene:.3f})"
            )
            if real_m is None or real_m <= 0:
                messagebox.showerror("오류", "0보다 큰 값을 입력해야 합니다.")
                return

            self.measure_real_dist_m = real_m
            scale = real_m / dist_scene
            print(f"[INFO] metric scale = {scale:.6f}")

            # ⭐ 전역 메트릭 스케일 누적
            self.metric_scale *= scale

            # 현재 씬/가구 좌표에 스케일 즉시 반영
            self.scene_plane.apply_scale(scale)

            for info in self.furniture_items:
                info["world_x"] *= scale
                info["world_y"] *= scale

            self.measure_points_world = [
                (p1[0] * scale, p1[1] * scale),
                (p2[0] * scale, p2[1] * scale),
            ]

            # 가상 자는 새 스케일 기준으로 다시 생성하도록 초기화
            self.ruler_world_p1 = None
            self.ruler_world_p2 = None
            self.ruler_dragging_point = None

            messagebox.showinfo(
                "스케일 적용 완료",
                f"스케일 factor = {scale:.4f} 가 적용되었습니다.\n"
                f"3D 뷰어에서 해당 선을 확인할 수 있습니다."
            )

            # 스냅샷/가구 다시 렌더 (이 과정에서 가상 자도 새 길이로 재생성됨)
            self._load_and_align_scene()
            self._render_snapshot_and_set_camera()
            for idx in range(len(self.furniture_items)):
                self._redraw_furniture(idx)

            # 거리 설정 모드 종료
            self._schedule_live_preview_refresh()
            self.toggle_measure_mode()

    # ----------------- 3D 씬 구성 (방 + 가구 + 가상자) -----------------
    def _build_scene_with_furniture(self):
        if self.scene_plane is None:
            return None

        scene = self.scene_plane.copy()

        # 가상 자 3D 표시 (ON 일 때만)
        if (
            self.ruler_enabled
            and self.ruler_world_p1 is not None
            and self.ruler_world_p2 is not None
        ):
            add_virtual_ruler(
                scene,
                self.ruler_world_p1,
                self.ruler_world_p2,
                thickness=0.03
            )

        if not self.templates:
            return scene

        global_scale = self.furniture_world_scale_3d

        for info in self.furniture_items:
            tpl_name = info["template"]
            if tpl_name not in self.templates:
                continue
            tpl = self.templates[tpl_name]

            self._ensure_target_size(info)

            wx = info["world_x"]
            wy = info["world_y"]
            yaw_deg = info.get("yaw_deg", 0.0)
            s = info.get("scale", 1.0)

            w0 = tpl["footprint_w"]
            d0 = tpl["footprint_d"]
            target_w = info["target_w"]
            target_d = info["target_d"]

            final_w = target_w * s * global_scale
            final_d = target_d * s * global_scale

            if w0 <= 1e-6 or d0 <= 1e-6:
                sx = s * global_scale
                sy = s * global_scale
            else:
                sx = final_w / w0
                sy = final_d / d0
            sz = (sx + sy) * 0.5

            base_mesh = tpl["mesh"]
            mesh = base_mesh.copy()

            T_scale = np.eye(4)
            T_scale[0, 0] = sx
            T_scale[1, 1] = sy
            T_scale[2, 2] = sz
            mesh.apply_transform(T_scale)

            mesh.apply_transform(rotation_matrix(np.deg2rad(yaw_deg), [0, 0, 1]))
            mesh.apply_translation([wx, wy, 0.0])

            scene.add_geometry(mesh)

        return scene

    # ----------------- 회전 -----------------
    def rotate_selected(self, delta_deg):
        if self.selected_index is None:
            return
        if self.selected_index < 0 or self.selected_index >= len(self.furniture_items):
            return

        info = self.furniture_items[self.selected_index]
        info["yaw_deg"] = (info.get("yaw_deg", 0.0) + delta_deg) % 360.0
        self._redraw_furniture(self.selected_index)
        self._schedule_live_preview_refresh()

    def start_rotate(self, delta_deg):
        # 기존 회전 예약 취소
        self.stop_rotate()
        self._rotate_delta = delta_deg
        # 탭 시 1도만 바로 적용
        self.rotate_selected(delta_deg)
        # 일정 시간 꾹 누르면 연속 회전 시작
        self._rotate_delay_job = self.after(250, self._start_rotate_loop)

    def _start_rotate_loop(self):
        self._rotate_delay_job = None
        if self._rotate_delta == 0:
            return
        self._schedule_rotate_repeat()

    def _schedule_rotate_repeat(self):
        if self._rotate_delta == 0:
            return
        self._rotate_job = self.after(40, self._rotate_repeat)

    def _rotate_repeat(self):
        if self._rotate_delta == 0:
            self._rotate_job = None
            return
        self.rotate_selected(self._rotate_delta)
        self._schedule_rotate_repeat()

    def stop_rotate(self):
        self._rotate_delta = 0
        if self._rotate_delay_job is not None:
            try:
                self.after_cancel(self._rotate_delay_job)
            except Exception:
                pass
            self._rotate_delay_job = None
        if self._rotate_job is not None:
            try:
                self.after_cancel(self._rotate_job)
            except Exception:
                pass
            self._rotate_job = None

    # ----------------- 스케일 변경 (비율) -----------------
    def on_scale_change(self, _value):
        if self.selected_index is None:
            return
        if self.selected_index < 0 or self.selected_index >= len(self.furniture_items):
            return

        s = float(self.scale_var.get())
        info = self.furniture_items[self.selected_index]
        info["scale"] = s
        self._redraw_furniture(self.selected_index)
        self._schedule_live_preview_refresh()

    # ----------------- 가구 크기 직접 지정 -----------------
    def apply_furniture_size(self):
        """입력된 가로/세로 값을 기준으로 target_w, target_d를 강제 조정"""
        if self.selected_index is None:
            messagebox.showinfo("알림", "먼저 가구를 선택하세요.")
            return
        if self.selected_index < 0 or self.selected_index >= len(self.furniture_items):
            return

        info = self.furniture_items[self.selected_index]
        self._ensure_target_size(info)

        try:
            w_val = float(self.width_entry_var.get())
            d_val = float(self.depth_entry_var.get())
        except ValueError:
            messagebox.showerror("오류", "가로/세로를 숫자로 입력하세요.")
            return

        if w_val <= 0 or d_val <= 0:
            messagebox.showerror("오류", "가로/세로는 0보다 커야 합니다.")
            return

        scale = info["scale"] if info["scale"] > 0 else 1.0
        info["target_w"] = w_val / scale
        info["target_d"] = d_val / scale

        self._redraw_furniture(self.selected_index)
        self._schedule_live_preview_refresh()
        messagebox.showinfo(
            "가구 크기 적용",
            f"선택 가구 크기를 가로 {w_val:.3f}, 세로 {d_val:.3f} 로 설정했습니다.\n"
            "이후 스케일 슬라이더를 조정하면 이 크기를 기준으로 전체 비율이 변경됩니다."
        )

    # ----------------- 3D 팝업 (버튼 눌렀을 때만) -----------------
    def show_3d_popup(self):
        if self.scene_plane is None:
            return
        if not self.templates:
            messagebox.showwarning("템플릿 없음", "먼저 furniture_templates 폴더에 템플릿을 넣어주세요.")
            return

        scene = self._build_scene_with_furniture()
        if scene is None:
            return

        try:
            scene.show()
        except Exception as e:
            messagebox.showerror("3D 보기 오류", str(e))

    # ----------------- 실시간 3D 미리보기 -----------------
    def toggle_live_preview(self):
        self.live_preview_on = not self.live_preview_on
        if self.live_preview_on:
            self.btn_live3d.config(text="실시간 3D 미리보기: ON")
            # 뷰어 위치를 메인 창 오른쪽에 붙여서 표시
            try:
                x = self.winfo_rootx() + self.winfo_width() + 20
                y = self.winfo_rooty()
                self.live_view_init_loc = (x, y)
            except Exception:
                self.live_view_init_loc = None
            self._start_live_viewer_thread()
            self._schedule_live_preview_refresh(0)
        else:
            self.btn_live3d.config(text="실시간 3D 미리보기: OFF")
            if self.live_preview_job is not None:
                self.after_cancel(self.live_preview_job)
                self.live_preview_job = None
            self._send_live_view_command("close")

    def _schedule_live_preview_refresh(self, delay_ms=None):
        if not self.live_preview_on:
            return
        if delay_ms is None:
            delay_ms = self.live_preview_delay_ms
        # 마지막 요청만 유효하도록 기존 예약을 취소하고 다시 예약 (디바운스)
        if self.live_preview_job is not None:
            try:
                self.after_cancel(self.live_preview_job)
            except Exception:
                pass
            self.live_preview_job = None
        self.live_preview_job = self.after(delay_ms, self.refresh_live_preview)

    def refresh_live_preview(self):
        if not self.live_preview_on:
            self.live_preview_job = None
            return
        self.live_preview_job = None
        scene = self._build_scene_with_furniture()
        if scene is None:
            return

        self._send_live_view_command(scene)

    # ----------------- pyglet 기반 라이브 뷰어 -----------------
    def _start_live_viewer_thread(self):
        if self.live_view_thread is not None and self.live_view_thread.is_alive():
            return
        self.live_view_queue = queue.Queue()

        def worker():
            import pyglet
            try:
                initial_scene = self._build_scene_with_furniture()
                if initial_scene is None:
                    return
                viewer = trimesh.viewer.SceneViewer(
                    initial_scene,
                    start_loop=False,
                    caption="실시간 3D 뷰어",
                    resizable=True,
                )
                # 화면 왼쪽에 크게 띄우기
                try:
                    sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
                    viewer.set_location(0, 0)
                    viewer.set_size(max(int(sw * 0.5), 800), max(int(sh * 0.9), 700))
                except Exception:
                    if self.live_view_init_loc is not None:
                        try:
                            viewer.set_location(*self.live_view_init_loc)
                        except Exception:
                            pass
                self.live_viewer = viewer

                def pump(_dt):
                    try:
                        while True:
                            cmd = self.live_view_queue.get_nowait()
                            if cmd == "close":
                                viewer.close()
                                pyglet.app.exit()
                                return
                            if isinstance(cmd, trimesh.Scene):
                                old_cam = None
                                if viewer.scene is not None:
                                    old_cam = np.array(viewer.scene.camera_transform)
                                viewer.scene = viewer._scene = cmd
                                if old_cam is not None and np.all(np.isfinite(old_cam)):
                                    viewer.scene.camera_transform = old_cam
                                    viewer._initial_camera_transform = old_cam.copy()
                                else:
                                    viewer.reset_view()
                                viewer._update_vertex_list()
                    except queue.Empty:
                        pass

                pyglet.clock.schedule_interval(pump, 1 / 30.0)
                pyglet.app.run()
            except Exception as e:
                print("[경고] 라이브 뷰어 스레드 오류:", e)
            finally:
                self.live_viewer = None

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        self.live_view_thread = th

    def _send_live_view_command(self, cmd):
        if self.live_view_queue is None:
            return
        try:
            self.live_view_queue.put_nowait(cmd)
        except Exception:
            pass


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    try:
        app = SnapshotFloorApp()
        app.mainloop()
    except Exception as e:
        print("[오류]", e)
        try:
            messagebox.showerror("오류", str(e))
        except Exception:
            pass

