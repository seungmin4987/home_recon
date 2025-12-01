import os
import locale
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import numpy as np
import cv2
import requests
import tempfile
import trimesh
from trimesh.geometry import plane_transform
import trimesh.transformations as tf


# ---- 한글 로케일 (가능 시) ----
try:
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
except locale.Error:
    pass


# ============================================================
# 🔎 평면 헤더 파싱 유틸
# ============================================================
def parse_plane_header(header_str):
    """
    서버 헤더 "Plane-Equation"에 들어온 문자열을 (a,b,c,d) float 튜플로 파싱
    예: "(0.0, 1.0, 0.0, -0.3)" 또는 "0.0,1.0,0.0,-0.3"
    """
    if not header_str:
        return None
    s = header_str.strip()
    s = s.strip("()[]")
    parts = s.replace(",", " ").split()
    if len(parts) < 4:
        return None
    try:
        a, b, c, d = [float(p) for p in parts[:4]]
        return (a, b, c, d)
    except ValueError:
        return None


# ============================================================
# 🪑 가구 데이터 클래스
# ============================================================
class Furniture:
    def __init__(self, cx, cy, size_x, size_y, height, angle_deg=0.0):
        # 월드 좌표계 (z=0 상 평면 기준)
        self.cx = cx
        self.cy = cy
        self.size_x = size_x
        self.size_y = size_y
        self.height = height
        self.angle_deg = angle_deg


# ============================================================
# 🧭 가구 배치 시뮬레이터 (2D + 3D)
# ============================================================
class FurnitureSimulator:
    def __init__(self, scene):
        """
        scene: plane_transform까지 적용된, 바닥이 z=0으로 정렬된 trimesh.Scene
        """
        self.scene_base = scene
        self.furnitures = []
        self.selected_idx = None

        # 스케일: 1 unit = meter_per_unit (m)
        self.meter_per_unit = 1.0

        # 2D 캔버스 설정
        self.canvas_size = 700
        self.margin = 20

        # 스케일 기준선 관련 상태
        self.scale_mode = 0  # 0: normal, 1: 첫 점, 2: 둘째 점
        self.scale_p1 = None
        self.scale_p2 = None

        # 가구 추가 모드 여부
        self.add_mode = False

        # 장면 bounds 기반 2D 매핑 초기화
        self._compute_bounds()

        # Tk 윈도우 생성
        self.win = tk.Toplevel()
        self.win.title("가구 배치 시뮬레이터")
        self._build_ui()
        self._draw_canvas()

    # ---------- 장면 bounds ----------
    def _compute_bounds(self):
        try:
            bounds = self.scene_base.bounds  # (2,3)
            min_b, max_b = bounds
        except Exception:
            min_b = np.array([-1, -1, 0], dtype=float)
            max_b = np.array([1, 1, 1], dtype=float)

        self.min_x, self.min_y = float(min_b[0]), float(min_b[1])
        self.max_x, self.max_y = float(max_b[0]), float(max_b[1])

        if self.max_x <= self.min_x:
            self.max_x = self.min_x + 1.0
        if self.max_y <= self.min_y:
            self.max_y = self.min_y + 1.0

        width = self.max_x - self.min_x
        height = self.max_y - self.min_y

        usable = self.canvas_size - 2 * self.margin
        sx = usable / width
        sy = usable / height
        self.scale2d = min(sx, sy)

    # ---------- 좌표 변환 ----------
    def world_to_canvas(self, x, y):
        u = self.margin + (x - self.min_x) * self.scale2d
        v = self.canvas_size - (self.margin + (y - self.min_y) * self.scale2d)
        return u, v

    def canvas_to_world(self, u, v):
        y = (self.canvas_size - v - self.margin) / self.scale2d + self.min_y
        x = (u - self.margin) / self.scale2d + self.min_x
        return x, y

    # ---------- UI ----------
    def _build_ui(self):
        main = ttk.Frame(self.win)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # 왼쪽: 2D 캔버스
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(
            left,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="white"
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # 오른쪽: 컨트롤 패널
        right = ttk.Frame(main)
        right.pack(side="left", fill="y", padx=(10, 0))

        self.info_label = ttk.Label(right, text="① 스케일 설정(선택) → ② 가구 크기 설정 → ③ 가구 추가/이동/회전")
        self.info_label.pack(pady=(0, 10))

        # 스케일 관련
        ttk.Label(right, text="[스케일 설정]").pack(anchor="w")
        ttk.Button(right, text="기준선 두 점 찍기", command=self.start_scale_mode).pack(fill="x", pady=2)
        self.scale_label = ttk.Label(right, text="현재: 1 unit = 1.0 m")
        self.scale_label.pack(anchor="w", pady=(0, 10))

        # 가구 크기 입력
        frame_size = ttk.LabelFrame(right, text="가구 크기 (미터)")
        frame_size.pack(fill="x", pady=5)

        ttk.Label(frame_size, text="가로 X (m):").grid(row=0, column=0, sticky="w")
        self.entry_w = ttk.Entry(frame_size, width=8)
        self.entry_w.grid(row=0, column=1, sticky="w")
        self.entry_w.insert(0, "1.0")

        ttk.Label(frame_size, text="세로 Y (m):").grid(row=1, column=0, sticky="w")
        self.entry_d = ttk.Entry(frame_size, width=8)
        self.entry_d.grid(row=1, column=1, sticky="w")
        self.entry_d.insert(0, "1.0")

        ttk.Label(frame_size, text="높이 Z (m):").grid(row=2, column=0, sticky="w")
        self.entry_h = ttk.Entry(frame_size, width=8)
        self.entry_h.grid(row=2, column=1, sticky="w")
        self.entry_h.insert(0, "0.8")

        # 가구 추가 / 3D 보기
        ttk.Button(right, text="가구 추가 모드", command=self.start_add_mode).pack(fill="x", pady=(10, 4))
        ttk.Button(right, text="3D 미리보기 열기", command=self.show_3d_preview).pack(fill="x", pady=(0, 10))

        # 회전 슬라이더
        frame_rot = ttk.LabelFrame(right, text="선택한 가구 회전 (deg)")
        frame_rot.pack(fill="x", pady=5)
        self.rot_var = tk.DoubleVar(value=0.0)
        self.rot_slider = ttk.Scale(frame_rot, from_=0, to=359, variable=self.rot_var, command=self.on_rotate_change)
        self.rot_slider.pack(fill="x", padx=4, pady=4)

        self.selected_label = ttk.Label(right, text="선택된 가구: 없음")
        self.selected_label.pack(anchor="w", pady=(5, 0))

    # ---------- 캔버스 그리기 ----------
    def _draw_canvas(self):
        self.canvas.delete("all")

        # 바닥 bounding box
        x0, y0 = self.world_to_canvas(self.min_x, self.min_y)
        x1, y1 = self.world_to_canvas(self.max_x, self.max_y)
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#cccccc")

        # 스케일 기준점 표시
        if self.scale_p1 is not None:
            u, v = self.world_to_canvas(*self.scale_p1)
            self.canvas.create_oval(u-3, v-3, u+3, v+3, fill="red")
        if self.scale_p2 is not None:
            u, v = self.world_to_canvas(*self.scale_p2)
            self.canvas.create_oval(u-3, v-3, u+3, v+3, fill="red")
            # 기준선
            u1, v1 = self.world_to_canvas(*self.scale_p1)
            self.canvas.create_line(u1, v1, u, v, fill="red", dash=(4, 2))

        # 가구 그리기
        for idx, f in enumerate(self.furnitures):
            self._draw_furniture(idx, f)

    def _draw_furniture(self, idx, f: Furniture):
        # 월드 좌표에서 사각형 모서리 4개(회전 포함)
        angle = np.deg2rad(f.angle_deg)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # 로컬 좌표 사각형 (중심 기준)
        hx = f.size_x / 2.0
        hy = f.size_y / 2.0
        corners_local = np.array([
            [-hx, -hy],
            [ hx, -hy],
            [ hx,  hy],
            [-hx,  hy],
        ])

        # 로컬 -> 월드
        rot = np.array([[cos_a, -sin_a],
                        [sin_a,  cos_a]])
        corners_world = (corners_local @ rot.T) + np.array([f.cx, f.cy])

        # 월드 -> 캔버스
        pts = []
        for x, y in corners_world:
            u, v = self.world_to_canvas(x, y)
            pts.extend([u, v])

        color = "#66aaee" if idx == self.selected_idx else "#4477cc"
        self.canvas.create_polygon(pts, fill=color, outline="black", width=1)

    # ---------- 이벤트 핸들러 ----------
    def on_canvas_click(self, event):
        wx, wy = self.canvas_to_world(event.x, event.y)

        # 스케일 기준선 지정 모드
        if self.scale_mode == 1:
            self.scale_p1 = (wx, wy)
            self.scale_p2 = None
            self.scale_mode = 2
            self.info_label.config(text="두 번째 기준점을 클릭하세요.")
            self._draw_canvas()
            return
        elif self.scale_mode == 2:
            self.scale_p2 = (wx, wy)
            self.scale_mode = 0
            self.info_label.config(text="기준선 거리를 입력해 스케일을 설정하세요.")
            self._draw_canvas()
            self._ask_scale_distance()
            return

        # 가구 추가 모드
        if self.add_mode:
            self.add_furniture_at(wx, wy)
            self.add_mode = False
            self.info_label.config(text="가구 추가 완료. 다른 가구를 추가하려면 다시 '가구 추가 모드' 클릭.")
            return

        # 일반 모드: 가구 선택
        clicked_idx = self._hit_test_furniture(wx, wy)
        self.selected_idx = clicked_idx
        if clicked_idx is not None:
            f = self.furnitures[clicked_idx]
            self.rot_var.set(f.angle_deg)
            self.selected_label.config(text=f"선택된 가구: #{clicked_idx} (각도 {f.angle_deg:.1f}°)")
        else:
            self.selected_label.config(text="선택된 가구: 없음")
        self._draw_canvas()

    def _hit_test_furniture(self, wx, wy):
        for idx, f in enumerate(self.furnitures):
            if self._point_in_furniture(wx, wy, f):
                return idx
        return None

    def _point_in_furniture(self, x, y, f: Furniture):
        # 월드 포인트 -> 가구 로컬 좌표계(중심+회전 반대)
        dx = x - f.cx
        dy = y - f.cy
        angle = -np.deg2rad(f.angle_deg)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        lx = cos_a * dx - sin_a * dy
        ly = sin_a * dx + cos_a * dy

        if abs(lx) <= f.size_x / 2.0 and abs(ly) <= f.size_y / 2.0:
            return True
        return False

    def start_scale_mode(self):
        self.scale_mode = 1
        self.scale_p1 = None
        self.scale_p2 = None
        self.info_label.config(text="첫 번째 기준점을 클릭하세요.")
        self._draw_canvas()

    def _ask_scale_distance(self):
        # 기준선 길이(모델 단위)
        if self.scale_p1 is None or self.scale_p2 is None:
            return
        p1 = np.array(self.scale_p1)
        p2 = np.array(self.scale_p2)
        dist_model = float(np.linalg.norm(p2 - p1))
        if dist_model < 1e-6:
            messagebox.showwarning("경고", "두 점이 너무 가깝습니다.")
            return

        real_m = simpledialog.askfloat(
            "스케일 설정",
            f"선택한 두 점 사이 거리가 실제로 몇 미터인가요?\n(모델 거리: {dist_model:.3f} units)",
            minvalue=0.01
        )
        if real_m is None:
            self.info_label.config(text="스케일 설정이 취소되었습니다.")
            return

        self.meter_per_unit = real_m / dist_model  # 1 unit = meter_per_unit 미터
        self.scale_label.config(text=f"현재: 1 unit ≈ {self.meter_per_unit:.3f} m")
        self.info_label.config(text="스케일이 설정되었습니다. 이제 가구 크기를 미터 단위로 입력할 수 있습니다.")

    def start_add_mode(self):
        self.add_mode = True
        self.info_label.config(text="가구를 놓을 위치를 2D 평면에서 클릭하세요.")

    def add_furniture_at(self, wx, wy):
        # 입력된 가구 크기를 미터 → 모델 단위로 변환
        try:
            w_m = float(self.entry_w.get())
            d_m = float(self.entry_d.get())
            h_m = float(self.entry_h.get())
        except ValueError:
            messagebox.showerror("오류", "가구 크기를 올바른 숫자로 입력하세요.")
            return

        if self.meter_per_unit <= 0:
            self.meter_per_unit = 1.0

        # 모델 단위
        w_u = w_m / self.meter_per_unit
        d_u = d_m / self.meter_per_unit
        h_u = h_m / self.meter_per_unit

        f = Furniture(wx, wy, w_u, d_u, h_u, angle_deg=0.0)
        self.furnitures.append(f)
        self.selected_idx = len(self.furnitures) - 1
        self.selected_label.config(text=f"선택된 가구: #{self.selected_idx}")
        self.rot_var.set(0.0)
        self._draw_canvas()

    def on_rotate_change(self, value):
        if self.selected_idx is None:
            return
        try:
            angle = float(value)
        except ValueError:
            return
        self.furnitures[self.selected_idx].angle_deg = angle
        self._draw_canvas()

    # ---------- 3D 미리보기 ----------
    def show_3d_preview(self):
        try:
            # 베이스 장면 복사
            base_scene = self.scene_base.copy()

            # 가구를 box mesh로 추가
            for f in self.furnitures:
                # extents: (size_x, size_y, height)
                box = trimesh.creation.box(extents=(f.size_x, f.size_y, f.height))

                # 회전 (z축 기준)
                angle_rad = np.deg2rad(f.angle_deg)
                R = tf.rotation_matrix(angle_rad, [0, 0, 1])

                # z=0에서 시작하도록 z축 방향으로 1/2 높이만큼 올림
                T = tf.translation_matrix([f.cx, f.cy, f.height / 2.0])

                M = tf.concatenate_matrices(R, T)
                box.apply_transform(M)

                # 색상
                color = np.array([220, 120, 120, 200], dtype=np.uint8)
                box.visual.vertex_colors = color

                base_scene.add_geometry(box)

            base_scene.show()

        except Exception as e:
            messagebox.showerror("3D 미리보기 오류", f"3D 장면 표시 중 오류:\n{e}")


# ============================================================
# 🌐 서버 전송 함수 (GLB 저장 + 바닥 정렬 + 시뮬레이터 실행)
# ============================================================
def upload_to_server(images, seg_mask, seg_image_path):
    SERVER_URL = "https://untribal-memorisingly-joanne.ngrok-free.dev/upload"  # 필요시 변경

    try:
        # 여러 이미지를 전송 리스트로 구성
        files = [("files", (os.path.basename(p), open(p, "rb"), "image/jpeg")) for p in images]

        # 🔹 세그를 수행한 원본 이미지 파일명 전달
        seg_name = os.path.basename(seg_image_path)
        data = {"seg_name": seg_name}

        # 세그멘테이션 마스크 임시 저장 후 추가
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, seg_mask)
            tmp_path = tmp.name
            files.append(("seg_image", (os.path.basename(tmp_path), open(tmp_path, "rb"), "image/png")))

        # 서버 POST 요청 (data 포함)
        res = requests.post(SERVER_URL, files=files, data=data)
        os.remove(tmp_path)

        if res.status_code != 200:
            messagebox.showerror("오류", f"서버 오류: {res.status_code}\n{res.text}")
            return

        # 🔹 GLB를 현재 스크립트 기준 폴더에 저장
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, "received_glb")
        os.makedirs(save_dir, exist_ok=True)

        out_path = os.path.join(save_dir, "received_model.glb")
        with open(out_path, "wb") as f:
            f.write(res.content)

        # 🔹 평면 방정식 파싱
        plane_str = res.headers.get("Plane-Equation", "")
        plane_eq = parse_plane_header(plane_str)

        msg = f"✅ GLB 파일 저장됨:\n{out_path}"
        if plane_eq is not None:
            a, b, c, d = plane_eq
            msg += f"\n\n📐 평면 방정식:\n{a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0"
        else:
            msg += "\n\n⚠️ 평면 방정식을 파싱하지 못했습니다."
        messagebox.showinfo("수신 완료", msg)

        # 🔹 GLB 로드 & 바닥 평면을 z=0으로 정렬
        mesh_or_scene = trimesh.load(out_path)

        # 항상 Scene 형태로 통일
        if isinstance(mesh_or_scene, trimesh.Scene):
            scene = mesh_or_scene
        else:
            scene = trimesh.Scene(mesh_or_scene)

        if plane_eq is not None:
            a, b, c, d = plane_eq
            n = np.array([a, b, c], dtype=np.float64)
            if np.linalg.norm(n) > 1e-6:
                origin = -d * n / (np.dot(n, n) + 1e-12)  # 평면 위 한 점
                T = plane_transform(origin, n)
                scene.apply_transform(T)

        # 필요하면 바닥 평면 메쉬 추가 (얇은 box)
        try:
            bounds = scene.bounds
            min_b, max_b = bounds
            size = max_b - min_b
            if not np.all(np.isfinite(size)):
                size = np.array([1.0, 1.0, 1.0])

            size_x = max(size[0], 1.0)
            size_y = max(size[1], 1.0)
            px = size_x * 1.2
            py = size_y * 1.2
            thickness = max(size_x, size_y) * 0.01

            center_x = (min_b[0] + max_b[0]) / 2.0
            center_y = (min_b[1] + max_b[1]) / 2.0

            plane_mesh = trimesh.creation.box(extents=(px, py, thickness))
            plane_mesh.apply_translation([center_x, center_y, -thickness / 2.0])
            plane_color = np.array([180, 230, 200, 100], dtype=np.uint8)
            plane_mesh.visual.vertex_colors = plane_color
            scene.add_geometry(plane_mesh)
        except Exception as e:
            print(f"⚠️ 평면 메쉬 추가 경고: {e}")

        # 🔹 가구 배치 시뮬레이터 실행 (새 Toplevel 창)
        FurnitureSimulator(scene)

    except Exception as e:
        messagebox.showerror("전송 실패", str(e))


# ============================================================
# 🎨 세그멘테이션 창
# ============================================================
class SegEditor(tk.Toplevel):
    def __init__(self, master, image_path, on_complete):
        super().__init__(master)
        self.title("바닥 영역 선택")
        self.image_path = image_path
        self.on_complete = on_complete
        self.result_mask = None
        self.preview_img = None

        self.img_bgr = cv2.imread(image_path)
        if self.img_bgr is None:
            messagebox.showerror("오류", f"이미지 로드 실패: {image_path}")
            self.destroy()
            return

        self.h0, self.w0 = self.img_bgr.shape[:2]
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        self.mask = np.full((self.h0, self.w0), cv2.GC_PR_BGD, np.uint8)

        self.scale = min(1.0, 1200 / self.w0, 800 / self.h0)
        self.disp_w, self.disp_h = int(self.w0 * self.scale), int(self.h0 * self.scale)
        self.brush = 14
        self.drawing = False

        self._build_ui()
        self._render_canvas()

    def _build_ui(self):
        ttk.Label(self, text="바닥 부분을 칠한 뒤, '바닥 영역 선택 완료' 버튼을 누르세요.").pack(pady=(8, 4))
        self.canvas = tk.Canvas(self, width=self.disp_w, height=self.disp_h, bg="#111")
        self.canvas.pack(padx=8, pady=(4, 10))
        self.canvas.bind("<ButtonPress-1>", self._on_down)
        self.canvas.bind("<B1-Motion>", self._on_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_up)
        ttk.Button(self, text="바닥 영역 선택 완료", command=self.finish_segmentation).pack(pady=(4, 8))

    def _canvas_to_image_xy(self, x, y):
        ix, iy = int(x / self.scale), int(y / self.scale)
        return np.clip(ix, 0, self.w0 - 1), np.clip(iy, 0, self.h0 - 1)

    def _on_down(self, e):
        self.drawing = True
        self._paint(e.x, e.y)

    def _on_move(self, e):
        if self.drawing:
            self._paint(e.x, e.y)

    def _on_up(self, e):
        self.drawing = False

    def _paint(self, x, y):
        ix, iy = self._canvas_to_image_xy(x, y)
        cv2.circle(self.mask, (ix, iy), self.brush, cv2.GC_FGD, -1)
        self._render_canvas()

    def _render_canvas(self):
        overlay = self.img_rgb.copy()
        overlay[self.mask == cv2.GC_FGD] = (
            overlay[self.mask == cv2.GC_FGD] * 0.5 + np.array([0, 120, 255]) * 0.5
        ).astype(np.uint8)

        disp = Image.fromarray(cv2.resize(overlay, (self.disp_w, self.disp_h)))
        if self.preview_img is not None:
            thumb = self.preview_img.resize((self.disp_w // 4, self.disp_h // 4))
            disp.paste(thumb, (self.disp_w - thumb.width - 8, self.disp_h - thumb.height - 8))
        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")

    def finish_segmentation(self):
        self.configure(cursor="watch")
        self.update_idletasks()
        try:
            # 🔹 빠른 GrabCut (다운샘플링 0.3)
            small_scale = 0.3
            small_bgr = cv2.resize(self.img_bgr, None, fx=small_scale, fy=small_scale)
            small_mask = cv2.resize(self.mask, (small_bgr.shape[1], small_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(small_bgr, small_mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
            mask_full = cv2.resize(small_mask, (self.w0, self.h0), interpolation=cv2.INTER_NEAREST)
            mask2 = np.where((mask_full == cv2.GC_FGD) | (mask_full == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
            self.result_mask = (mask2 * 255).astype(np.uint8)

            preview = self.img_rgb.copy()
            preview[mask2 == 1] = (preview[mask2 == 1] * 0.6 + np.array([80, 255, 120]) * 0.4).astype(np.uint8)
            self.preview_img = Image.fromarray(preview)
            self._render_canvas()

            messagebox.showinfo("확인", "바닥 영역 미리보기를 확인했습니다.")
            self.on_complete(self.result_mask)
            self.destroy()

        except Exception as e:
            messagebox.showerror("오류", str(e))
        finally:
            self.configure(cursor="")


# ============================================================
# 🪶 메인 GUI (드래그앤드롭 + 썸네일 선택)
# ============================================================
class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("바닥 영역 기반 3D 리컨스트럭션 클라이언트")
        self.geometry("1000x600")
        self.configure(bg="#222")

        self.images = []
        self.labels = []
        self.selected = None          # 세그 할 이미지 경로
        self.seg_mask = None          # 세그 결과 마스크

        ttk.Label(self, text="이미지를 드래그앤드롭 하세요.", background="#222", foreground="white").pack(pady=(12, 4))
        self.drop_area = tk.Label(self, text="Drop Images Here", bg="#333", fg="white", width=80, height=6)
        self.drop_area.pack(pady=6)
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind("<<Drop>>", self.on_drop)

        ttk.Label(self, text="썸네일 중 하나를 클릭해 세그멘테이션 대상으로 선택하세요.", background="#222", foreground="white").pack()

        self.thumb_frame = ttk.Frame(self)
        self.thumb_frame.pack(padx=10, pady=(8, 12))

        self.seg_btn = ttk.Button(self, text="바닥 영역 선택하기", command=self.start_segmentation, state="disabled")
        self.seg_btn.pack(pady=(8, 6))
        self.reconstruct_btn = ttk.Button(self, text="3D 리컨스트럭션 수행", command=self.run_reconstruction, state="disabled")
        self.reconstruct_btn.pack(pady=(4, 10))

    def on_drop(self, event):
        files = self.tk.splitlist(event.data)
        valid = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not valid:
            messagebox.showwarning("알림", "이미지 파일만 드래그하세요.")
            return
        self.images = valid
        self.show_thumbnails()
        self.seg_btn.configure(state="normal")

    def show_thumbnails(self):
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        self.labels.clear()
        cols = 5
        r = c = 0
        for path in self.images:
            im = Image.open(path)
            im.thumbnail((160, 160))
            tkim = ImageTk.PhotoImage(im)
            lb = tk.Label(self.thumb_frame, image=tkim, relief="solid", bd=2)
            lb.image = tkim
            lb.grid(row=r, column=c, padx=5, pady=5)
            lb.bind("<Button-1>", lambda e, p=path, l=lb: self.select_one(p, l))
            self.labels.append(lb)
            c += 1
            if c == cols:
                c = 0
                r += 1

    def select_one(self, path, label):
        for lb in self.labels:
            lb.config(highlightthickness=0)
        label.config(highlightbackground="green", highlightthickness=3)
        self.selected = path
        messagebox.showinfo("선택됨", f"선택한 이미지: {os.path.basename(path)}")

    def start_segmentation(self):
        if not self.selected:
            messagebox.showwarning("알림", "먼저 이미지를 하나 선택하세요.")
            return
        editor = SegEditor(self, self.selected, self.on_segmentation_done)
        self.wait_window(editor)

    def on_segmentation_done(self, mask):
        self.seg_mask = mask
        self.reconstruct_btn.configure(state="normal")

    def run_reconstruction(self):
        if not self.images or self.seg_mask is None or self.selected is None:
            messagebox.showwarning("알림", "먼저 바닥 영역을 선택하세요.")
            return
        upload_to_server(self.images, self.seg_mask, self.selected)


# ============================================================
# 🚀 실행
# ============================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()

