import os
import locale
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import font as tkfont
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import numpy as np
import cv2
import requests
import tempfile
import trimesh
from trimesh.geometry import plane_transform
import json
import threading
import io
import time


# ---- 한글 로케일 ----
try:
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
except locale.Error:
    pass


# ============================================================
# 🔎 평면 헤더 파싱 유틸
# ============================================================
def parse_plane_header(header_str):
    if not header_str:
        return None
    s = header_str.strip().strip("()[]")
    parts = s.replace(",", " ").split()
    try:
        a, b, c, d = [float(x) for x in parts[:4]]
        return (a, b, c, d)
    except:
        return None


# ============================================================
# 🌐 서버 업로드 함수 (1차/2차 자동 처리)
# ============================================================
def safe_message(kind, title, message, on_ok=None):
    fn = {
        "info": messagebox.showinfo,
        "warning": messagebox.showwarning,
        "error": messagebox.showerror
    }.get(kind)
    if fn is None:
        return
    root = tk._default_root

    def _run():
        try:
            fn(title, message)
            if callable(on_ok):
                try:
                    on_ok()
                except Exception:
                    pass
        except Exception:
            pass

    if root is not None and root.winfo_exists():
        try:
            root.after(0, _run)
        except Exception:
            _run()
    else:
        _run()


def upload_to_server(images, seg_mask, seg_image_path, on_wait_glb=None):
    SERVER_URL = "https://untribal-memorisingly-joanne.ngrok-free.dev/upload"

    try:
        print(f"[upload] start: {len(images)} images, seg_mask={seg_mask is not None}")
        files = []
        opened_handles = []  # keep references alive

        def _compress_image(path, max_side=1800, quality=85):
            im = Image.open(path).convert("RGB")
            w, h = im.size
            if max(w, h) > max_side:
                scale = max_side / float(max(w, h))
                im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True, subsampling="4:2:0")
            buf.seek(0)
            return buf

        for p in images:
            try:
                buf = _compress_image(p)
                fname = os.path.splitext(os.path.basename(p))[0] + ".jpg"
                files.append(("files", (fname, buf, "image/jpeg")))
                opened_handles.append(buf)
            except Exception as e:
                print(f"[upload] compress failed for {p}: {e}, fallback to raw")
                fh = open(p, "rb")
                files.append(("files", (os.path.basename(p), fh, "image/jpeg")))
                opened_handles.append(fh)

        data = {}
        tmp_path = None

        # -------------------------------
        # ⭐ seg_mask 존재 → 2차 요청
        # -------------------------------
        if seg_mask is not None and seg_image_path is not None:
            seg_name = os.path.basename(seg_image_path)
            data["seg_name"] = seg_name

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                cv2.imwrite(tmp.name, seg_mask)
                tmp_path = tmp.name
            files.append(("seg_image", (os.path.basename(tmp_path), open(tmp_path, "rb"), "image/png")))

        # -------------------------------
        # 서버 POST
        # -------------------------------
        res = requests.post(SERVER_URL, files=files, data=data, timeout=60)
        print(f"[upload] response status: {res.status_code}, content-type: {res.headers.get('Content-Type')}")
        if tmp_path:
            os.remove(tmp_path)
        for h in opened_handles:
            try:
                h.close()
            except Exception:
                pass

        # -------------------------------
        # ⭐ 1차/2차 모두: 서버가 응답을 주었으면 전송 완료 → GLB 대기 상태 라벨로 전환
        # -------------------------------
        if res.status_code == 200 and callable(on_wait_glb):
            try:
                on_wait_glb()
            except Exception:
                pass

        # -------------------------------
        # ⭐ 1차 요청 응답
        # -------------------------------
        if res.status_code == 200 and "application/json" in res.headers.get("Content-Type", ""):
            print("✔ 1차 업로드 완료 (세그 기다리는 중)")
            return None

        # -------------------------------
        # ⭐ 2차 요청 응답 (GLB)
        # -------------------------------
        if res.status_code == 200:
            if callable(on_wait_glb):
                try:
                    on_wait_glb()
                except Exception:
                    pass
            base_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(base_dir, "received_glb")
            os.makedirs(save_dir, exist_ok=True)

            out_path = os.path.join(save_dir, "received_model.glb")
            with open(out_path, "wb") as f:
                f.write(res.content)

            plane_str = res.headers.get("Plane-Equation", "")
            plane_eq = parse_plane_header(plane_str)

            # 메타 저장
            meta_path = os.path.join(save_dir, "received_model_meta.json")
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump({
                    "glb_path": out_path,
                    "seg_image_name": data.get("seg_name"),
                    "plane_equation": plane_eq,
                    "plane_equation_raw_header": plane_str
                }, mf, ensure_ascii=False, indent=2)

            # 3D 리컨 완료 안내 (확인 후 종료)
            def _close_app():
                if tk._default_root is not None and tk._default_root.winfo_exists():
                    tk._default_root.destroy()
            safe_message("info", "완료", "모델이 생성되었습니다.\n가구 배치 시뮬레이터에서 확인해 주세요.", on_ok=_close_app)

            # GLB 시각화 (임시 비활성화)
            # try:
            #     mesh_or_scene = trimesh.load(out_path)
            #     scene = mesh_or_scene if isinstance(mesh_or_scene, trimesh.Scene) else trimesh.Scene(mesh_or_scene)
            #
            #     if plane_eq is not None:
            #         a, b, c, d = plane_eq
            #         n = np.array([a, b, c])
            #         if np.linalg.norm(n) > 1e-6:
            #             origin = -d * n / (np.dot(n, n) + 1e-12)
            #             T = plane_transform(origin, n)
            #             scene.apply_transform(T)
            #
            #     # 바닥 평면 mesh 추가
            #     try:
            #         bounds = scene.bounds
            #         min_b, max_b = bounds
            #         size = max_b - min_b
            #         if not np.all(np.isfinite(size)):
            #             size = np.array([1.0, 1.0, 1.0])
            #         px = max(size[0], 1.0) * 1.2
            #         py = max(size[1], 1.0) * 1.2
            #         thickness = max(size[0], size[1]) * 0.01
            #         cx = (min_b[0] + max_b[0]) / 2
            #         cy = (min_b[1] + max_b[1]) / 2
            #
            #         plane_mesh = trimesh.creation.box(extents=(px, py, thickness))
            #         plane_mesh.apply_translation([cx, cy, -thickness/2])
            #         plane_mesh.visual.vertex_colors = np.array([180,230,200,150], np.uint8)
            #         scene.add_geometry(plane_mesh)
            #     except Exception as e:
            #         print("평면 메쉬 생성 실패:", e)
            #
            #     scene.show()
            #
            # except Exception as e:
            #     safe_message("error", "시각화 오류", str(e))

        else:
            safe_message("error", "오류", f"서버 오류: {res.status_code}\n{res.text[:500]}")

    except Exception as e:
        print(f"[upload] error: {e}")
        safe_message("error", "전송 실패", str(e))


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
        # 다크 테마 팔레트
        palette = {
            "bg": "#0f1115",
            "panel": "#151921",
            "canvas": "#1b1f27",
            "text": "#f5f5f5",
            "subtext": "#c4c4c4",
            "accent": "#3a6aa8",
            "accent_hover": "#4b7fbf",
            "accent_pressed": "#2f5989",
            "outline": "#2b3240",
        }
        self.configure(bg=palette["bg"])

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        try:
            default_font = tkfont.nametofont("TkDefaultFont")
            default_font.configure(size=max(default_font.cget("size"), 11))
            text_font = tkfont.nametofont("TkTextFont")
            text_font.configure(size=max(text_font.cget("size"), 11))
        except Exception:
            pass

        style.configure("Seg.TFrame", background=palette["panel"])
        style.configure("Seg.TLabel", background=palette["panel"], foreground=palette["text"])
        style.configure(
            "Seg.TButton",
            background=palette["accent"],
            foreground=palette["text"],
            bordercolor=palette["outline"],
            focusthickness=2,
            focustcolor=palette["outline"],
            padding=(8, 4)
        )
        style.map(
            "Seg.TButton",
            background=[
                ("pressed", palette["accent_pressed"]),
                ("active", palette["accent_hover"])
            ]
        )

        ttk.Label(self, text="바닥 부분을 칠한 뒤, 미리보기 → 리컨스트럭션 순으로 진행하세요.", style="Seg.TLabel").pack(pady=(8, 4))
        self.canvas = tk.Canvas(self, width=self.disp_w, height=self.disp_h, bg=palette["canvas"], highlightthickness=0)
        self.canvas.pack(padx=8, pady=(4, 10))
        self.canvas.bind("<ButtonPress-1>", self._on_down)
        self.canvas.bind("<B1-Motion>", self._on_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_up)
        btn_wrap = ttk.Frame(self, style="Seg.TFrame")
        btn_wrap.pack(pady=(4, 8))
        self.preview_btn = ttk.Button(btn_wrap, text="세그 결과 미리보기", command=self.preview_segmentation, style="Seg.TButton")
        self.preview_btn.grid(row=0, column=0, padx=4)
        self.recon_btn = ttk.Button(btn_wrap, text="리컨스트럭션 수행", command=self.finish_segmentation, state="disabled", style="Seg.TButton")
        self.recon_btn.grid(row=0, column=1, padx=4)

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
            thumb = self.preview_img.resize((self.disp_w//4, self.disp_h//4))
            disp.paste(thumb, (self.disp_w - thumb.width - 8, self.disp_h - thumb.height - 8))
        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.create_image(0,0,image=self.tk_img,anchor="nw")

    def finish_segmentation(self):
        if self.result_mask is None:
            messagebox.showwarning("알림", "먼저 '세그 결과 미리보기'를 눌러 세그멘테이션을 완료하세요.")
            return
        self.on_complete(self.result_mask)
        self.destroy()

    def preview_segmentation(self):
        self.configure(cursor="watch")
        self.update_idletasks()
        try:
            small_scale = 0.3
            small_bgr = cv2.resize(self.img_bgr, None, fx=small_scale, fy=small_scale)
            small_mask = cv2.resize(self.mask, (small_bgr.shape[1], small_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            bgd, fgd = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)
            cv2.grabCut(small_bgr, small_mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
            mask_full = cv2.resize(small_mask, (self.w0, self.h0), interpolation=cv2.INTER_NEAREST)

            mask2 = np.where((mask_full == cv2.GC_FGD) | (mask_full == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
            self.result_mask = (mask2 * 255).astype(np.uint8)

            preview = self.img_rgb.copy()
            preview[mask2==1] = (preview[mask2==1]*0.6 + np.array([80,255,120])*0.4).astype(np.uint8)
            self.preview_img = Image.fromarray(preview)
            self._render_canvas()

            # 미리보기 완료 후 리컨 버튼 활성화
            self.recon_btn.configure(state="normal")

        except Exception as e:
            messagebox.showerror("오류", str(e))
        finally:
            self.configure(cursor="")


# ============================================================
# 🖼️ 메인 GUI (드롭 → 자동 업로드, 세그 → 자동 2차 업로드)
# ============================================================
class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("3D reconstruction")
        self.geometry("1000x600")
        self.configure(bg="#222")

        self.images = []
        self.labels = []
        self.selected = None
        self.seg_mask = None
        self.spinner_active = False
        self.spinner_base_text = ""
        self.spinner_timer = None
        self.spinner_start_time = None

        ttk.Label(self, text="이미지를 드래그앤드롭 하세요.", background="#222", foreground="white").pack(pady=(12,4))

        self.drop_area = tk.Frame(self, bg="#333", width=880, height=220, highlightbackground="#555", highlightthickness=1)
        self.drop_area.pack(pady=6)
        self.drop_area.pack_propagate(False)
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind("<<Drop>>", self.on_drop)

        self.drop_hint = tk.Label(self.drop_area, text="Drop Images Here", bg="#333", fg="white")
        self.drop_hint.place(relx=0.5, rely=0.5, anchor="center")

        # 업로드 상태 스피너 (드롭 영역 아래)
        self.spinner_frame = tk.Frame(self, bg="#222")
        self.spinner_label = tk.Label(self.spinner_frame, text="전송 중...", bg="#222", fg="white")
        self.spinner_label.pack(pady=(0,6))
        self.spinner_progress = ttk.Progressbar(self.spinner_frame, mode="indeterminate", length=260)
        self.spinner_progress.pack()
        self.spinner_frame.pack(pady=(0,0))
        self.spinner_frame.pack_forget()

        ttk.Label(self, text="썸네일 중 하나를 선택해 바닥 영역을 지정하세요.", background="#222", foreground="white").pack(pady=(8,0))

        self.seg_btn = ttk.Button(self, text="바닥 영역 선택하기", command=self.start_segmentation, state="disabled")
        self.seg_btn.pack(pady=(8,6))

        # 2차 업로드 버튼은 필요 없으므로 숨김
        self.reconstruct_btn = ttk.Button(self, text="(disabled)", state="disabled")
        self.reconstruct_btn.pack_forget()

    # -------------------------------
    # 드래그 앤 드롭 처리
    # -------------------------------
    def on_drop(self, event):
        files = self.tk.splitlist(event.data)
        valid = [f for f in files if f.lower().endswith((".png",".jpg",".jpeg"))]
        if not valid:
            messagebox.showwarning("알림", "이미지 파일만 드래그하세요.")
            return

        self.images = valid
        self.show_thumbnails()
        self.seg_btn.configure(state="normal")

        # ⭐ 1차 업로드 자동 실행 (seg 없음) - 스피너 표시 없이 전송만
        self.run_upload_async(self.images, seg_mask=None, seg_image_path=None, show_spinner=False)

    # -------------------------------
    # 썸네일 표시 & 선택
    # -------------------------------
    def show_thumbnails(self):
        for w in self.drop_area.winfo_children():
            if w is self.drop_hint:
                continue
            w.destroy()
        self.labels.clear()
        try:
            if self.drop_hint.winfo_exists():
                self.drop_hint.place_forget()
        except Exception:
            pass

        cols = 5
        r = c = 0
        pad = 8
        for path in self.images:
            im = Image.open(path)
            im.thumbnail((160,160))
            tk_im = ImageTk.PhotoImage(im)
            lb = tk.Label(self.drop_area, image=tk_im, bg="#2d2d2d", relief="solid", bd=2)
            lb.image = tk_im
            lb.grid(row=r, column=c, padx=pad, pady=pad)
            lb.bind("<Button-1>", lambda e,p=path,l=lb: self.select_one(p,l))
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

    # -------------------------------
    # 세그 편집 시작
    # -------------------------------
    def start_segmentation(self):
        if not self.selected:
            messagebox.showwarning("알림", "먼저 썸네일을 클릭하세요.")
            return
        editor = SegEditor(self, self.selected, self.on_segmentation_done)
        self.wait_window(editor)

    # -------------------------------
    # 세그 완료 → 2차 업로드 자동 실행
    # -------------------------------
    def on_segmentation_done(self, mask):
        self.seg_mask = mask

        # ⭐ 세그 완료 시 자동 2차 업로드 (스피너 표시)
        self.run_upload_async(self.images, self.seg_mask, self.selected, show_spinner=True, initial_text="3D 모델 기다리는 중...")

    # -------------------------------
    # 업로드 비동기 실행 + 스피너 제어
    # -------------------------------
    def run_upload_async(self, images, seg_mask, seg_image_path, show_spinner=False, initial_text="전송 중..."):
        if show_spinner:
            self._start_spinner(initial_text)
        threading.Thread(target=self._upload_worker, args=(images, seg_mask, seg_image_path, show_spinner), daemon=True).start()

    def _upload_worker(self, images, seg_mask, seg_image_path, show_spinner):
        try:
            upload_to_server(
                images,
                seg_mask,
                seg_image_path,
                on_wait_glb=self._spinner_phase_wait_glb
            )
        except Exception:
            # 오류 발생 시에만 스피너를 멈추고 안내
            try:
                if self.winfo_exists():
                    self.after(0, self._stop_spinner)
            except Exception:
                pass
            raise

    def _start_spinner(self, text):
        if not self.winfo_exists():
            return
        self.spinner_base_text = text
        self.spinner_start_time = time.time()
        self.spinner_active = True
        self._update_spinner_elapsed()
        if not self.spinner_frame.winfo_ismapped():
            self.spinner_frame.pack(pady=(2,4))
        try:
            self.spinner_progress.start(10)
        except Exception:
            pass
        try:
            if self.drop_hint.winfo_exists():
                self.drop_hint.place_forget()
        except Exception:
            pass

    def _stop_spinner(self):
        if not self.winfo_exists() or not self.drop_area.winfo_exists():
            return
        try:
            self.spinner_progress.stop()
        except Exception:
            pass
        # 스피너 프레임은 유지하여 라벨만 바꾸도록 한다
        self.spinner_active = False
        if self.spinner_timer is not None:
            try:
                self.after_cancel(self.spinner_timer)
            except Exception:
                pass
            self.spinner_timer = None
        if not self.labels:
            try:
                if self.drop_hint.winfo_exists():
                    self.drop_hint.place(relx=0.5, rely=0.5, anchor="center")
            except Exception:
                pass

    def _spinner_phase_wait_glb(self):
        if not self.winfo_exists():
            return
        try:
            self.after(0, lambda: self._set_spinner_text("3D 모델 기다리는 중..."))
        except Exception:
            pass

    def _set_spinner_text(self, text):
        if not self.winfo_exists() or not self.spinner_active:
            return
        self.spinner_base_text = text
        self._update_spinner_elapsed()

    def _update_spinner_elapsed(self):
        if not self.spinner_active or not self.winfo_exists():
            return
        elapsed = int(time.time() - (self.spinner_start_time or time.time()))
        self.spinner_label.config(text=f"{self.spinner_base_text} ({elapsed}초)")
        try:
            self.spinner_timer = self.after(1000, self._update_spinner_elapsed)
        except Exception:
            pass


# ============================================================
# 🚀 실행
# ============================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()
