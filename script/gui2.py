import os
import locale
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import requests
from PIL import Image, ImageTk
import numpy as np
import cv2

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    TKDND = True
except Exception:
    TKDND = False


# ---- 한글 로케일 ----
try:
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
except locale.Error:
    pass


# ============================================================
# 🌐 서버 업로드 함수
# ============================================================
def upload_to_server(images, seg_image, seg_target):
    SERVER_URL = "https://<YOUR_NGROK_URL>/upload"  # ⚠️ ngrok 주소 수정
    try:
        print(f"📤 총 {len(images)}장의 이미지와 세그멘테이션 전송 중...")
        files = [("files", (os.path.basename(p), open(p, "rb"), "image/jpeg")) for p in images]
        seg_file = ("seg_image", (os.path.basename(seg_image), open(seg_image, "rb"), "image/png"))
        data = {"seg_target": os.path.basename(seg_target)}

        res = requests.post(SERVER_URL, files=files + [seg_file], data=data)
        if res.status_code == 200:
            js = res.json()
            eq = js.get("plane_equation", None)
            if eq:
                a, b, c, d = eq
                messagebox.showinfo("결과", f"✅ 평면 방정식:\n{a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f}=0")
            else:
                messagebox.showinfo("결과", "3D 모델 생성 완료 (평면 정보 없음)")
        else:
            messagebox.showerror("서버 오류", f"{res.status_code}\n{res.text}")
    except Exception as e:
        messagebox.showerror("전송 실패", str(e))


# ============================================================
#  세그멘테이션 편집기 (원본 유지 + 썸네일 미리보기)
# ============================================================
class SegEditor(tk.Toplevel):
    def __init__(self, master, image_path):
        super().__init__(master)
        self.title("세그멘테이션 편집 (미리보기 포함)")
        self.image_path = image_path
        self.result_seg_path = None

        self.img_bgr = cv2.imread(image_path)
        if self.img_bgr is None:
            messagebox.showerror("오류", f"이미지 로드 실패: {image_path}")
            self.destroy()
            return
        self.h0, self.w0 = self.img_bgr.shape[:2]
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)

        max_w, max_h = 1100, 820
        sw = max_w / self.w0
        sh = max_h / self.h0
        self.scale = min(1.0, sw, sh)
        self.disp_w, self.disp_h = int(self.w0 * self.scale), int(self.h0 * self.scale)

        self.mask = np.full((self.h0, self.w0), cv2.GC_PR_BGD, np.uint8)
        self.mode = "fg"
        self.brush = 14
        self.drawing = False
        self.final_mask = None
        self.last_result_rgb = None

        self._build_ui()
        self._render_canvas()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=(10, 6))
        ttk.Label(top, text="좌클릭 드래그로 시드 그리기 (전경/배경 전환 가능)").pack(side="left")
        ttk.Button(top, text="닫기", command=self.destroy).pack(side="right")

        mid = ttk.Frame(self)
        mid.pack(fill="x", padx=10, pady=6)
        ttk.Button(mid, text="전경 모드", command=lambda: self._set_mode("fg")).pack(side="left", padx=4)
        ttk.Button(mid, text="배경 모드", command=lambda: self._set_mode("bg")).pack(side="left", padx=4)
        ttk.Label(mid, text="브러시 크기").pack(side="left", padx=(12, 2))
        self.sld = ttk.Scale(mid, from_=4, to=40, value=self.brush, orient="horizontal", command=self._on_brush)
        self.sld.pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(mid, text="세그멘테이션 실행", command=self.run_grabcut).pack(side="left", padx=6)
        ttk.Button(mid, text="결과 저장", command=self.save_outputs).pack(side="left", padx=4)

        self.canvas = tk.Canvas(self, width=self.disp_w, height=self.disp_h, bg="#222222", highlightthickness=0)
        self.canvas.pack(padx=10, pady=(6, 10))
        self.canvas.bind("<ButtonPress-1>", self._on_down)
        self.canvas.bind("<B1-Motion>", self._on_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_up)

    def _set_mode(self, mode):
        self.mode = mode
        print(f"🖌 모드: {mode}")

    def _on_brush(self, v):
        self.brush = int(float(v))

    def _canvas_to_image_xy(self, x, y):
        ix, iy = int(x / self.scale), int(y / self.scale)
        ix = np.clip(ix, 0, self.w0 - 1)
        iy = np.clip(iy, 0, self.h0 - 1)
        return ix, iy

    def _on_down(self, e): self.drawing = True; self._paint(e.x, e.y)
    def _on_move(self, e):  self._paint(e.x, e.y) if self.drawing else None
    def _on_up(self, e):    self.drawing = False

    def _paint(self, x, y):
        ix, iy = self._canvas_to_image_xy(x, y)
        if self.mode == "fg":
            cv2.circle(self.mask, (ix, iy), self.brush, cv2.GC_FGD, -1)
        else:
            cv2.circle(self.mask, (ix, iy), self.brush, cv2.GC_BGD, -1)
        self._render_canvas()

    def _render_canvas(self):
        base = self.img_rgb.copy()
        overlay = base.copy()
        overlay[self.mask == cv2.GC_FGD] = (overlay[self.mask == cv2.GC_FGD] * 0.5 + np.array([0, 120, 255]) * 0.5).astype(np.uint8)
        overlay[self.mask == cv2.GC_BGD] = (overlay[self.mask == cv2.GC_BGD] * 0.5 + np.array([255, 80, 80]) * 0.5).astype(np.uint8)
        disp = Image.fromarray(overlay).resize((self.disp_w, self.disp_h))

        # ✅ 결과 미리보기 썸네일 표시
        if self.last_result_rgb is not None:
            thumb = Image.fromarray(self.last_result_rgb).resize((self.disp_w // 4, self.disp_h // 4))
            disp.paste(thumb, (self.disp_w - thumb.width - 8, self.disp_h - thumb.height - 8))

        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")

    def run_grabcut(self):
        self.configure(cursor="watch")
        self.update_idletasks()
        try:
            scale = 0.4
            small_bgr = cv2.resize(self.img_bgr, None, fx=scale, fy=scale)
            small_mask = cv2.resize(self.mask, (small_bgr.shape[1], small_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(small_bgr, small_mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
            m_full = cv2.resize(small_mask, (self.w0, self.h0), interpolation=cv2.INTER_NEAREST)
            mask2 = np.where((m_full == cv2.GC_FGD) | (m_full == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
            preview = self.img_rgb.copy()
            preview[mask2 == 1] = (preview[mask2 == 1] * 0.6 + np.array([80, 255, 120]) * 0.4).astype(np.uint8)
            self.last_result_rgb = preview
            self.final_mask = (mask2 * 255).astype(np.uint8)
            self._render_canvas()
            messagebox.showinfo("완료", "GrabCut 완료 (미리보기 갱신됨).")
        except Exception as e:
            messagebox.showerror("오류", str(e))
        finally:
            self.configure(cursor="")

    def save_outputs(self):
        if self.final_mask is None:
            messagebox.showwarning("알림", "먼저 세그멘테이션을 실행하세요.")
            return
        base, _ = os.path.splitext(self.image_path)
        seg_png = f"{base}_seg.png"
        cv2.imwrite(seg_png, self.final_mask)
        self.result_seg_path = seg_png
        messagebox.showinfo("저장됨", f"세그멘테이션 마스크 저장 완료:\n{os.path.basename(seg_png)}")
        self.destroy()


# ============================================================
#  메인 앱 (드래그&드롭 + 서버 전송)
# ============================================================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("VGGT 바닥 시드 지정 + 서버 전송")
        self.images, self.labels = [], []
        self.selected = None
        self.seg_image = None

        top = ttk.Frame(root)
        top.pack(fill="x", padx=10, pady=(10, 6))
        ttk.Label(top, text="이미지 드래그 또는 [파일 열기]").pack(side="left")
        ttk.Button(top, text="파일 열기", command=self.open_files).pack(side="right")

        mid = ttk.Frame(root)
        mid.pack(fill="x", padx=10, pady=6)
        ttk.Button(mid, text="세그멘테이션 편집", command=self.open_editor).pack(side="left", padx=4)
        ttk.Button(mid, text="서버로 전송", command=self.upload_all).pack(side="left", padx=4)

        self.info = ttk.Label(root, text="(이미지 선택 ▶ 세그멘테이션 ▶ 서버 전송)")
        self.info.pack(fill="x", padx=10, pady=(0, 6))

        self.grid = ttk.Frame(root)
        self.grid.pack(padx=10, pady=(4, 10))

        if TKDND:
            root.drop_target_register(DND_FILES)
            root.dnd_bind("<<Drop>>", self.on_drop)

    def on_drop(self, evt):
        files = self.root.tk.splitlist(evt.data)
        imgs = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if imgs: self.display(imgs)

    def open_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg;*.jpeg;*.png")])
        if files: self.display(files)

    def display(self, paths):
        for lb in self.labels: lb.destroy()
        self.labels.clear()
        self.images = list(paths)
        cols = 4
        r = c = 0
        for p in paths:
            im = Image.open(p); im.thumbnail((180, 180))
            tkim = ImageTk.PhotoImage(im)
            lb = tk.Label(self.grid, image=tkim, bd=2, relief="solid")
            lb.image = tkim
            lb.grid(row=r, column=c, padx=5, pady=5)
            lb.bind("<Button-1>", lambda e, path=p, lab=lb: self.select_one(path, lab))
            self.labels.append(lb)
            c += 1
            if c == cols: c = 0; r += 1
        self.info.configure(text=f"{len(self.images)}장 로드됨. 세그멘테이션할 이미지를 선택하세요.")

    def select_one(self, path, label):
        for lb in self.labels: lb.config(highlightthickness=0)
        label.config(highlightbackground="green", highlightthickness=3)
        self.selected = path
        self.info.configure(text=f"선택됨: {os.path.basename(path)}")

    def open_editor(self):
        if not self.selected:
            messagebox.showwarning("알림", "먼저 이미지를 선택하세요.")
            return
        editor = SegEditor(self.root, self.selected)
        self.root.wait_window(editor)
        if editor.result_seg_path:
            self.seg_image = editor.result_seg_path
            messagebox.showinfo("세그멘테이션 완료", f"저장됨: {os.path.basename(self.seg_image)}")

    def upload_all(self):
        if not self.images:
            messagebox.showwarning("알림", "이미지를 불러오세요.")
            return
        if not self.seg_image or not self.selected:
            messagebox.showwarning("알림", "세그멘테이션을 먼저 수행하세요.")
            return
        upload_to_server(self.images, self.seg_image, self.selected)


# ============================================================
# 🚀 실행
# ============================================================
def main():
    Root = TkinterDnD.Tk() if TKDND else tk.Tk()
    App(Root)
    Root.mainloop()


if __name__ == "__main__":
    main()

