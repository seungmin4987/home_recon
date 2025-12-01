import os
import locale
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
import cv2
import numpy as np
from PIL import Image, ImageTk
import tempfile
import requests


# 한글 로케일
try:
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
except locale.Error:
    pass


# ============================================================
# 🌐 서버 전송 함수
# ============================================================
def upload_to_server(images, seg_mask):
    SERVER_URL = "https://untribal-memorisingly-joanne.ngrok-free.dev/upload"
    try:
        files = [("files", (os.path.basename(p), open(p, "rb"), "image/jpeg")) for p in images]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, seg_mask)
            tmp_path = tmp.name
            files.append(("seg_image", (os.path.basename(tmp_path), open(tmp_path, "rb"), "image/png")))

        res = requests.post(SERVER_URL, files=files)
        os.remove(tmp_path)

        if res.status_code == 200:
            data = res.json()
            eq = data.get("plane_equation")
            if eq:
                a, b, c, d = eq
                messagebox.showinfo("결과", f"✅ 평면 방정식:\n{a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f}=0")
            else:
                messagebox.showinfo("결과", "✅ 3D 리컨스트럭션 완료 (평면 정보 없음)")
        else:
            messagebox.showerror("오류", f"서버 응답 오류: {res.status_code}\n{res.text}")

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
            # 결과 썸네일 오른쪽 아래 표시
            thumb = self.preview_img.resize((self.disp_w // 4, self.disp_h // 4))
            disp.paste(thumb, (self.disp_w - thumb.width - 8, self.disp_h - thumb.height - 8))
        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")

    def finish_segmentation(self):
        self.configure(cursor="watch")
        self.update_idletasks()
        try:
            # 🔹 GrabCut 빠르게 수행 (다운샘플링 적용)
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
        self.title("바닥 영역 기반 3D 리컨스트럭션")
        self.geometry("1000x600")
        self.configure(bg="#222")

        self.images = []
        self.labels = []
        self.selected = None
        self.seg_mask = None

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
        if not self.images or self.seg_mask is None:
            messagebox.showwarning("알림", "먼저 바닥 영역을 선택하세요.")
            return
        upload_to_server(self.images, self.seg_mask)


# ============================================================
# 🚀 실행
# ============================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()

