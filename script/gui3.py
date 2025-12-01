import os
import locale
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import requests
from PIL import Image, ImageTk
import numpy as np
import cv2


# ---- 한글 로케일 ----
try:
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
except locale.Error:
    pass


# ============================================================
# 🌐 서버 업로드 함수
# ============================================================
def upload_to_server(image_path, seg_path):
    SERVER_URL = "https://<YOUR_NGROK_URL>/upload"  # ⚠️ ngrok 주소로 변경
    try:
        files = [
            ("files", (os.path.basename(image_path), open(image_path, "rb"), "image/jpeg")),
            ("seg_image", (os.path.basename(seg_path), open(seg_path, "rb"), "image/png")),
        ]
        print(f"📤 서버로 전송 중: {os.path.basename(seg_path)} ...")
        res = requests.post(SERVER_URL, files=files)
        if res.status_code == 200:
            js = res.json()
            eq = js.get("plane_equation")
            if eq:
                a, b, c, d = eq
                messagebox.showinfo("결과", f"✅ 평면 방정식:\n{a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f}=0")
            else:
                messagebox.showinfo("결과", "✅ 전송 완료 (평면 방정식 없음)")
        else:
            messagebox.showerror("서버 오류", f"{res.status_code}\n{res.text}")
    except Exception as e:
        messagebox.showerror("전송 실패", str(e))


# ============================================================
# 🧩 세그멘테이션 편집기
# ============================================================
class SegEditor(tk.Toplevel):
    def __init__(self, master, image_path):
        super().__init__(master)
        self.title("세그멘테이션 편집")
        self.image_path = image_path
        self.result_seg_path = None

        # 이미지 로드
        self.img_bgr = cv2.imread(image_path)
        if self.img_bgr is None:
            messagebox.showerror("오류", f"이미지 로드 실패: {image_path}")
            self.destroy()
            return

        self.h0, self.w0 = self.img_bgr.shape[:2]
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        self.mask = np.full((self.h0, self.w0), cv2.GC_PR_BGD, np.uint8)

        self.scale = min(1.0, 1100 / self.w0, 820 / self.h0)
        self.disp_w, self.disp_h = int(self.w0 * self.scale), int(self.h0 * self.scale)

        self.mode = "fg"
        self.brush = 14
        self.drawing = False
        self.last_result_rgb = None
        self.final_mask = None

        self._build_ui()
        self._render_canvas()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=(10, 6))
        ttk.Label(top, text="좌클릭 드래그로 전경/배경 지정 후 [세그멘테이션 실행]").pack(side="left")

        mid = ttk.Frame(self)
        mid.pack(fill="x", padx=10, pady=6)
        ttk.Button(mid, text="전경 모드", command=lambda: self._set_mode("fg")).pack(side="left", padx=4)
        ttk.Button(mid, text="배경 모드", command=lambda: self._set_mode("bg")).pack(side="left", padx=4)
        ttk.Button(mid, text="세그멘테이션 실행", command=self.run_grabcut).pack(side="left", padx=8)
        ttk.Button(mid, text="결과 저장", command=self.save_outputs).pack(side="left", padx=4)

        self.canvas = tk.Canvas(self, width=self.disp_w, height=self.disp_h, bg="#222", highlightthickness=0)
        self.canvas.pack(padx=10, pady=(6, 10))
        self.canvas.bind("<ButtonPress-1>", self._on_down)
        self.canvas.bind("<B1-Motion>", self._on_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_up)

    def _set_mode(self, mode):
        self.mode = mode
        print(f"🖌 모드: {mode}")

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
        color = cv2.GC_FGD if self.mode == "fg" else cv2.GC_BGD
        cv2.circle(self.mask, (ix, iy), self.brush, color, -1)
        self._render_canvas()

    def _render_canvas(self):
        overlay = self.img_rgb.copy()
        overlay[self.mask == cv2.GC_FGD] = (overlay[self.mask == cv2.GC_FGD] * 0.5 + np.array([0, 120, 255]) * 0.5).astype(np.uint8)
        overlay[self.mask == cv2.GC_BGD] = (overlay[self.mask == cv2.GC_BGD] * 0.5 + np.array([255, 80, 80]) * 0.5).astype(np.uint8)
        disp = Image.fromarray(overlay).resize((self.disp_w, self.disp_h))
        if self.last_result_rgb is not None:
            thumb = Image.fromarray(self.last_result_rgb).resize((self.disp_w // 4, self.disp_h // 4))
            disp.paste(thumb, (self.disp_w - thumb.width - 8, self.disp_h - thumb.height - 8))
        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")

    def run_grabcut(self):
        try:
            small_scale = 0.4
            small_bgr = cv2.resize(self.img_bgr, None, fx=small_scale, fy=small_scale)
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
            messagebox.showinfo("완료", "세그멘테이션 완료! 오른쪽 아래 미리보기를 확인하세요.")
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def save_outputs(self):
        if self.final_mask is None:
            messagebox.showwarning("알림", "먼저 세그멘테이션을 실행하세요.")
            return
        base, _ = os.path.splitext(self.image_path)
        seg_png = f"{base}_seg.png"
        cv2.imwrite(seg_png, self.final_mask)
        self.result_seg_path = seg_png

        # ✅ 사용자 확인 메시지
        if messagebox.askyesno("서버 전송", "이 세그멘테이션 결과로 서버에 전송하시겠습니까?"):
            upload_to_server(self.image_path, seg_png)
        else:
            messagebox.showinfo("저장됨", f"로컬에만 저장되었습니다:\n{os.path.basename(seg_png)}")

        self.destroy()


# ============================================================
# 🪶 메인 GUI (버튼 하나만 존재)
# ============================================================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("바닥 평면 세그멘테이션")
        self.btn = ttk.Button(root, text="이미지 선택 및 세그멘테이션 시작", command=self.start_seg)
        self.btn.pack(padx=40, pady=40)

    def start_seg(self):
        file = filedialog.askopenfilename(
            title="이미지 선택",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
        )
        if file:
            SegEditor(self.root, file)


# ============================================================
# 🚀 실행
# ============================================================
def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

