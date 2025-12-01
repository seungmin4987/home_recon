# floor_seg_gui_simple.py
import os
import locale
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# 드래그&드랍(optional)
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    TKDND = True
except Exception:
    TKDND = False

from PIL import Image, ImageTk
import numpy as np
import cv2


# ---- 한글 로케일(가능 시) ----
try:
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
except locale.Error:
    pass


# =========================
#  세그멘테이션 편집기
# =========================
class SegEditor(tk.Toplevel):
    """
    마우스만 사용:
    - 좌클릭 드래그로 시드 그리기
      (모드는 버튼으로 전경/배경 전환)
    - [세그멘테이션 실행] : 다운샘플링 기반 GrabCut → 결과 미리보기
    - [시드 지우기] : 초기화 후 다시 그려 재실행
    - [결과 저장] : *_seg.png / *_preview.png / *_mask.png / *_mask.npy
    """
    def __init__(self, master, image_path):
        super().__init__(master)
        self.title("세그멘테이션 편집 (마우스 전용)")
        self.image_path = image_path

        # 원본 이미지 로드 (BGR)
        self.img_bgr = cv2.imread(image_path)
        if self.img_bgr is None:
            messagebox.showerror("오류", f"이미지 로드 실패: {image_path}")
            self.destroy()
            return
        self.h0, self.w0 = self.img_bgr.shape[:2]
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)

        # 표시 크기(너무 크면 축소)
        max_w, max_h = 1100, 820
        sw = max_w / self.w0
        sh = max_h / self.h0
        self.scale = min(1.0, sw, sh)
        self.disp_w = int(self.w0 * self.scale)
        self.disp_h = int(self.h0 * self.scale)

        # GrabCut 호환 마스크: 0=BG,1=FG,2=PR_BG,3=PR_FG
        self.mask = np.full((self.h0, self.w0), cv2.GC_PR_BGD, np.uint8)

        # 상태
        self.mode = "fg"            # 'fg' or 'bg'
        self.brush = 14             # 브러시 반경
        self.drawing = False
        self.last_result_rgb = None  # 마지막 미리보기 결과
        self.final_mask = None       # 최종 저장용 이진 마스크(255/0)

        # UI
        self._build_ui()
        self._render_canvas()

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=(10, 6))

        self.lbl_info = ttk.Label(top, text="좌클릭 드래그로 시드 그립니다. 전경/배경은 아래 버튼으로 전환하세요.")
        self.lbl_info.pack(side="left")

        ttk.Button(top, text="닫기", command=self.destroy).pack(side="right", padx=4)

        mid = ttk.Frame(self)
        mid.pack(fill="x", padx=10, pady=6)

        # 모드 토글
        self.btn_fg = ttk.Button(mid, text="전경 모드", command=lambda: self._set_mode("fg"))
        self.btn_bg = ttk.Button(mid, text="배경 모드", command=lambda: self._set_mode("bg"))
        self.btn_fg.pack(side="left", padx=4)
        self.btn_bg.pack(side="left", padx=4)

        # 브러시 슬라이더
        ttk.Label(mid, text="브러시 크기").pack(side="left", padx=(16, 4))
        self.sld_brush = ttk.Scale(mid, from_=4, to=40, value=self.brush, orient="horizontal",
                                   command=self._on_brush_change)
        self.sld_brush.pack(side="left", padx=4, fill="x", expand=True)
        self.lbl_br = ttk.Label(mid, text=f"{self.brush}px")
        self.lbl_br.pack(side="left", padx=(4, 12))

        # 동작 버튼
        ttk.Button(mid, text="세그멘테이션 실행", command=self.run_grabcut).pack(side="left", padx=4)
        ttk.Button(mid, text="시드 지우기", command=self.reset_seeds).pack(side="left", padx=4)
        ttk.Button(mid, text="결과 저장", command=self.save_outputs).pack(side="left", padx=4)

        # 캔버스
        self.canvas = tk.Canvas(self, width=self.disp_w, height=self.disp_h, bg="#222222", highlightthickness=0)
        self.canvas.pack(padx=10, pady=(6, 10))

        # 이벤트 (마우스만)
        self.canvas.bind("<ButtonPress-1>", self._on_lbutton_down)
        self.canvas.bind("<B1-Motion>", self._on_lbutton_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_lbutton_up)

        # 초기 모드
        self._set_mode("fg")

    def _set_mode(self, mode):
        self.mode = mode
        if mode == "fg":
            self.btn_fg.configure(text="전경 모드 ✅")
            self.btn_bg.configure(text="배경 모드")
        else:
            self.btn_fg.configure(text="전경 모드")
            self.btn_bg.configure(text="배경 모드 ✅")

    def _on_brush_change(self, v):
        self.brush = int(float(v))
        self.lbl_br.configure(text=f"{self.brush}px")

    # ---------- 좌표 변환 ----------
    def _canvas_to_image_xy(self, x, y):
        ix = int(round(x / self.scale))
        iy = int(round(y / self.scale))
        ix = np.clip(ix, 0, self.w0 - 1)
        iy = np.clip(iy, 0, self.h0 - 1)
        return ix, iy

    # ---------- 페인팅 ----------
    def _on_lbutton_down(self, e):
        self.drawing = True
        self._paint_at(e.x, e.y)

    def _on_lbutton_move(self, e):
        if self.drawing:
            self._paint_at(e.x, e.y)

    def _on_lbutton_up(self, e):
        self.drawing = False

    def _paint_at(self, cx, cy):
        ix, iy = self._canvas_to_image_xy(cx, cy)
        radius = max(1, self.brush)

        # 마스크에 반영
        if self.mode == "fg":
            cv2.circle(self.mask, (ix, iy), radius, cv2.GC_FGD, -1)
        else:
            cv2.circle(self.mask, (ix, iy), radius, cv2.GC_BGD, -1)

        # 즉시 미리보기(시드 오버레이)
        self._render_canvas()

    # ---------- 렌더링 ----------
    def _render_canvas(self):
        """
        캔버스 표시:
        - 기본: 원본 위에 시드(전경=파랑, 배경=빨강) 반투명 오버레이
        - GrabCut 결과가 있으면 우하단에 썸네일
        """
        base = self.img_rgb.copy()
        overlay = base.copy()

        # 전경 시드 = 파랑톤
        fg_seed = (self.mask == cv2.GC_FGD)
        overlay[fg_seed] = (overlay[fg_seed] * 0.5 + np.array([0, 120, 255]) * 0.5).astype(np.uint8)
        # 배경 시드 = 빨강톤
        bg_seed = (self.mask == cv2.GC_BGD)
        overlay[bg_seed] = (overlay[bg_seed] * 0.5 + np.array([255, 80, 80]) * 0.5).astype(np.uint8)

        disp = Image.fromarray(overlay).resize((self.disp_w, self.disp_h), Image.LANCZOS)

        # 결과 썸네일(있으면)
        if self.last_result_rgb is not None:
            thumb = Image.fromarray(self.last_result_rgb).resize(
                (max(1, self.disp_w // 4), max(1, self.disp_h // 4)), Image.LANCZOS
            )
            disp = disp.copy()
            disp.paste(thumb, (self.disp_w - thumb.width - 8, self.disp_h - thumb.height - 8))

        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")

    # ---------- GrabCut (속도 개선: 다운샘플링) ----------
    def run_grabcut(self):
        """다운샘플링 기반 GrabCut → 업샘플 후 미리보기 반영"""
        self.configure(cursor="watch")
        self.update_idletasks()
        try:
            # 1) 축소
            scale_grabcut = 0.4  # 0.3~0.5 추천
            small_bgr = cv2.resize(self.img_bgr, None, fx=scale_grabcut, fy=scale_grabcut, interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(self.mask, (small_bgr.shape[1], small_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

            m = small_mask.copy()
            unknown = (m != cv2.GC_BGD) & (m != cv2.GC_FGD)
            m[unknown] = cv2.GC_PR_BGD

            # 2) GrabCut 실행
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(small_bgr, m, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

            # 3) 업샘플
            m_full = cv2.resize(m, (self.w0, self.h0), interpolation=cv2.INTER_NEAREST)
            mask2 = np.where((m_full == cv2.GC_FGD) | (m_full == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

            # 4) 미리보기/최종 마스크
            preview = self.img_rgb.copy()
            preview[mask2 == 1] = (preview[mask2 == 1] * 0.6 + np.array([80, 255, 120]) * 0.4).astype(np.uint8)

            self.last_result_rgb = preview
            self.final_mask = (mask2 * 255).astype(np.uint8)
            self._render_canvas()
            messagebox.showinfo("완료", "고속 GrabCut 완료 (다운샘플링 기반). 결과가 미리보기에 반영되었습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"GrabCut 실행 중 오류:\n{e}")
        finally:
            self.configure(cursor="")

    def reset_seeds(self):
        self.mask.fill(cv2.GC_PR_BGD)
        self.last_result_rgb = None
        self.final_mask = None
        self._render_canvas()

    def save_outputs(self):
        if self.final_mask is None:
            messagebox.showwarning("알림", "먼저 [세그멘테이션 실행]을 해주세요.")
            return

        base, _ = os.path.splitext(self.image_path)
        seg_png = f"{base}_seg.png"
        prev_png = f"{base}_preview.png"
        mask_png = f"{base}_mask.png"
        mask_npy = f"{base}_mask.npy"

        seg_rgb = cv2.cvtColor((self.img_rgb * (self.final_mask[:, :, None] // 255)).astype(np.uint8), cv2.COLOR_RGB2BGR)
        prev_bgr = cv2.cvtColor(self.last_result_rgb, cv2.COLOR_RGB2BGR)

        cv2.imwrite(seg_png, seg_rgb)
        cv2.imwrite(prev_png, prev_bgr)
        cv2.imwrite(mask_png, self.final_mask)
        np.save(mask_npy, self.final_mask)

        messagebox.showinfo("저장됨", f"저장 완료:\n{os.path.basename(seg_png)}\n{os.path.basename(prev_png)}\n"
                                    f"{os.path.basename(mask_png)}\n{os.path.basename(mask_npy)}")


# =========================
#  메인 앱 (드래그&드롭 + 갤러리)
# =========================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("VGGT 바닥 시드 지정 (드래그&드롭 + GrabCut)")

        self.images = []
        self.labels = []
        self.selected = None

        top = ttk.Frame(root)
        top.pack(fill="x", padx=10, pady=(10, 6))

        ttk.Label(top, text="이미지 파일을 드래그&드롭하거나 [파일 열기]로 불러오세요.").pack(side="left")
        ttk.Button(top, text="파일 열기", command=self.open_files).pack(side="right")

        mid = ttk.Frame(root)
        mid.pack(fill="x", padx=10, pady=6)
        ttk.Button(mid, text="세그멘테이션 편집", command=self.open_editor).pack(side="left")

        self.info = ttk.Label(root, text="(이미지 1개 선택) ▶ 세그멘테이션 편집")
        self.info.pack(fill="x", padx=10, pady=(0, 6))

        self.grid = ttk.Frame(root)
        self.grid.pack(padx=10, pady=(4, 10))

        if TKDND:
            root.drop_target_register(DND_FILES)
            root.dnd_bind("<<Drop>>", self.on_drop)

    def on_drop(self, evt):
        files = self.root.tk.splitlist(evt.data)
        imgs = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not imgs:
            messagebox.showwarning("알림", "이미지 파일만 지원합니다.")
            return
        self.display(imgs)

    def open_files(self):
        files = filedialog.askopenfilenames(title="이미지 선택", filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if files:
            self.display(files)

    def display(self, paths):
        # 초기화
        for lb in self.labels:
            lb.destroy()
        self.labels.clear()
        self.images = list(paths)
        self.selected = None

        # 썸네일 그리드
        cols = 4
        r = c = 0
        for p in self.images:
            try:
                im = Image.open(p)
                im.thumbnail((180, 180))
                tkim = ImageTk.PhotoImage(im)
                lb = tk.Label(self.grid, image=tkim, bd=2, relief="solid")
                lb.image = tkim
                lb.grid(row=r, column=c, padx=5, pady=5)
                lb.bind("<Button-1>", lambda e, path=p, lab=lb: self.select_one(path, lab))
                self.labels.append(lb)
                c += 1
                if c == cols:
                    c = 0
                    r += 1
            except Exception as ex:
                print("썸네일 실패:", p, ex)

        self.info.configure(text=f"{len(self.images)}장의 이미지가 로드되었습니다. 하나를 클릭해 선택하세요.")

    def select_one(self, path, label):
        for lb in self.labels:
            lb.config(highlightthickness=0)
        label.config(highlightbackground="green", highlightthickness=3)
        self.selected = path
        self.info.configure(text=f"선택됨: {os.path.basename(path)}")

    def open_editor(self):
        if not self.selected:
            messagebox.showwarning("알림", "먼저 이미지를 하나 선택하세요.")
            return
        SegEditor(self.root, self.selected)


def main():
    Root = TkinterDnD.Tk() if TKDND else tk.Tk()
    App(Root)
    Root.mainloop()


if __name__ == "__main__":
    main()

