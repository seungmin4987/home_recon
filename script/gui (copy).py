import os
import locale
import tkinter as tk
from tkinter import ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk

# ✅ 한글 로케일 설정 (폰트 깨짐 방지)
try:
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
except locale.Error:
    pass

class FloorSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VGGT 바닥 선택 도구 (Drag & Drop 지원)")
        self.images = []          # 이미지 경로 리스트
        self.image_labels = []    # 썸네일 라벨들
        self.selected = set()     # 선택된 이미지 경로

        # ✅ 스타일 및 폰트 설정
        style = ttk.Style()
        style.configure("TLabel", font=("NanumGothic", 12))
        style.configure("TButton", font=("NanumGothic", 11))
        style.configure("TFrame", font=("NanumGothic", 11))

        # 안내문
        self.label_info = ttk.Label(self.root, text="여기에 이미지를 드래그 앤 드랍 하세요.", font=("NanumGothic", 13))
        self.label_info.pack(pady=10)

        # 격자 표시용 프레임
        self.frame = ttk.Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        # 다음 단계 버튼
        ttk.Button(self.root, text="다음 단계 (바닥 선택)", command=self.enable_selection).pack(pady=8)

        # DnD 이벤트 등록
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)

    def on_drop(self, event):
        """드롭된 파일 처리"""
        file_list = self.root.tk.splitlist(event.data)
        image_files = [f for f in file_list if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not image_files:
            self.label_info.config(text="⚠️ 지원되지 않는 파일 형식입니다.", foreground="red")
            return
        self.display_images(image_files)

    def display_images(self, image_files):
        """이미지를 격자로 표시"""
        # 기존 이미지 초기화
        for lbl in self.image_labels:
            lbl.destroy()
        self.image_labels.clear()
        self.images.clear()

        # 새 이미지 불러오기
        row, col = 0, 0
        for path in image_files:
            try:
                img = Image.open(path)
                img.thumbnail((150, 150))
                tk_img = ImageTk.PhotoImage(img)
                lbl = tk.Label(self.frame, image=tk_img, relief="solid", borderwidth=2)
                lbl.image = tk_img
                lbl.grid(row=row, column=col, padx=5, pady=5)
                self.image_labels.append(lbl)
                self.images.append(path)
                col += 1
                if col == 4:  # 한 줄에 4개
                    col = 0
                    row += 1
            except Exception as e:
                print(f"❌ {path} 로드 실패: {e}")
        self.label_info.config(
            text=f"{len(self.images)}장의 이미지가 로드되었습니다.",
            foreground="black"
        )

    def enable_selection(self):
        """선택 모드 활성화"""
        if not self.image_labels:
            self.label_info.config(text="먼저 이미지를 드래그 앤 드랍 해주세요.", foreground="red")
            return
        self.label_info.config(
            text="바닥이라고 생각되는 이미지를 클릭하세요. (초록색 테두리 = 선택됨)",
            foreground="blue"
        )
        for lbl, path in zip(self.image_labels, self.images):
            lbl.bind("<Button-1>", lambda e, p=path, l=lbl: self.toggle_selection(p, l))

    def toggle_selection(self, path, label):
        """클릭 시 선택/해제"""
        if path in self.selected:
            self.selected.remove(path)
            label.config(highlightbackground="black", highlightthickness=0)
        else:
            self.selected.add(path)
            label.config(highlightbackground="green", highlightthickness=3)
        print("현재 선택된 바닥 이미지:", list(self.selected))

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = FloorSelectorApp(root)
    root.mainloop()

