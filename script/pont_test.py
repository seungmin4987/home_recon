import tkinter as tk
from tkinter import font

root = tk.Tk()

# 폰트 목록 확인
fonts = font.families()
print("사용 가능한 폰트 수:", len(fonts))
print("NanumGothic" in fonts)

# 폰트 지정
lbl = tk.Label(root, text="안녕하세요, 나눔고딕 폰트 테스트!", font=("NanumGothic", 13))
lbl.pack(pady=20)

root.mainloop()

