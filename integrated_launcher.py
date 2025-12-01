import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox


class LauncherApp(tk.Tk):
    """
    Virtual Model House 런처:
    - 버튼 1: 이미지 리컨스트럭션 스튜디오(GUI.py)
    - 버튼 2: 가구 배치 시뮬레이터(furniture_simulator.py)
    각 스크립트는 별도 프로세스로 띄워 Tk 충돌을 피한다.
    """

    def __init__(self):
        super().__init__()
        self.title("Virtual Model House")
        self.geometry("380x240")
        self.resizable(False, False)
        self.configure(bg="#0f1115")

        self.base_dir = os.path.abspath(os.path.dirname(__file__))
        self.gui_proc = None
        self.sim_proc = None

        self._setup_style()

        ttk.Label(self, text="Virtual Model House", style="Title.TLabel").pack(pady=(18, 4))
        ttk.Label(self,
                  text="3D 재구성과 배치 시뮬레이션을\n한 곳에서 실행하세요.",
                  style="Sub.TLabel").pack(pady=(0, 10))

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=8, padx=20, fill="x")

        ttk.Button(btn_frame, text="1. Reconstruction Studio 열기", style="Primary.TButton",
                   command=self.launch_gui).pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="2. Furniture Simulator 열기", style="Secondary.TButton",
                   command=self.launch_simulator).pack(fill="x", pady=5)

        self.status_var = tk.StringVar(value="대기 중")
        ttk.Label(self, textvariable=self.status_var, style="Status.TLabel").pack(pady=(14, 8))

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("TFrame", background="#0f1115")
        style.configure("Title.TLabel", font=("Helvetica", 16, "bold"), background="#0f1115", foreground="#e7ecf4")
        style.configure("Sub.TLabel", font=("Helvetica", 10), background="#0f1115", foreground="#9ba3b0")
        style.configure("Status.TLabel", font=("Helvetica", 9), background="#0f1115", foreground="#6c7686")
        style.configure("Primary.TButton",
                        font=("Helvetica", 11, "bold"),
                        background="#3a7afe",
                        foreground="#ffffff",
                        padding=10)
        style.map("Primary.TButton",
                  background=[("active", "#2d63d6")],
                  foreground=[("active", "#ffffff")])
        style.configure("Secondary.TButton",
                        font=("Helvetica", 11, "bold"),
                        background="#20242c",
                        foreground="#e7ecf4",
                        padding=10)
        style.map("Secondary.TButton",
                  background=[("active", "#161a21")],
                  foreground=[("active", "#ffffff")])

    def launch_gui(self):
        self._launch_script("GUI.py", proc_attr="gui_proc", label="Reconstruction Studio")

    def launch_simulator(self):
        self._launch_script("furniture_simulator.py", proc_attr="sim_proc", label="Furniture Simulator")

    def _launch_script(self, script_name, proc_attr, label):
        proc = getattr(self, proc_attr)
        if proc is not None and proc.poll() is None:
            messagebox.showinfo("알림", f"{label}가 이미 실행 중입니다.")
            return

        script_path = os.path.join(self.base_dir, script_name)
        if not os.path.exists(script_path):
            messagebox.showerror("오류", f"파일을 찾을 수 없습니다:\n{script_path}")
            return

        try:
            new_proc = subprocess.Popen([sys.executable, script_path], cwd=self.base_dir)
            setattr(self, proc_attr, new_proc)
            self.status_var.set(f"{label} 실행 중 (PID {new_proc.pid})")
        except Exception as e:
            messagebox.showerror("실행 실패", str(e))

    def on_close(self):
        # 런처 종료 시 자식 프로세스는 그대로 둔다.
        self.destroy()


if __name__ == "__main__":
    app = LauncherApp()
    app.mainloop()
