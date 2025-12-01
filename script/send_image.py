import requests
import glob
import os

# ✅ 코랩 ngrok 주소 (Colab 서버 실행 후 출력된 URL 복사해서 넣기)
SERVER_URL = "https://untribal-memorisingly-joanne.ngrok-free.dev/upload"

# ✅ 전송할 이미지 디렉토리 (하드코딩)
IMAGE_DIR = "/home/seungmin/home_recon/sample"

# 전송할 이미지 목록
image_paths = glob.glob(f"{IMAGE_DIR}/*.jpg") + glob.glob(f"{IMAGE_DIR}/*.png")

# multipart 데이터 생성
files = [("files", (os.path.basename(p), open(p, "rb"), "image/jpeg")) for p in image_paths]

print(f"📤 {len(files)}개의 이미지를 전송 중...")
response = requests.post(SERVER_URL, files=files)

print("서버 응답:", response.text)

