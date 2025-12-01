import requests

# 🔹 Colab 쪽 ngrok URL
COLAB_URL = "https://untribal-memorisingly-joanne.ngrok-free.dev/"  # 여기에 실제 URL 입력

# 1️⃣ 연결 테스트
res = requests.get(f"{COLAB_URL}/ping")
print("서버 응답:", res.json())

# 2️⃣ 메시지 전송
data = {"from": "로컬PC", "msg": "전송완료."}
res = requests.post(f"{COLAB_URL}/message", json=data)
print("응답:", res.json())

