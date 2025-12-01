from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# 1️⃣ 인증 초기화
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # 처음 한 번은 브라우저가 열림

# 2️⃣ 드라이브 객체 생성
drive = GoogleDrive(gauth)

# 3️⃣ 업로드할 파일 지정
upload_file = 'test.jpg'  # 여기에 업로드할 파일 경로 지정

# 4️⃣ 업로드 수행
gfile = drive.CreateFile({'title': upload_file})  # 파일 이름 지정
gfile.SetContentFile(upload_file)
gfile.Upload()

print(f"✅ 업로드 완료: {upload_file}")
print(f"📎 파일 URL: https://drive.google.com/file/d/{gfile['id']}/view")
