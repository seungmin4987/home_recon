import os
import numpy as np
import open3d as o3d
import trimesh

# ==============================
# 1️⃣ 파일 경로 설정
# ==============================
OUTPUT_DIR = "/home/seungmin/home_recon/VGGT_Output"
points_path = os.path.join(OUTPUT_DIR, "world_points.npy")

# ==============================
# 2️⃣ 데이터 로드
# ==============================
points = np.load(points_path)
print(f"✅ world_points.npy shape: {points.shape}")

# (S, H, W, 3) 또는 (H, W, 3) 모두 지원
if points.ndim == 3:
    points_all = points.reshape(-1, 3)
elif points.ndim == 4:
    S, H, W, _ = points.shape
    points_all = points.reshape(-1, 3)
    print(f"📸 총 {S}개의 프레임 포인트 통합 완료 ({H}x{W})")
else:
    raise ValueError("⚠️ world_points.npy의 형태가 올바르지 않습니다.")

# 유효 포인트만 필터링
mask = np.isfinite(points_all).all(axis=1)
points_all = points_all[mask]

# ==============================
# 3️⃣ Open3D 포인트클라우드 생성
# ==============================
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_all)

# ==============================
# 4️⃣ 메시 재구성 (Poisson 또는 Ball Pivoting)
# ==============================
print("🌀 메시 재구성 중... (Poisson)")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# 너무 먼 외곽 제거
bbox = pcd.get_axis_aligned_bounding_box()
mesh_crop = mesh.crop(bbox)

# ==============================
# 5️⃣ Trimesh 변환 후 시각화
# ==============================
vertices = np.asarray(mesh_crop.vertices)
faces = np.asarray(mesh_crop.triangles)

tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
tm.show()

# 또는 Open3D 시각화
o3d.visualization.draw_geometries([mesh_crop])

