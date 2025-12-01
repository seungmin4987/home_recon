import os
import json
import numpy as np
import trimesh
from trimesh.geometry import plane_transform


def load_meta_and_glb():
    """
    received_glb/received_model_meta.json을 읽고
    GLB 경로와 평면 방정식을 반환
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "received_glb")

    meta_path = os.path.join(save_dir, "received_model_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"메타파일을 찾을 수 없습니다: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    glb_path = meta.get("glb_path")
    plane_eq = meta.get("plane_equation")

    if glb_path is None:
        # 메타에 glb_path가 없다면 기본 경로로 가정
        glb_path = os.path.join(save_dir, "received_model.glb")

    if not os.path.isabs(glb_path):
        # 상대경로라면 save_dir 기준으로 변경
        glb_path = os.path.join(save_dir, os.path.basename(glb_path))

    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"GLB 파일을 찾을 수 없습니다: {glb_path}")

    return glb_path, plane_eq


def add_floor_plane(scene: trimesh.Scene):
    """
    scene의 bounds를 기준으로 z=0에 얇은 바닥 plane mesh 추가
    """
    try:
        bounds = scene.bounds  # (2, 3) : [min, max]
        min_b, max_b = bounds
        size = max_b - min_b

        if not np.all(np.isfinite(size)):
            size = np.array([1.0, 1.0, 1.0])

        size_x = max(size[0], 1.0)
        size_y = max(size[1], 1.0)

        # 약간 여유 있는 바닥
        px = size_x * 1.2
        py = size_y * 1.2
        thickness = max(size_x, size_y) * 0.01  # 매우 얇게

        center_x = (min_b[0] + max_b[0]) / 2.0
        center_y = (min_b[1] + max_b[1]) / 2.0

        plane_mesh = trimesh.creation.box(extents=(px, py, thickness))
        # 윗면이 z=0에 오도록 살짝 아래로 내림
        plane_mesh.apply_translation([center_x, center_y, -thickness / 2.0])

        # 살짝 반투명 연한 색
        plane_color = np.array([180, 230, 200, 150], dtype=np.uint8)
        plane_mesh.visual.vertex_colors = plane_color

        scene.add_geometry(plane_mesh)
    except Exception as e:
        print(f"[경고] 바닥 평면 메쉬 추가 중 오류: {e}")


def main():
    # 1) 메타 + glb 경로 읽기
    glb_path, plane_eq = load_meta_and_glb()
    print(f"[INFO] GLB 경로: {glb_path}")
    print(f"[INFO] 평면 방정식: {plane_eq}")

    # 2) GLB 로드
    mesh_or_scene = trimesh.load(glb_path)

    # 항상 Scene으로 통일
    if isinstance(mesh_or_scene, trimesh.Scene):
        scene = mesh_or_scene
    else:
        scene = trimesh.Scene(mesh_or_scene)

    # 3) 평면 방정식이 있으면, z=0으로 정렬
    if plane_eq is not None:
        try:
            a, b, c, d = plane_eq
            n = np.array([a, b, c], dtype=np.float64)
            if np.linalg.norm(n) > 1e-6:
                # 평면 위의 한 점
                origin = -d * n / (np.dot(n, n) + 1e-12)
                # plane_transform: 이 평면이 z=0, normal=+z가 되도록
                T = plane_transform(origin, n)
                scene.apply_transform(T)
                print("[INFO] 평면 정렬(transform) 적용 완료.")
            else:
                print("[경고] 평면 법선 벡터의 노름이 너무 작아 정렬을 건너뜁니다.")
        except Exception as e:
            print(f"[경고] 평면 정렬 중 오류: {e}")
    else:
        print("[INFO] 평면 방정식이 없어 정렬을 생략합니다.")

    # 4) 바닥 plane mesh 추가
    add_floor_plane(scene)

    # 5) 뷰어 띄우기
    print("[INFO] 3D 뷰어를 실행합니다.")
    scene.show()


if __name__ == "__main__":
    main()

