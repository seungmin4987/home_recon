import os
import json
import numpy as np
import trimesh
from trimesh.geometry import plane_transform

# ----------------- 설정값 -----------------
CEILING_LIFT_DIST = 2.5   # 천장 자동 계산 실패 시 fallback
PLANE_FLIPPED = True      # 🔥 스냅샷 코드와 동일: transform용 플립 ON


# ============================================================
# 메타 + GLB 로딩 (plane_equation 포함)
# ============================================================
def load_meta_and_glb():
    """
    received_glb/received_model_meta.json 에서
    - glb_path
    - plane_equation (a, b, c, d)
    를 읽어오는 함수
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "received_glb")

    meta_path = os.path.join(save_dir, "received_model_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    glb_path = meta.get("glb_path") or os.path.join(save_dir, "received_model.glb")
    if not os.path.isabs(glb_path):
        glb_path = os.path.join(save_dir, os.path.basename(glb_path))
    if not os.path.exists(glb_path):
        raise FileNotFoundError(glb_path)

    plane_eq = meta.get("plane_equation")
    if plane_eq is None or len(plane_eq) < 4:
        raise ValueError("plane_equation 이 메타파일에 없습니다.")

    return glb_path, tuple(plane_eq)


# ============================================================
# 정렬된 Scene 에서 점 + 색상 모으기
#   (scene.dump(concatenate=True) 사용: graph transform 이 반영된 좌표)
# ============================================================
def scene_to_points_with_colors(scene: trimesh.Scene):
    merged = scene.dump(concatenate=True)  # world 좌표로 합쳐진 geometry

    if isinstance(merged, trimesh.Trimesh):
        pts = np.asarray(merged.vertices, dtype=np.float64)

        vc = None
        if hasattr(merged, "visual") and hasattr(merged.visual, "vertex_colors"):
            vc = np.asarray(merged.visual.vertex_colors)
        if vc is None or vc.shape[0] != pts.shape[0]:
            vc = np.tile(
                np.array([[200, 200, 200, 255]], dtype=np.uint8),
                (pts.shape[0], 1)
            )
        else:
            vc = vc.astype(np.uint8)
            # (N,3) 이면 alpha 채워주기
            if vc.ndim == 2 and vc.shape[1] == 3:
                alpha = np.full((vc.shape[0], 1), 255, dtype=np.uint8)
                vc = np.concatenate([vc, alpha], axis=1)

        return pts, vc

    elif isinstance(merged, trimesh.points.PointCloud):
        pts = np.asarray(merged.vertices, dtype=np.float64)
        vc = None
        if hasattr(merged, "colors") and merged.colors is not None:
            vc = np.asarray(merged.colors)
        if vc is None or vc.shape[0] != pts.shape[0]:
            vc = np.tile(
                np.array([[200, 200, 200, 255]], dtype=np.uint8),
                (pts.shape[0], 1)
            )
        else:
            vc = vc.astype(np.uint8)
            if vc.ndim == 2 and vc.shape[1] == 3:
                alpha = np.full((vc.shape[0], 1), 255, dtype=np.uint8)
                vc = np.concatenate([vc, alpha], axis=1)
        return pts, vc

    else:
        raise TypeError(f"scene_to_points_with_colors: 알 수 없는 타입: {type(merged)}")


# ============================================================
# 바닥 / 천장 / 서브천장 plane 시각화용 박스
#   - bounds 인자를 줘도 되고, 안 주면 scene.bounds 사용
# ============================================================
def add_floor_plane(scene: trimesh.Scene, bounds=None):
    """
    평면 정렬 후 z=0 근처에 얇은 바닥 plane 추가 (시각화용)
    """
    try:
        if bounds is None:
            bmin, bmax = scene.bounds
        else:
            bmin, bmax = bounds

        size = bmax - bmin
        if not np.all(np.isfinite(size)):
            size = np.array([1.0, 1.0, 1.0])

        sx, sy = max(size[0], 1.0), max(size[1], 1.0)
        px, py = sx * 1.2, sy * 1.2
        thickness = max(sx, sy) * 0.01

        cx = (bmin[0] + bmax[0]) * 0.5
        cy = (bmin[1] + bmax[1]) * 0.5

        plane_mesh = trimesh.creation.box(extents=(px, py, thickness))
        # z=0 이 바닥이 되도록, 박스의 윗면이 z=0에 오게 배치
        plane_mesh.apply_translation([cx, cy, -thickness / 2.0])
        plane_mesh.visual.vertex_colors = np.array([130, 118, 95, 150], np.uint8)
        scene.add_geometry(plane_mesh)
    except Exception as e:
        print("[경고] floor plane 추가 실패:", e)


def add_ceiling_plane(scene: trimesh.Scene,
                      ceiling_height: float,
                      color_rgba=None,
                      bounds=None):
    """
    z축 기준 ceiling_height 위치에 얇은 천장 plane 추가 (시각화용)
    """
    if color_rgba is None:
        color_rgba = np.array([200, 80, 80, 130], np.uint8)  # 기본: 빨간색 계열

    try:
        if bounds is None:
            bmin, bmax = scene.bounds
        else:
            bmin, bmax = bounds

        size = bmax - bmin
        if not np.all(np.isfinite(size)):
            size = np.array([1.0, 1.0, 1.0])

        sx, sy = max(size[0], 1.0), max(size[1], 1.0)
        px, py = sx * 1.2, sy * 1.2
        thickness = max(sx, sy) * 0.01

        cx = (bmin[0] + bmax[0]) * 0.5
        cy = (bmin[1] + bmax[1]) * 0.5

        plane_mesh = trimesh.creation.box(extents=(px, py, thickness))
        plane_mesh.apply_translation([cx, cy, ceiling_height - thickness / 2.0])
        plane_mesh.visual.vertex_colors = np.array(color_rgba, np.uint8)
        scene.add_geometry(plane_mesh)
    except Exception as e:
        print("[경고] ceiling plane 추가 실패:", e)


def add_sub_ceiling_plane(scene: trimesh.Scene,
                          ceiling_height: float,
                          ratio: float = 0.95,
                          bounds=None):
    """
    천장보다 약간 아래 평면 추가.
    - ceiling_height 가 H라면, 서브천장 높이는 H * ratio (기본 0.95)
    """
    sub_height = ceiling_height * ratio
    print(f"[INFO] sub-ceiling height = {sub_height:.3f} (ratio={ratio})")

    color = np.array([80, 200, 80, 130], np.uint8)  # 초록색
    add_ceiling_plane(scene,
                      sub_height,
                      color_rgba=color,
                      bounds=bounds)
    return sub_height


# ============================================================
# 1) 씬 정렬 + z=0 floor snap + auto ceiling 높이 계산
#    (스냅샷 코드 _load_and_align_scene 와 동일 로직)
# ============================================================
def align_scene_and_compute_ceiling():
    glb_path, plane_eq = load_meta_and_glb()
    print("[INFO] GLB:", glb_path)
    print("[INFO] Plane eq (raw):", plane_eq)

    # GLB 로드
    mesh_or_scene = trimesh.load(glb_path)
    if isinstance(mesh_or_scene, trimesh.Scene):
        scene = mesh_or_scene
    else:
        scene = trimesh.Scene(mesh_or_scene)

    # 1) 바닥 평면 기준으로 법선 방향 정리 (방 안쪽이 +dist 되도록)
    a, b, c, d = plane_eq
    n = np.array([a, b, c], dtype=float)
    nn = float(np.dot(n, n))
    if nn < 1e-8:
        raise ValueError("평면 법선이 너무 작음")
    n_len = np.sqrt(nn)

    try:
        merged = scene.dump(concatenate=True)
        if isinstance(merged, trimesh.Trimesh) and merged.vertices.shape[0] > 0:
            verts = merged.vertices
            signed = (verts @ n + d) / n_len
            median_dist = float(np.median(signed))
            # 방 안쪽(천장 방향)이 + 가 되도록 방향 자동 결정
            if median_dist < 0.0:
                n = -n
                d = -d
                print("[INFO] plane orientation flipped so that inside is +dist")
        else:
            print("[WARN] verts 가 비어서 plane orientation 추정 생략.")
    except Exception as e:
        print("[WARN] plane orientation 추정 중 오류:", e)

    # 2) PLANE_FLIPPED 옵션 적용 + plane_transform 로 정렬
    a2, b2, c2 = n[0], n[1], n[2]
    d2 = d

    if PLANE_FLIPPED:
        a2, b2, c2, d2 = -a2, -b2, -c2, -d2
        print("[INFO] PLANE_FLIPPED 적용됨 (transform 용).")

    n2 = np.array([a2, b2, c2], dtype=float)
    nn2 = float(np.dot(n2, n2))
    if nn2 < 1e-8:
        raise ValueError("평면 법선이 너무 작음 (after orient/flip)")

    # 새 평면(ax+by+cz+d=0)의 한 점 p0
    p0 = -d2 * n2 / nn2

    # plane_transform(p0, n2): p0를 원점으로 옮기고, n2를 +Z축으로 보정
    T = plane_transform(p0, n2)
    scene.apply_transform(T)

    # 3) 바닥을 z ≈ 0 으로 스냅
    try:
        merged2 = scene.dump(concatenate=True)
        z_all = merged2.vertices[:, 2]
    except Exception:
        z_all = None

    if z_all is not None and z_all.size > 0:
        z_floor = float(np.quantile(z_all, 0.01))
        scene.apply_translation([0.0, 0.0, -z_floor])
        print(f"[INFO] floor snapped: ~{z_floor:.4f} → 0.0")
    else:
        print("[WARN] floor snap 생략 (z_all 없음)")

    # 4) z 분포 기준으로 천장 높이 자동 선택 (95% quantile)
    auto_ceiling_height = None
    try:
        merged3 = scene.dump(concatenate=True)
        z_all2 = merged3.vertices[:, 2]
        if z_all2.size > 0:
            z_min = float(np.quantile(z_all2, 0.01))
            z_max = float(np.quantile(z_all2, 0.99))
            z_95 = float(np.quantile(z_all2, 0.95))
            auto_ceiling_height = max(z_95, 0.0)
            print(
                f"[INFO] z-range≈[{z_min:.3f}, {z_max:.3f}], "
                f"auto ceiling(95%)≈{auto_ceiling_height:.3f}"
            )
        else:
            print("[WARN] z_all2 가 비어 있어 auto ceiling 계산 불가.")
    except Exception as e:
        print("[WARN] auto ceiling 계산 중 오류:", e)

    if auto_ceiling_height is None:
        print("[WARN] auto ceiling 높이 계산에 실패하여 fallback 사용")
        auto_ceiling_height = CEILING_LIFT_DIST

    # 정렬 + 스냅된 상태의 bounds (전체 방 기준)
    aligned_bounds = scene.bounds

    # 디버깅용 최종 z-range
    try:
        merged4 = scene.dump(concatenate=True)
        z_all4 = merged4.vertices[:, 2]
        z_min4 = float(np.quantile(z_all4, 0.01))
        z_max4 = float(np.quantile(z_all4, 0.99))
        print(
            f"[DEBUG] aligned scene final z-range≈[{z_min4:.3f}, {z_max4:.3f}]"
        )
    except Exception:
        pass

    return scene, auto_ceiling_height, aligned_bounds


# ============================================================
# 씬 정렬 + 점 분할 + 바닥/천장/서브천장 추가
#   (정렬 부분은 위 align_scene_and_compute_ceiling() 과 동일)
# ============================================================
def build_scene_with_planes():
    """
    1) GLB 로드 + 평면 정렬 (스냅샷 코드와 동일)
    2) z 분포(95% quantile)로 천장 높이 자동 추정
    3) 서브천장(높이의 5% 아래) 기준으로 점들을 위/아래로 나눈 뒤,
       서브천장 **아래(z <= sub_height)** 점들만 남김
    4) 바닥 + 천장 + 서브천장 plane 시각화
    """
    # 1) 정렬 + auto ceiling 계산
    scene_aligned, auto_ceiling_height, aligned_bounds = align_scene_and_compute_ceiling()

    # 2) 서브천장 높이
    ratio = 0.95
    sub_height = auto_ceiling_height * ratio
    print(f"[INFO] sub-ceiling height = {sub_height:.3f} (ratio={ratio})")

    # 3) 정렬된 좌표계에서 점 + 색 추출 후, 서브천장 기준으로 분할
    pts, cols = scene_to_points_with_colors(scene_aligned)
    z_pts = pts[:, 2]

    mask_keep = z_pts <= sub_height   # 🔥 항상 '아래' 점만 사용
    num_keep = int(mask_keep.sum())
    num_drop = int((~mask_keep).sum())

    print(f"[INFO] points kept (below)={num_keep}, dropped (above)={num_drop}")

    if num_keep == 0:
        print("[WARN] 서브천장 아래 점이 없습니다. 전체 점을 그대로 사용합니다.")
        pts_sel = pts
        cols_sel = cols
    else:
        pts_sel = pts[mask_keep]
        cols_sel = cols[mask_keep]

    # 4) 최종 Scene: 선택된 점(PointCloud) + 바닥/천장/서브천장 plane
    final_scene = trimesh.Scene()

    pc = trimesh.points.PointCloud(pts_sel, colors=cols_sel)
    final_scene.add_geometry(pc)

    # 바닥/천장 plane 은 "정렬된 전체 방 bounds" 기준으로 생성
    add_floor_plane(final_scene, bounds=aligned_bounds)
    add_ceiling_plane(final_scene,
                      auto_ceiling_height,
                      bounds=aligned_bounds)
    add_sub_ceiling_plane(final_scene,
                          auto_ceiling_height,
                          ratio=ratio,
                          bounds=aligned_bounds)

    # 디버깅용: 선택된 점들의 z-range
    try:
        z_min_sel = float(np.quantile(pts_sel[:, 2], 0.01))
        z_max_sel = float(np.quantile(pts_sel[:, 2], 0.99))
        print(
            f"[DEBUG] selected points final z-range≈[{z_min_sel:.3f}, {z_max_sel:.3f}]"
        )
    except Exception:
        pass

    return final_scene


# ============================================================
# 실행: 3D 뷰어로 확인
# ============================================================
if __name__ == "__main__":
    scene = build_scene_with_planes()
    print("[INFO] 3D 뷰어를 띄웁니다. (닫으면 프로그램 종료)")
    scene.show()

