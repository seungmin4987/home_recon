# ============================================================
# 📦 Flask + VGGT 3D 리컨 + 세그 기반 평면(SVD) 추정 서버
#   - 1차 요청: 이미지(files[])만 전송 → 서버는 저장만 하고 OK 반환
#   - 2차 요청: 이미지(files[]) + seg_image + seg_name → 전체 reconstruction 수행
# ============================================================

from flask import Flask, request, send_file, jsonify
from pyngrok import ngrok
import os
import sys
import time
import gc

import numpy as np
import cv2
import torch
from PIL import Image
import trimesh
import numpy as np


# ============================================================
# 🔧 경로 / 환경 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "received")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.5  # depth confidence threshold

# 🔹 ngrok 토큰
ngrok.set_auth_token("35hElFWApZtMXQG7xIZHkWANvHj_eTLqqAbRUt3nCiNAypYs")

# ============================================================
# 🔹 VGGT import
# ============================================================
sys.path.append(os.path.join(BASE_DIR, "vggt"))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


app = Flask(__name__)


# ============================================================
# 🧠 VGGT 모델 로드
# ============================================================
print("VGGT 모델 로딩 중...")
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
state_dict = torch.hub.load_state_dict_from_url(_URL)
model.load_state_dict(state_dict)
model.eval().to(DEVICE)
print("🔥 VGGT 모델 로드 완료")


# ============================================================
# 🧩 VGGT 추론
# ============================================================
def run_vggt_on_images(image_paths):
    if len(image_paths) == 0:
        raise ValueError("No images provided to VGGT")

    image_paths_sorted = sorted(image_paths)
    imgs_tensor = load_and_preprocess_images(image_paths_sorted).to(DEVICE)

    dtype = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype, enabled=(DEVICE == "cuda")):
        preds = model(imgs_tensor)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], imgs_tensor.shape[-2:])
    preds["extrinsic"] = extrinsic
    preds["intrinsic"] = intrinsic

    for k in list(preds.keys()):
        if isinstance(preds[k], torch.Tensor):
            preds[k] = preds[k].cpu().numpy().squeeze(0)

    depth_map = preds["depth"]
    world_pts = unproject_depth_map_to_point_map(
        depth_map,
        preds["extrinsic"],
        preds["intrinsic"],
    )

    preds["world_points_from_depth"] = world_pts
    return preds, image_paths_sorted


# ============================================================
# 🧩 GLB 포인트 클라우드 생성
# ============================================================
def voxel_downsample_points(pts, cols, voxel_size=0.01, max_points=800000):
    """
    간단한 voxel 그리드 다운샘플링.
    - pts: (N,3) float32
    - cols: (N,3/4) uint8
    """
    if voxel_size <= 0 or pts.shape[0] == 0:
        return pts, cols

    grid = np.floor(pts / voxel_size).astype(np.int32)
    key = grid.view([("x", np.int32), ("y", np.int32), ("z", np.int32)])
    _, uniq_idx = np.unique(key, return_index=True)
    pts_ds = pts[uniq_idx]
    cols_ds = cols[uniq_idx]

    if pts_ds.shape[0] > max_points:
        sel = np.random.choice(pts_ds.shape[0], max_points, replace=False)
        pts_ds = pts_ds[sel]
        cols_ds = cols_ds[sel]

    return pts_ds, cols_ds


def export_colored_glb(preds, img_paths, out_path, conf_thres=0.5):
    world_pts = preds["world_points_from_depth"]
    conf_map = preds.get("depth_conf", None)

    S, H, W, _ = world_pts.shape
    S = min(S, len(img_paths))

    all_pts, all_cols = [], []

    for i in range(S):
        pts = world_pts[i].reshape(-1, 3)
        valid = np.isfinite(pts).all(axis=1)

        if conf_map is not None:
            conf = conf_map[i].reshape(-1)
            valid &= conf > conf_thres

        pts = pts[valid]
        img = np.array(Image.open(img_paths[i]).convert("RGB").resize((W, H)))
        cols = (img.reshape(-1, 3) / 255.0)[valid]

        all_pts.append(pts)
        all_cols.append(cols)

    pts_cat = np.concatenate(all_pts, axis=0)
    cols_cat = np.concatenate(all_cols, axis=0)

    # 다운샘플 + 양자화 (가시성 유지용)
    pts_cat = pts_cat.astype(np.float32)
    cols_uint8 = np.clip(cols_cat * 255.0, 0, 255).astype(np.uint8)
    pts_ds, cols_ds = voxel_downsample_points(pts_cat, cols_uint8, voxel_size=0.01, max_points=800000)

    print(f"GLB 포인트 수: 원본 {pts_cat.shape[0]} → 다운샘플 {pts_ds.shape[0]}")

    cloud = trimesh.points.PointCloud(pts_ds, colors=cols_ds)
    cloud.export(out_path)
    print(f"GLB 저장 완료: {out_path}")

    return pts_ds


# ============================================================
# 🧮 세그 기반 바닥 평면 추정
# ============================================================
def estimate_floor_plane_from_3d(preds, seg_mask_path, image_index=0, max_samples=50000):
    world_pts = preds["world_points_from_depth"]
    S, H, W, _ = world_pts.shape

    pts_map = world_pts[image_index]

    seg = cv2.imread(seg_mask_path, cv2.IMREAD_GRAYSCALE)
    if seg is None:
        print("⚠️ 세그 마스크 없음 → 기본 평면 사용")
        return (0.0, 1.0, 0.0, 0.0)

    if seg.shape != (H, W):
        seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)

    mask = seg > 128
    ys, xs = np.where(mask)

    if len(xs) < 50:
        print("⚠️ 세그 픽셀 너무 적음 → 기본 평면")
        return (0.0, 1.0, 0.0, 0.0)

    pts = pts_map[ys, xs, :]
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]

    if pts.shape[0] < 50:
        print("⚠️ 3D 유효 포인트 너무 적음 → 기본 평면")
        return (0.0, 1.0, 0.0, 0.0)

    if pts.shape[0] > max_samples:
        idx = np.random.choice(pts.shape[0], max_samples, replace=False)
        pts = pts[idx]

    centroid = np.mean(pts, axis=0)
    pts_c = pts - centroid
    _, _, vh = np.linalg.svd(pts_c, full_matrices=False)
    normal = vh[-1]
    normal /= (np.linalg.norm(normal) + 1e-12)
    d = -np.dot(normal, centroid)

    a, b, c = normal
    print(f"📐 바닥 평면: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    return (float(a), float(b), float(c), float(d))


# ============================================================
# 🌐 Flask 라우트
# ============================================================
@app.route("/", methods=["GET"])
def home():
    return "VGGT Reconstruction Server Running!"


@app.route("/upload", methods=["POST"])
def upload_images():
    """
    1) 1차 요청:
        - files[]만 존재
        → 이미지 저장만 하고 "ok" 반환

    2) 2차 요청:
        - files[] + seg_image + seg_name
        → VGGT + GLB + 평면 추정 수행
    """
    try:
        files = request.files.getlist("files")
        seg_file = request.files.get("seg_image")   # None이면 1차 요청
        seg_name = request.form.get("seg_name", None)

        # -----------------------------
        # 1) 모든 요청에서 이미지 저장
        # -----------------------------
        saved_paths = []
        for f in files:
            filename = f.filename
            if filename == "":
                continue
            save_path = os.path.join(UPLOAD_DIR, filename)
            f.save(save_path)
            saved_paths.append(save_path)

        if not saved_paths:
            return jsonify({"error": "이미지가 전송되지 않음"}), 400

        print(f"[UPLOAD] 이미지 {len(saved_paths)}개 저장 완료")

        # -----------------------------
        # ⭐ 1차 요청: seg 없음
        # -----------------------------
        if seg_file is None:
            print("[UPLOAD] 세그 없음 → 1차 업로드로 처리 완료")
            return jsonify({"status": "ok", "msg": "이미지 저장 완료 (세그 기다리는 중)"}), 200

        # -----------------------------
        # ⭐ 2차 요청: 세그 존재 → 전체 reconstruction 실행
        # -----------------------------
        print("[UPLOAD] 세그 파일 감지 → 2차 재구성 시작")

        seg_path = os.path.join(UPLOAD_DIR, "floor_mask.png")
        seg_file.save(seg_path)
        print(f"[UPLOAD] 세그 마스크 저장: {seg_path}")

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        start = time.time()
        preds, ordered_paths = run_vggt_on_images(saved_paths)
        print(f"🔥 VGGT 추론 완료: {time.time() - start:.2f}s")

        seg_index = 0
        if seg_name:
            for i, p in enumerate(ordered_paths):
                if os.path.basename(p) == seg_name:
                    seg_index = i
                    break

        print(f"세그 뷰 index = {seg_index}, 파일 = {os.path.basename(ordered_paths[seg_index])}")

        glb_path = os.path.join(OUTPUT_DIR, "reconstructed_scene.glb")
        export_colored_glb(preds, ordered_paths, glb_path, conf_thres=CONF_THRES)

        plane_eq = estimate_floor_plane_from_3d(preds, seg_path, image_index=seg_index)

        response = send_file(glb_path, as_attachment=True, download_name="reconstructed_scene.glb")
        response.headers["Plane-Equation"] = str(plane_eq)
        print("GLB + 평면 방정식 전송 완료")

        return response

    except Exception as e:
        print(f"[ERROR] 서버 처리 중 오류: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🚀 서버 실행
# ============================================================
if __name__ == "__main__":
    public_url = ngrok.connect(5000).public_url
    print(f"\n외부 접속 URL: {public_url}")
    print(f"업로드 엔드포인트: {public_url}/upload\n")
    app.run(port=5000)
