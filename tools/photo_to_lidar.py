#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ảnh RGB -> đám mây điểm kiểu LiDAR (.bin + .label) cho rellis_25d_generator.

Depth Anything V2 (độ sâu) + SegFormer (nhãn -> ID RELLIS) -> chiếu ngược -> cân mặt đất (RANSAC).
Cần: pip install torch transformers

Ví dụ (từ gốc repo):
  python tools/photo_to_lidar.py anh.jpg --out raw_photo --simulate-os1
  python rellis_25d_generator/main.py --data-dir raw_photo --seq 00000 --frame 000000 --out rellis_25d_generator/output_photo
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# Console Windows mặc định cp1252 không in được tiếng Việt
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
SEG_MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"

# Từ khoá trong tên lớp ADE20K -> ID lớp RELLIS-3D (duyệt theo thứ tự, khớp đầu tiên thắng)
ADE_TO_RELLIS = [
    ("sky", 7),
    ("person", 17),
    ("tree", 4), ("palm", 4),
    ("plant", 19), ("flower", 19),
    ("grass", 3), ("field", 3),
    ("road", 10), ("runway", 10),
    ("sidewalk", 23), ("floor", 23), ("pavement", 23), ("stairs", 23), ("stairway", 23), ("step", 23),
    ("dirt", 1), ("earth", 1), ("path", 1), ("land", 1), ("sand", 1), ("hill", 1), ("mountain", 1),
    ("rock", 34), ("stone", 34),
    ("water", 6), ("sea", 6), ("river", 6), ("lake", 6), ("pool", 6),
    ("car", 8), ("truck", 8), ("bus", 8), ("van", 8), ("bicycle", 8), ("motorbike", 8), ("minibike", 8),
    ("fence", 18), ("railing", 18),
    ("pole", 5), ("streetlight", 5), ("traffic light", 5), ("signboard", 5),
    ("building", 12), ("house", 12), ("wall", 12), ("skyscraper", 12), ("tower", 12),
    ("bench", 9), ("box", 9),
]
GROUND_IDS = {1, 3, 10, 23, 31, 33, 34}
SKY_ID = 7


# Thông số camera
def estimate_intrinsics(img: Image.Image, hfov_deg: float):
    """fx từ EXIF FocalLengthIn35mmFilm nếu có, ngược lại từ FOV ngang giả định."""
    W, H = img.size
    fx = None
    try:
        exif = img.getexif()
        f35 = exif.get_ifd(0x8769).get(41989) or exif.get(41989)
        if f35:
            fx = float(f35) / 36.0 * max(W, H)
            print(f"[Camera] EXIF tiêu cự 35mm = {f35} mm -> fx = {fx:.1f} px")
    except Exception:
        pass
    if fx is None:
        fx = (W / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        print(f"[Camera] Không có EXIF, giả định FOV ngang {hfov_deg}° -> fx = {fx:.1f} px")
    return fx, fx, W / 2.0, H / 2.0


# Mô hình học sâu (import lười để phần hình học chạy được không cần torch)
def predict_depth(img: Image.Image, model_id: str, device: str):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    with torch.no_grad():
        out = model(**proc(images=img, return_tensors="pt").to(device))
    depth = torch.nn.functional.interpolate(
        out.predicted_depth.unsqueeze(1), size=img.size[::-1], mode="bicubic", align_corners=False
    )[0, 0]
    return depth.cpu().numpy().astype(np.float32)


def predict_semantics(img: Image.Image, model_id: str, device: str):
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    proc = AutoImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
    with torch.no_grad():
        logits = model(**proc(images=img, return_tensors="pt").to(device)).logits
    logits = torch.nn.functional.interpolate(logits, size=img.size[::-1], mode="bilinear", align_corners=False)
    ade = logits.argmax(1)[0].cpu().numpy()

    lut = np.zeros(len(model.config.id2label), dtype=np.uint16)  # mặc định 0 = void
    for ade_id, name in model.config.id2label.items():
        name_l = name.lower()
        tokens = set(re.findall(r"[a-z]+", name_l))
        for kw, rid in ADE_TO_RELLIS:
            if (" " in kw and kw in name_l) or kw in tokens:
                lut[int(ade_id)] = rid
                break
    return lut[ade]


# Hình học
def backproject(depth, labels, rgb, fx, fy, cx, cy, max_depth):
    """Pixel -> điểm 3D trong hệ LiDAR (x trước, y trái, z lên), bỏ trời và điểm quá xa."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    valid = np.isfinite(depth) & (depth > 0.3) & (depth < max_depth) & (labels != SKY_ID)
    z = depth[valid]
    xc = (u[valid] - cx) * z / fx      # camera: x phải
    yc = (v[valid] - cy) * z / fy      # camera: y xuống
    pts = np.stack([z, -xc, -yc], axis=1)
    return pts, labels[valid], rgb[valid]


def fit_ground_plane(pts, iters=300, thresh=0.08, seed=0):
    """RANSAC mặt phẳng n·p + d = 0, n hướng lên (n_z > 0)."""
    rng = np.random.default_rng(seed)
    best_inl, best = -1, None
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if n[2] < 0:
            n = -n
        if n[2] < 0.5:            # bỏ mặt phẳng nghiêng quá 60° (không phải mặt đất)
            continue
        d = -n @ s[0]
        inl = np.count_nonzero(np.abs(pts @ n + d) < thresh)
        if inl > best_inl:
            best_inl, best = inl, (n, d)
    if best is None:
        raise RuntimeError("RANSAC không tìm được mặt đất")
    n, d = best
    # Tinh chỉnh bằng bình phương tối thiểu trên các inlier
    inl_pts = pts[np.abs(pts @ n + d) < thresh]
    c = inl_pts.mean(0)
    n = np.linalg.svd(inl_pts - c, full_matrices=False)[2][-1]
    if n[2] < 0:
        n = -n
    return n, -n @ c, len(inl_pts) / len(pts)


def rotation_to_z(n):
    """Ma trận xoay đưa vector n về (0, 0, 1) (công thức Rodrigues)."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s, c = np.linalg.norm(v), float(n @ z)
    if s < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def level_to_ground(pts, labels, sensor_height=None):
    """Xoay cho mặt đất nằm ngang; trả về điểm mới và độ cao camera đo được (m)."""
    ground = pts[np.isin(labels, list(GROUND_IDS))]
    if len(ground) < 500:
        print("[Mặt đất] Ít điểm có nhãn mặt đất, dùng 40% điểm thấp nhất để fit.")
        ground = pts[pts[:, 2] <= np.quantile(pts[:, 2], 0.4)]
    if len(ground) > 200_000:
        ground = ground[np.random.default_rng(0).choice(len(ground), 200_000, replace=False)]
    n, d, ratio = fit_ground_plane(ground)
    R = rotation_to_z(n)
    out = pts @ R.T
    cam_h = float(d)  # khoảng cách từ camera (gốc) đến mặt phẳng
    tilt = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
    print(f"[Mặt đất] Nghiêng {tilt:.1f}° | camera cao {cam_h:.2f} m so với mặt đất | inlier {ratio:.0%}")
    if sensor_height is not None:
        out[:, 2] += cam_h - sensor_height
        print(f"[Mặt đất] Dịch để cảm biến cao {sensor_height:.2f} m")
    return out, cam_h


def simulate_os1_64(pts, vfov=(-16.6, 16.6), rings=64, h_res=1024, tol_deg=0.06):
    """Giữ điểm nằm trên 64 tia dọc x 1024 góc ngang (1 điểm gần nhất mỗi ô) như OS1-64."""
    rng_xy = np.hypot(pts[:, 0], pts[:, 1])
    elev = np.degrees(np.arctan2(pts[:, 2], rng_xy))
    step = (vfov[1] - vfov[0]) / (rings - 1)
    ring = np.round((elev - vfov[0]) / step).astype(np.int64)
    on_ring = (ring >= 0) & (ring < rings) & (np.abs(elev - (vfov[0] + ring * step)) < tol_deg)
    az = np.arctan2(pts[:, 1], pts[:, 0])
    az_bin = np.floor((az + np.pi) / (2 * np.pi) * h_res).astype(np.int64) % h_res
    idx = np.flatnonzero(on_ring)
    key = ring[idx] * h_res + az_bin[idx]
    r3 = np.linalg.norm(pts[idx], axis=1)
    order = np.lexsort((r3, key))
    _, first = np.unique(key[order], return_index=True)
    keep = idx[order[first]]
    print(f"[OS1-64] Giữ {len(keep):,}/{len(pts):,} điểm trên {len(np.unique(ring[keep]))} tia")
    return keep


# Lưu file
def save_outputs(pts, labels, rgb, out_root: Path, seq: str, frame: str):
    bin_dir = out_root / seq / "os1_cloud_node_kitti_bin"
    lbl_dir = out_root / seq / "os1_cloud_node_semantickitti_label_id"
    bin_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    intensity = (rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)) / 255.0
    np.hstack([pts, intensity[:, None]]).astype(np.float32).tofile(bin_dir / f"{frame}.bin")
    labels.astype(np.uint32).tofile(lbl_dir / f"{frame}.label")

    ply = out_root / f"{seq}_{frame}.ply"
    with open(ply, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property ushort label\nend_header\n")
        for p, c, l in zip(pts, rgb, labels):
            f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {c[0]} {c[1]} {c[2]} {l}\n")
    print(f"[Lưu] {bin_dir / (frame + '.bin')}\n[Lưu] {lbl_dir / (frame + '.label')}\n[Lưu] {ply}")


def main():
    ap = argparse.ArgumentParser(description="Ảnh RGB đời thực -> đám mây điểm kiểu LiDAR RELLIS-3D")
    ap.add_argument("image", help="Đường dẫn ảnh .jpg/.png")
    ap.add_argument("--out", default="raw_photo", help="Thư mục gốc đầu ra (dùng làm --data-dir)")
    ap.add_argument("--seq", default="00000")
    ap.add_argument("--frame", default="000000")
    ap.add_argument("--hfov", type=float, default=70.0, help="FOV ngang (độ) khi ảnh không có EXIF")
    ap.add_argument("--max-depth", type=float, default=35.0, help="Bỏ điểm xa hơn (m)")
    ap.add_argument("--sensor-height", type=float, default=None,
                    help="Ép độ cao cảm biến (m). Mặc định giữ độ cao camera đo được")
    ap.add_argument("--simulate-os1", action="store_true", help="Lấy mẫu thưa theo 64 tia OS1-64")
    ap.add_argument("--max-width", type=int, default=1280, help="Thu nhỏ ảnh để chạy nhanh")
    ap.add_argument("--depth-model", default=DEPTH_MODEL)
    ap.add_argument("--seg-model", default=SEG_MODEL)
    ap.add_argument("--device", default=None, help="cuda / cpu (mặc định tự chọn)")
    args = ap.parse_args()

    device = args.device
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    img = Image.open(args.image)
    fx, fy, cx, cy = estimate_intrinsics(img, args.hfov)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if img.width > args.max_width:
        s = args.max_width / img.width
        img = img.resize((args.max_width, round(img.height * s)), Image.BICUBIC)
        fx, fy = fx * s, fy * s
    cx, cy = img.width / 2.0, img.height / 2.0
    print(f"[Ảnh] {img.width}x{img.height} | thiết bị: {device}")

    print("[1/4] Ước lượng độ sâu...")
    depth = predict_depth(img, args.depth_model, device)
    print(f"      độ sâu {np.nanmin(depth):.1f} – {np.nanmax(depth):.1f} m")
    print("[2/4] Phân đoạn ngữ nghĩa...")
    labels = predict_semantics(img, args.seg_model, device)

    print("[3/4] Chiếu ngược & cân mặt đất...")
    pts, lbl, rgb = backproject(depth, labels, np.asarray(img), fx, fy, cx, cy, args.max_depth)
    pts, _ = level_to_ground(pts, lbl, args.sensor_height)

    if args.simulate_os1:
        keep = simulate_os1_64(pts)
        pts, lbl, rgb = pts[keep], lbl[keep], rgb[keep]

    print("[4/4] Lưu kết quả...")
    ids, cnt = np.unique(lbl, return_counts=True)
    print("      Nhãn RELLIS:", {int(i): int(c) for i, c in zip(ids, cnt)})
    save_outputs(pts, lbl, rgb, Path(args.out), args.seq, args.frame)


if __name__ == "__main__":
    main()
