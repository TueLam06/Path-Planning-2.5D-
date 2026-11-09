#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Công cụ kiểm tra cấu trúc file .npz và tính chỉ số định lượng trên tập dữ liệu 2.5D đã sinh.
"""

import glob
from pathlib import Path
import numpy as np

from config import RELLIS_NAMES


def inspect_npz(npz_path):
    """In chi tiết toàn bộ các layer, phân bố độ cao/độ dốc/nhãn và đường đi A* của 1 file .npz."""
    p = Path(npz_path)
    if not p.is_file():
        raise FileNotFoundError(f"Không tìm thấy file .npz: {npz_path}")

    data = np.load(p, allow_pickle=True)
    print("=" * 64)
    print(f">> THÔNG TIN FILE: {p}")
    print("=" * 64)

    print("\n[1. Danh sách các Layer trong tệp]:")
    for k in sorted(data.files):
        val = data[k]
        if isinstance(val, np.ndarray) and val.ndim > 0:
            if val.size > 0:
                print(
                    f"  - {k:<15}: shape = {str(val.shape):<14} dtype = {str(val.dtype):<10} "
                    f"range = [{val.min():.3f}, {val.max():.3f}]"
                )
            else:
                print(f"  - {k:<15}: shape = {str(val.shape):<14} dtype = {str(val.dtype):<10} (empty)")
        else:
            print(f"  - {k:<15}: {val}")

    print("\n[2. Thống kê địa hình]:")
    if "slope" in data:
        slope_deg = np.degrees(data["slope"])
        print(f"  - Độ dốc trung bình    : {slope_deg.mean():.2f}° (max: {slope_deg.max():.2f}°)")
    if "roughness" in data:
        print(
            f"  - Độ gồ ghề trung bình : {data['roughness'].mean():.3f} m "
            f"(max: {data['roughness'].max():.3f} m)"
        )
    if "elevation" in data:
        elev = data["elevation"]
        print(f"  - Cao độ mặt đất z     : [{elev.min():.2f} m -> {elev.max():.2f} m]")
    if "costmap" in data:
        cm = data["costmap"]
        obs_pct = (cm == 255).mean() * 100.0
        print(
            f"  - Costmap (0..255)     : mean = {cm.mean():.1f}, "
            f"ô cấm đi (255) = {(cm == 255).sum():,} ({obs_pct:.1f}%)"
        )

    if "semantic" in data:
        print("\n[3. Thống kê nhãn ngữ nghĩa RELLIS-3D]:")
        sem = data["semantic"]
        unique_sem, counts = np.unique(sem, return_counts=True)
        total_cells = sem.size
        for s_id, cnt in zip(unique_sem, counts):
            name = RELLIS_NAMES.get(int(s_id), "unknown")
            pct = (cnt / total_cells) * 100.0
            print(f"  - Class {int(s_id):2d} ({name:<9}): {cnt:5d} ô ({pct:5.1f}%)")

    if "path_semantic" in data or "path_geom" in data:
        print("\n[4. Đường đi A* nhúng sẵn]:")
        if "path_semantic" in data:
            p_sem = data["path_semantic"]
            print(f"  - Semantic + Geometry : {len(p_sem)} bước")
        if "path_geom" in data:
            p_geo = data["path_geom"]
            print(f"  - Geometry-Only       : {len(p_geo)} bước")
    print("=" * 64)


def evaluate_dataset(out_dir="output_25d", sample_step: int = 1):
    """Đánh giá định lượng toàn bộ các file *_25d_dataset.npz trong thư mục kết quả."""
    npz_files = sorted(glob.glob(str(Path(out_dir) / "**" / "*_25d_dataset.npz"), recursive=True))
    if not npz_files:
        npz_files = sorted(glob.glob(str(Path(out_dir) / "*.npz")))

    if not npz_files:
        print(f"[Evaluate] Không tìm thấy file .npz nào trong {out_dir}")
        return None

    sample_files = npz_files[:: max(1, sample_step)]
    total_frames = len(npz_files)
    astar_success_sem = 0
    astar_success_geom = 0
    cost_sem_list = []
    cost_geom_list = []
    high_risk_cells_avoided = 0

    for f in sample_files:
        data = np.load(f, allow_pickle=True)
        if "costmap" not in data or "path_semantic" not in data or "path_geom" not in data:
            continue
        costmap = data["costmap"]
        p_sem = data["path_semantic"]
        p_geo = data["path_geom"]

        if len(p_sem) > 0:
            astar_success_sem += 1
            cost_sem_list.append(float(np.mean([costmap[pt[0], pt[1]] for pt in p_sem])))
        if len(p_geo) > 0:
            astar_success_geom += 1
            cost_geom_list.append(float(np.mean([costmap[pt[0], pt[1]] for pt in p_geo])))
            geo_danger = sum(1 for pt in p_geo if costmap[pt[0], pt[1]] >= 180)
            sem_danger = (
                sum(1 for pt in p_sem if costmap[pt[0], pt[1]] >= 180) if len(p_sem) > 0 else 0
            )
            high_risk_cells_avoided += max(0, geo_danger - sem_danger)

    n_eval = len(sample_files)
    avg_sem = float(np.mean(cost_sem_list)) if cost_sem_list else 0.0
    avg_geom = float(np.mean(cost_geom_list)) if cost_geom_list else 0.0
    reduction_pct = ((avg_geom - avg_sem) / avg_geom * 100.0) if avg_geom > 0 else 0.0

    print("=" * 64)
    print(" ĐÁNH GIÁ ĐỊNH LƯỢNG BỘ DỮ LIỆU 2.5D RELLIS-3D")
    print("=" * 64)
    print(f" - Tổng số file .npz tìm thấy   : {total_frames:,}")
    print(f" - Số file đem vào đánh giá      : {n_eval:,}")
    if cost_sem_list or cost_geom_list:
        print(f" - Tỉ lệ thành công Semantic A*  : {astar_success_sem / max(1, n_eval) * 100:.1f}%")
        print(f" - Tỉ lệ thành công Geo-Only A*  : {astar_success_geom / max(1, n_eval) * 100:.1f}%")
        print(f" - Cost trung bình Semantic A*   : {avg_sem:.2f} / 255")
        print(f" - Cost trung bình Geo-Only A*   : {avg_geom:.2f} / 255")
        print(f" - Mức giảm thiểu rủi ro địa hình: {reduction_pct:.2f}%")
        print(f" - Số ô rủi ro cao (>=180) né được: {high_risk_cells_avoided}")
    else:
        print(" - Tệp .npz không chứa đường đi A* đối chứng (chế độ group1 hoặc rỗng).")
    print("=" * 64)
    return {
        "total_frames": total_frames,
        "evaluated": n_eval,
        "success_sem": astar_success_sem,
        "success_geom": astar_success_geom,
        "avg_cost_sem": avg_sem,
        "avg_cost_geom": avg_geom,
        "risk_reduction_pct": reduction_pct,
        "high_risk_avoided": high_risk_cells_avoided,
    }

