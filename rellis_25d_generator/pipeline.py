#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline xử lý RELLIS-3D Point Cloud -> Bản đồ 2.5D (.npz) và tìm đường A*.

Cài đặt vector hóa hoàn toàn bằng NumPy & SciPy cho tốc độ cao, kèm sửa lỗi
nội suy vùng mù cảm biến (blind spot semantic propagation) và đóng gói bit an toàn
từ mã nguồn gốc Jikay71, đồng thời hỗ trợ chế độ tương thích Group 1 (`--mode group1`).
"""

import heapq
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, generate_binary_structure

from config import (
    GROUP1_OBSTACLE_IDS,
    GROUP1_SEMANTIC_COST_TABLE,
    LETHAL_CLASSES,
    RellisConfig,
    SEMANTIC_COST_TABLE,
)
from data_loader import RellisDataLoader
from visualizer import visualize_dashboard


def preprocess_point_cloud(pts, lbls, cfg: RellisConfig, filter_ego: bool = True):
    """
    Bước 1: Lọc ROI không gian quanh xe và (tùy chọn) loại bỏ thân xe Polaris W-MRZR + người lái.
    """
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    valid_mask = (
        (x >= cfg.X_MIN) & (x < cfg.X_MAX)
        & (y >= cfg.Y_MIN) & (y < cfg.Y_MAX)
        & (z >= cfg.Z_MIN) & (z < cfg.Z_MAX)
    )

    if filter_ego:
        in_vehicle_box = (
            (x >= cfg.VEHICLE_X_MIN) & (x <= cfg.VEHICLE_X_MAX)
            & (y >= cfg.VEHICLE_Y_MIN) & (y <= cfg.VEHICLE_Y_MAX)
            & (z >= cfg.VEHICLE_Z_MIN)
        )
        valid_mask &= ~in_vehicle_box

    return (
        x[valid_mask],
        y[valid_mask],
        z[valid_mask],
        pts[valid_mask, 3],
        lbls[valid_mask],
    )


def _majority_vote_labels(cell_flat_idx, lbls, total_cells: int, lethal_classes=None):
    """
    Bầu chọn nhãn ngữ nghĩa đa số cho từng ô (dùng bit-packing << 16 an toàn cho mọi SemanticKITTI ID),
    tùy chọn ưu tiên tuyệt đối các lớp vật cản nguy hiểm (lethal).
    """
    packed = (cell_flat_idx.astype(np.int64) << 16) | (lbls.astype(np.int64) & 0xFFFF)
    u_packed, u_counts = np.unique(packed, return_counts=True)
    u_cell = u_packed >> 16
    u_lbl = (u_packed & 0xFFFF).astype(np.uint16)

    order = np.lexsort((-u_counts, u_cell))
    u_cell_sorted = u_cell[order]
    u_lbl_sorted = u_lbl[order]
    _, first_idx = np.unique(u_cell_sorted, return_index=True)

    majority_flat = np.zeros(total_cells, dtype=np.uint16)
    majority_flat[u_cell_sorted[first_idx]] = u_lbl_sorted[first_idx]

    if lethal_classes:
        is_lethal_pt = np.isin(lbls, list(lethal_classes))
        if np.any(is_lethal_pt):
            majority_flat[cell_flat_idx[is_lethal_pt]] = lbls[is_lethal_pt].astype(np.uint16)

    return majority_flat


def build_elevation_grid(x, y, z, lbls, cfg: RellisConfig):
    """
    Bước 2 (Pipeline Jikay71): Chuyển Point Cloud -> lưới BEV (200x150).
      - elevation: z_min mỗi ô + ước lượng vùng mù (R < 3.5m) + nội suy lấp đầy EDT
      - slope_rad: tính từ gradient của elevation sau Gaussian smoothing (sigma=1.2)
      - roughness: std(z) trong ô quan sát được
      - step_height: z_max - z_min trong ô quan sát được
      - semantic_filled: bầu chọn đa số (ưu tiên vật cản) + nội suy lấp vùng mù/ô trống
    """
    H, W = cfg.GRID_H, cfg.GRID_W
    total_cells = H * W

    row_idx = np.clip(np.floor((cfg.X_MAX - x) / cfg.RES).astype(np.int32), 0, H - 1)
    col_idx = np.clip(np.floor((cfg.Y_MAX - y) / cfg.RES).astype(np.int32), 0, W - 1)
    cell_flat_idx = row_idx * W + col_idx

    count = np.zeros(total_cells, dtype=np.int32)
    np.add.at(count, cell_flat_idx, 1)
    valid_cell_mask = count > 0

    z_min_flat = np.full(total_cells, np.inf, dtype=np.float32)
    z_max_flat = np.full(total_cells, -np.inf, dtype=np.float32)
    z_sum_flat = np.zeros(total_cells, dtype=np.float32)
    z_sq_sum_flat = np.zeros(total_cells, dtype=np.float32)

    np.minimum.at(z_min_flat, cell_flat_idx, z)
    np.maximum.at(z_max_flat, cell_flat_idx, z)
    np.add.at(z_sum_flat, cell_flat_idx, z)
    np.add.at(z_sq_sum_flat, cell_flat_idx, z ** 2)

    z_mean_flat = np.zeros(total_cells, dtype=np.float32)
    roughness_flat = np.zeros(total_cells, dtype=np.float32)
    step_height_flat = np.zeros(total_cells, dtype=np.float32)
    elevation_flat = np.zeros(total_cells, dtype=np.float32)

    if np.any(valid_cell_mask):
        z_mean_flat[valid_cell_mask] = z_sum_flat[valid_cell_mask] / count[valid_cell_mask]
        var_z = (z_sq_sum_flat[valid_cell_mask] / count[valid_cell_mask]) - (
            z_mean_flat[valid_cell_mask] ** 2
        )
        roughness_flat[valid_cell_mask] = np.sqrt(np.maximum(0.0, var_z))
        step_height_flat[valid_cell_mask] = (
            z_max_flat[valid_cell_mask] - z_min_flat[valid_cell_mask]
        )
        elevation_flat[valid_cell_mask] = z_min_flat[valid_cell_mask]

    elevation = elevation_flat.reshape(H, W)
    roughness = roughness_flat.reshape(H, W)
    step_height = step_height_flat.reshape(H, W)
    observed_mask = valid_cell_mask.reshape(H, W)

    semantic_grid = _majority_vote_labels(
        cell_flat_idx, lbls, total_cells, lethal_classes=LETHAL_CLASSES
    ).reshape(H, W)

    # Xử lý vùng mù cảm biến dưới và sát thân xe (R < 3.5m)
    Y_grid, X_grid = np.ogrid[:H, :W]
    r_center = int(round((cfg.X_MAX - 0.0) / cfg.RES))
    c_center = int(round((cfg.Y_MAX - 0.0) / cfg.RES))
    dist_from_sensor = np.sqrt(
        ((Y_grid - r_center) * cfg.RES) ** 2 + ((X_grid - c_center) * cfg.RES) ** 2
    )

    near_ground_mask = observed_mask & (dist_from_sensor >= 3.0) & (dist_from_sensor <= 6.5)
    if np.any(near_ground_mask):
        estimated_ground_z = float(np.median(elevation[near_ground_mask]))
        candidates = semantic_grid[near_ground_mask]
        non_void = candidates[candidates != 0]
        if len(non_void) > 0:
            ground_class = int(np.bincount(non_void).argmax())
        else:
            ground_class = int(np.bincount(candidates).argmax())
    else:
        estimated_ground_z = 0.0
        ground_class = 23  # concrete fallback

    blind_spot = (~observed_mask) & (dist_from_sensor < 3.5)
    elevation[blind_spot] = estimated_ground_z

    semantic_filled = semantic_grid.copy()
    semantic_filled[blind_spot] = ground_class

    # Nội suy lấp đầy ô trống bằng Nearest Neighbor (EDT)
    elevation_filled = elevation.copy()
    valid_elev_mask = observed_mask | blind_spot
    if np.any(~valid_elev_mask) and np.any(valid_elev_mask):
        indices = distance_transform_edt(
            ~valid_elev_mask, return_distances=False, return_indices=True
        )
        elevation_filled = elevation[indices[0], indices[1]]
        semantic_filled = semantic_filled[indices[0], indices[1]]

    elevation_smooth = gaussian_filter(elevation_filled, sigma=1.2)
    grad_r, grad_c = np.gradient(elevation_smooth, cfg.RES, cfg.RES)
    slope_rad = np.arctan(np.sqrt(grad_r ** 2 + grad_c ** 2)).astype(np.float32)

    return elevation_filled, slope_rad, roughness, step_height, semantic_filled, observed_mask


def compute_costmap(
    elevation, slope_rad, roughness, step_height, semantic_grid, observed_mask, cfg: RellisConfig
):
    """
    Bước 3 (Pipeline Jikay71): Hợp nhất Ngữ nghĩa + Hình học -> Costmap (0..255).
    """
    H, W = cfg.GRID_H, cfg.GRID_W

    semantic_cost = np.zeros((H, W), dtype=np.float32)
    for cls_id, base_cost in SEMANTIC_COST_TABLE.items():
        semantic_cost[semantic_grid == cls_id] = base_cost

    max_slope_rad = np.radians(cfg.MAX_SLOPE_DEG)
    slope_cost = np.clip(slope_rad / max_slope_rad, 0.0, 1.0) * 255.0
    roughness_cost = np.clip(roughness / cfg.MAX_ROUGHNESS, 0.0, 1.0) * 255.0

    costmap = (
        cfg.W_SEM * semantic_cost
        + cfg.W_SLOPE * slope_cost
        + cfg.W_ROUGH * roughness_cost
    )

    lethal_mask = np.isin(semantic_grid, list(LETHAL_CLASSES))
    lethal_mask |= observed_mask & (slope_rad >= max_slope_rad)
    lethal_mask |= observed_mask & (step_height >= cfg.MAX_STEP_HEIGHT)

    # Bảo vệ footprint xe tại gốc xuất phát luôn khả hành
    Y, X = np.ogrid[:H, :W]
    r_center = int(round((cfg.X_MAX - 0.5) / cfg.RES))
    c_center = int(round((cfg.Y_MAX - 0.0) / cfg.RES))
    vehicle_footprint = (np.abs((Y - r_center) * cfg.RES) <= 2.2) & (
        np.abs((X - c_center) * cfg.RES) <= 1.4
    )
    lethal_mask[vehicle_footprint] = False

    struct = generate_binary_structure(2, 1)
    if cfg.INFLATION_CELLS > 0:
        inflated_lethal = binary_dilation(
            lethal_mask, structure=struct, iterations=cfg.INFLATION_CELLS
        )
        inflated_lethal[vehicle_footprint] = False
        near_obstacle = inflated_lethal & (~lethal_mask)
        costmap[near_obstacle] = np.maximum(costmap[near_obstacle], 220.0)

    costmap[lethal_mask] = 255.0
    costmap[vehicle_footprint] = np.minimum(costmap[vehicle_footprint], 25.0)
    costmap_uint8 = np.clip(costmap, 0, 255).astype(np.uint8)

    # Costmap đối chứng chỉ dùng hình học (Geometry-Only)
    geom_cost = 0.6 * slope_cost + 0.4 * roughness_cost
    geom_lethal = (observed_mask & (slope_rad >= max_slope_rad)) | (
        observed_mask & (step_height >= cfg.MAX_STEP_HEIGHT)
    )
    geom_lethal[vehicle_footprint] = False
    if cfg.INFLATION_CELLS > 0:
        inflated_geom = binary_dilation(
            geom_lethal, structure=struct, iterations=cfg.INFLATION_CELLS
        )
        inflated_geom[vehicle_footprint] = False
        near_geom = inflated_geom & (~geom_lethal)
        geom_cost[near_geom] = np.maximum(geom_cost[near_geom], 220.0)
    geom_cost[geom_lethal] = 255.0
    geom_cost[vehicle_footprint] = np.minimum(geom_cost[vehicle_footprint], 25.0)
    costmap_geom_only = np.clip(geom_cost, 0, 255).astype(np.uint8)

    return costmap_uint8, costmap_geom_only


def build_group1_maps(x, y, z, lbls, cfg: RellisConfig):
    """
    Pipeline chuẩn báo cáo Nhóm 1 (reproduce chính xác 000000.npz..000005.npz):
      - Lưới (H=150 theo Y, W=200 theo X), elevation = Z phân vị 25%, roughness = std(Z)
      - Không nội suy lấp lỗ (valid_mask đánh dấu ô có điểm)
      - slope = arctan(|grad(elevation, res)|)
      - semantic = majority vote
      - costmap = 0.5*sem_cost + 0.3*(clip(slope/0.8,0,1)*255) + 0.2*(clip(roughness/0.5,0,1)*255)
    """
    H, W = cfg.GROUP1_H, cfg.GROUP1_W
    total_cells = H * W

    col_idx = np.clip(np.floor((x - cfg.X_MIN) / cfg.RES).astype(np.int32), 0, W - 1)
    row_idx = np.clip(np.floor((y - cfg.Y_MIN) / cfg.RES).astype(np.int32), 0, H - 1)
    cell_flat_idx = row_idx * W + col_idx

    elevation_flat = np.zeros(total_cells, dtype=np.float32)
    roughness_flat = np.zeros(total_cells, dtype=np.float32)
    valid_flat = np.zeros(total_cells, dtype=np.uint8)

    # Nhóm điểm theo cell chỉ bằng argsort (vector hóa hoàn toàn)
    order = np.argsort(cell_flat_idx, kind="mergesort")
    sorted_cells = cell_flat_idx[order]
    sorted_z = z[order]
    unique_cells, split_starts, counts = np.unique(
        sorted_cells, return_index=True, return_counts=True
    )

    for idx, start, cnt in zip(unique_cells, split_starts, counts):
        z_cell = sorted_z[start : start + cnt]
        elevation_flat[idx] = np.percentile(z_cell, 25)
        roughness_flat[idx] = np.std(z_cell)
        valid_flat[idx] = 1

    elevation = elevation_flat.reshape(H, W)
    roughness = roughness_flat.reshape(H, W)
    valid_mask = valid_flat.reshape(H, W)

    gy, gx = np.gradient(elevation, cfg.RES, cfg.RES)
    slope = np.arctan(np.sqrt(gx ** 2 + gy ** 2)).astype(np.float32)

    semantic_flat = _majority_vote_labels(cell_flat_idx, lbls, total_cells, lethal_classes=None)
    for obs_id in sorted(GROUP1_OBSTACLE_IDS):
        is_obs = lbls == obs_id
        if np.any(is_obs):
            semantic_flat[cell_flat_idx[is_obs]] = obs_id
    semantic = semantic_flat.reshape(H, W)

    sem_cost = np.zeros((H, W), dtype=np.float32)
    for cls_id, cost_val in GROUP1_SEMANTIC_COST_TABLE.items():
        sem_cost[semantic == cls_id] = cost_val
    obs_mask = np.isin(semantic, list(GROUP1_OBSTACLE_IDS))
    sem_cost[obs_mask] = 255.0

    slope_norm = np.clip(slope / 0.8, 0.0, 1.0) * 255.0
    rough_norm = np.clip(roughness / 0.5, 0.0, 1.0) * 255.0

    costmap_f = 0.5 * sem_cost + 0.3 * slope_norm + 0.2 * rough_norm
    costmap_f[obs_mask] = 255.0
    costmap = np.clip(costmap_f, 0.0, 255.0).astype(np.uint8)

    return elevation, slope, roughness, semantic, costmap, valid_mask


def find_free_cell(costmap, r: int, c: int, max_r: int = 25, thresh: int = 240):
    """Tìm ô khả hành gần (r, c) nhất trong bán kính max_r."""
    H, W = costmap.shape
    r = int(np.clip(r, 0, H - 1))
    c = int(np.clip(c, 0, W - 1))
    if costmap[r, c] < thresh:
        return (r, c)
    for rad in range(1, max_r + 1):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and costmap[nr, nc] < thresh:
                    return (nr, nc)
    return (r, c)


def astar_search(costmap, start_rc, goal_rc, max_traversable_cost: int = 240):
    """
    Tìm đường A* 8 hướng trên costmap 2.5D:
      - Chi phí bước = khoảng cách * (1.0 + cost/255 * 8.0)
    """
    H, W = costmap.shape
    sr, sc = find_free_cell(costmap, start_rc[0], start_rc[1], thresh=max_traversable_cost)
    gr, gc = find_free_cell(costmap, goal_rc[0], goal_rc[1], thresh=max_traversable_cost)

    def heuristic(r1, c1, r2, c2):
        dr = abs(r1 - r2)
        dc = abs(c1 - c2)
        return (dr + dc) + (1.414 - 2.0) * min(dr, dc)

    moves = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1.414),
        (-1, 1, 1.414),
        (1, -1, 1.414),
        (1, 1, 1.414),
    ]

    open_set = [(0.0, 0, sr, sc)]
    came_from = {}
    g_score = {(sr, sc): 0.0}
    visited = set()
    counter = 1

    while open_set:
        _, _, curr_r, curr_c = heapq.heappop(open_set)
        if (curr_r, curr_c) == (gr, gc):
            path = []
            cur = (gr, gc)
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append((sr, sc))
            path.reverse()
            return path

        if (curr_r, curr_c) in visited:
            continue
        visited.add((curr_r, curr_c))

        for dr, dc, base_dist in moves:
            nr, nc = curr_r + dr, curr_c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            cell_cost = costmap[nr, nc]
            if cell_cost >= max_traversable_cost:
                continue

            step_cost = base_dist * (1.0 + (float(cell_cost) / 255.0) * 8.0)
            tentative_g = g_score[(curr_r, curr_c)] + step_cost

            if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                came_from[(nr, nc)] = (curr_r, curr_c)
                g_score[(nr, nc)] = tentative_g
                f_score = tentative_g + heuristic(nr, nc, gr, gc)
                heapq.heappush(open_set, (f_score, counter, nr, nc))
                counter += 1

    return None


def process_single_frame(
    loader: RellisDataLoader,
    sequence_id: str,
    frame_id: str,
    out_dir: Path,
    cfg: RellisConfig,
    mode: str = "jikay",
    save_preview: bool = True,
    skip_existing: bool = False,
    verbose: bool = True,
):
    """Xử lý trọn gói 1 frame -> lưu .npz và ảnh Dashboard (nếu yêu cầu)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "group1":
        npz_file = out_dir / f"{frame_id}.npz"
        preview_file = out_dir / f"{frame_id}_group1_dashboard.png"
    else:
        npz_file = out_dir / f"{sequence_id}_{frame_id}_25d_dataset.npz"
        preview_file = out_dir / f"{sequence_id}_{frame_id}_dashboard.png"

    if skip_existing and npz_file.is_file():
        if not save_preview or preview_file.is_file():
            if verbose:
                print(f"[Bỏ qua] Frame {sequence_id}/{frame_id} đã tồn tại.")
            return {"skipped": True, "npz_path": npz_file}

    t_start = time.time()
    pts, lbls = loader.load_frame(sequence_id, frame_id)

    if verbose:
        print(f"\n========================================================")
        print(f">> XỬ LÝ FRAME [{sequence_id} / {frame_id}] | Chế độ: {mode}")
        print(f"========================================================")
        print(f"[Bước 1] Điểm gốc: {len(pts):,} | Nhãn: {len(lbls):,}")

    if mode == "group1":
        x, y, z, _, lbls_filt = preprocess_point_cloud(pts, lbls, cfg, filter_ego=False)
        if verbose:
            print(f"[Bước 1] Sau lọc ROI: {len(x):,} điểm.")

        t0 = time.time()
        elevation, slope_rad, roughness, semantic_grid, costmap, valid_mask = build_group1_maps(
            x, y, z, lbls_filt, cfg
        )
        t1 = time.time()

        start_rc = (int(round((0.0 - cfg.Y_MIN) / cfg.RES)), int(round((0.0 - cfg.X_MIN) / cfg.RES)))
        goal_rc = (int(round((2.0 - cfg.Y_MIN) / cfg.RES)), int(round((20.0 - cfg.X_MIN) / cfg.RES)))
        path_semantic = astar_search(costmap, start_rc, goal_rc)
        path_geom = None

        np.savez_compressed(
            npz_file,
            elevation=elevation,
            semantic=semantic_grid,
            slope=slope_rad,
            roughness=roughness,
            costmap=costmap,
            valid_mask=valid_mask,
            resolution=cfg.RES,
            x_min=cfg.X_MIN,
            x_max=cfg.X_MAX,
            y_min=cfg.Y_MIN,
            y_max=cfg.Y_MAX,
        )
        if verbose:
            print(f"[Bước 2-3] Dựng lưới Group 1 {elevation.shape} xong trong {(t1 - t0)*1000:.1f} ms.")
    else:
        x, y, z, _, lbls_filt = preprocess_point_cloud(pts, lbls, cfg, filter_ego=True)
        if verbose:
            print(f"[Bước 1] Sau lọc ROI & thân xe: {len(x):,} điểm.")

        t0 = time.time()
        elevation, slope_rad, roughness, step_height, semantic_grid, observed_mask = (
            build_elevation_grid(x, y, z, lbls_filt, cfg)
        )
        t1 = time.time()
        costmap, costmap_geom = compute_costmap(
            elevation, slope_rad, roughness, step_height, semantic_grid, observed_mask, cfg
        )
        t2 = time.time()

        if verbose:
            print(f"[Bước 2] Dựng lưới 2.5D {elevation.shape}: {(t1 - t0)*1000:.1f} ms")
            print(f"[Bước 3] Tính Costmap & Inflation: {(t2 - t1)*1000:.1f} ms")

        start_rc = (
            int(round((cfg.X_MAX - 0.0) / cfg.RES)),
            int(round((cfg.Y_MAX - 0.0) / cfg.RES)),
        )
        goal_rc = (
            int(round((cfg.X_MAX - 20.0) / cfg.RES)),
            int(round((cfg.Y_MAX - 2.0) / cfg.RES)),
        )
        path_semantic = astar_search(costmap, start_rc, goal_rc)
        path_geom = astar_search(costmap_geom, start_rc, goal_rc)

        if verbose:
            if path_semantic:
                csum = sum(int(costmap[r, c]) for r, c in path_semantic)
                print(f"[A* Sem+Geo] {len(path_semantic)} bước | Tổng cost: {csum}")
            else:
                print("[A* Sem+Geo] Không tìm thấy đường.")
            if path_geom:
                csum_g = sum(int(costmap_geom[r, c]) for r, c in path_geom)
                print(f"[A* GeoOnly] {len(path_geom)} bước | Tổng cost: {csum_g}")

        np.savez_compressed(
            npz_file,
            elevation=elevation,
            slope=slope_rad,
            roughness=roughness,
            step_height=step_height,
            semantic=semantic_grid,
            costmap=costmap,
            costmap_geom=costmap_geom,
            observed_mask=observed_mask,
            resolution=cfg.RES,
            x_min=cfg.X_MIN,
            x_max=cfg.X_MAX,
            y_min=cfg.Y_MIN,
            y_max=cfg.Y_MAX,
            path_semantic=np.array(path_semantic) if path_semantic else np.empty(0),
            path_geom=np.array(path_geom) if path_geom else np.empty(0),
        )

    if verbose:
        print(f"[Lưu trữ] Đã lưu .npz tại: {npz_file}")

    if save_preview:
        visualize_dashboard(
            frame_id=f"{sequence_id}-{frame_id}",
            pts_bev={"x": x, "y": y, "z": z},
            elevation=elevation,
            slope_rad=slope_rad,
            roughness=roughness,
            semantic_grid=semantic_grid,
            costmap=costmap,
            path_semantic=path_semantic,
            path_geom=path_geom,
            start_rc=start_rc,
            goal_rc=goal_rc,
            out_fig_path=preview_file,
            cfg=cfg,
            mode=mode,
        )

    t_total = time.time() - t_start
    if verbose:
        print(f">> HOÀN TẤT {frame_id} trong {t_total:.2f}s")

    return {
        "skipped": False,
        "npz_path": npz_file,
        "preview_path": preview_file if save_preview else None,
        "elapsed_sec": t_total,
    }

