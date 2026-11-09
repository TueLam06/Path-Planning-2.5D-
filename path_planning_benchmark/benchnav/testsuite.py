# -*- coding: utf-8 -*-
"""
Hàm dùng chung cho bộ test hàng loạt (generate_testcases.py / run_testsuite.py).

- Nạp bản đồ 2.5D (.npz/.pt/.npy) một lần, dựng HybridMap mới cho mỗi lần chạy
  (các planner gọi clear_around_point nên làm bẩn bản đồ).
- Đồ thị lưới 8 hướng có trọng số = khoảng cách 3D để tính đường tham chiếu
  (shortest path chuẩn trên lưới) bằng scipy.sparse.csgraph.dijkstra.
- Kiểm chứng độc lập đường đi của planner: tới đích, không cắt vật cản.
"""
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from map_utils.map_loader import MapLoader  # noqa: E402
from map_utils.hybrid_map import HybridMap  # noqa: E402

PLANNER_NAMES = ["T-Hybrid A*", "2.5D RRT*", "2.5D Field D*", "2.5D LazyPRM*"]
LETHAL_IDS = (4, 5, 6, 8, 9, 12, 15, 17, 18, 27)
SOFT_IDS = (19, 31, 33, 34)
CLEAR_RADIUS = 1.2   # bán kính lớn nhất mà các planner tự dọn quanh start/goal
ROLL_LIMIT_DEG = float(np.degrees(0.55))


def load_map_data(path):
    """Đọc file bản đồ một lần, trả về dict dùng lại để dựng HybridMap."""
    X, Y, Z, res, meta = MapLoader().load_map(path, return_metadata=True)
    return {"X": X, "Y": Y, "Z": Z, "res": res, "meta": meta, "path": path}


def build_hmap(data, use_semantic=True):
    meta = data["meta"]
    return HybridMap(
        data["X"], data["Y"], data["Z"],
        resolution=data["res"],
        semantic=meta.get("semantic"),
        costmap=meta.get("costmap"),
        observed_mask=meta.get("observed_mask"),
        use_semantic=use_semantic,
    )


def build_planner(name, hm):
    """Cùng tham số với run_benchmark.py để kết quả so sánh được."""
    from planners.thybrid_a_star import THybridAStar
    from planners.rrt_star_2_5d import RRTStar25D
    from planners.field_d_star_2_5d import FieldDStar25D
    from planners.lazy_prm_star_2_5d import LazyPRMStar25D

    if name == "T-Hybrid A*":
        return THybridAStar(hm, max_iterations=4000)
    if name == "2.5D RRT*":
        return RRTStar25D(hm, max_iterations=1200)
    if name == "2.5D Field D*":
        return FieldDStar25D(hm)
    if name == "2.5D LazyPRM*":
        return LazyPRMStar25D(hm, num_samples=1800)
    raise ValueError(f"Planner không hỗ trợ: {name}")


# ----------------------------------------------------------------------------
# Đồ thị lưới & đường tham chiếu
# ----------------------------------------------------------------------------
def build_grid_graph(Z, free, res):
    """
    Đồ thị 8 hướng trên các ô free. Trọng số cạnh = độ dài 3D giữa tâm 2 ô.
    :return: (csr_graph, node_of_cell[ny,nx] (-1 nếu không phải node), cell_of_node[flat])
    """
    ny, nx = free.shape
    node_of_cell = -np.ones((ny, nx), dtype=np.int64)
    cell_of_node = np.flatnonzero(free.ravel())
    node_of_cell.ravel()[cell_of_node] = np.arange(len(cell_of_node))

    rows, cols, wts = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ya, yb = slice(0, ny - dy), slice(dy, ny)
        if dx >= 0:
            xa, xb = slice(0, nx - dx), slice(dx, nx)
        else:
            xa, xb = slice(-dx, nx), slice(0, nx + dx)
        m = free[ya, xa] & free[yb, xb]
        a = node_of_cell[ya, xa][m]
        b = node_of_cell[yb, xb][m]
        dz = (Z[yb, xb] - Z[ya, xa])[m]
        d = np.sqrt((res * np.hypot(dx, dy)) ** 2 + dz ** 2)
        rows += [a, b]
        cols += [b, a]
        wts += [d, d]
    n = len(cell_of_node)
    G = coo_matrix(
        (np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    ).tocsr()
    return G, node_of_cell, cell_of_node


def reconstruct(pred, src, dst):
    path = [dst]
    while path[-1] != src:
        p = pred[path[-1]]
        if p < 0:
            return None
        path.append(p)
    return path[::-1]


# ----------------------------------------------------------------------------
# Kiểm chứng đường đi
# ----------------------------------------------------------------------------
def verify_path(path, start, goal, data, use_semantic, goal_tol):
    """
    Kiểm tra độc lập với cờ success mà planner tự báo:
      - reached: điểm cuối cách goal <= goal_tol (m)
      - collisions: số mẫu (bước res/2 dọc từng đoạn) rơi vào ô vật cản / ngoài bản đồ
      - lethal/soft hits: số waypoint nằm trên lớp ngữ nghĩa nguy hiểm
    """
    out = {"reached": False, "goal_error_m": None, "collision_samples": None,
           "lethal_waypoint_hits": None, "soft_hazard_hits": None, "mean_costmap": None,
           "max_roll_fp_deg": None, "max_pitch_fp_deg": None}
    if not path or len(path) < 2:
        return out

    hm = build_hmap(data, use_semantic)
    hm.clear_around_point(start[0], start[1], radius=CLEAR_RADIUS)
    hm.clear_around_point(goal[0], goal[1], radius=CLEAR_RADIUS)

    pts = np.array([[p[0], p[1]] for p in path], dtype=float)
    err = float(np.hypot(pts[-1, 0] - goal[0], pts[-1, 1] - goal[1]))
    out["goal_error_m"] = err
    out["reached"] = err <= goal_tol

    step = hm.resolution * 0.125   # cùng bước với HybridMap.is_segment_free
    collisions = 0
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(int(np.ceil(np.hypot(*(b - a)) / step)), 1)
        for t in np.linspace(0.0, 1.0, n + 1):
            x, y = a + t * (b - a)
            if hm.is_occupied_2d(x, y):
                collisions += 1
    out["collision_samples"] = collisions

    # Roll/pitch đo THỐNG NHẤT cho mọi planner (chân đế 3 điểm, yaw theo hướng di chuyển).
    # Mỗi planner tự báo roll bằng cách khác nhau (pháp tuyến ô vs chân đế) nên không so được.
    rolls, pitches = [], []
    for i in range(len(pts)):
        a, b = (pts[i], pts[i + 1]) if i + 1 < len(pts) else (pts[i - 1], pts[i])
        yaw = float(np.arctan2(b[1] - a[1], b[0] - a[0])) if np.hypot(*(b - a)) > 1e-6 else float(path[i][3])
        r, p, _ = hm.project_footprint_3point(pts[i, 0], pts[i, 1], yaw)
        rolls.append(abs(r))
        pitches.append(abs(p))
    out["max_roll_fp_deg"] = float(np.degrees(max(rolls)))
    out["max_pitch_fp_deg"] = float(np.degrees(max(pitches)))

    meta = data["meta"]
    sem, cm = meta.get("semantic"), meta.get("costmap")
    lethal = soft = 0
    cm_vals = []
    for x, y in pts:
        g = hm.world_to_grid(x, y)
        if g is None:
            continue
        ix, iy = g
        if sem is not None:
            sid = int(sem[iy, ix])
            lethal += sid in LETHAL_IDS
            soft += sid in SOFT_IDS
        if cm is not None:
            cm_vals.append(float(cm[iy, ix]))
    if sem is not None:
        out["lethal_waypoint_hits"] = int(lethal)
        out["soft_hazard_hits"] = int(soft)
    if cm_vals:
        out["mean_costmap"] = float(np.mean(cm_vals))
    return out
