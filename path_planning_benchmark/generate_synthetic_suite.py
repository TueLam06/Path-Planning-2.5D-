#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sinh bộ test tổng hợp có kiểm soát độ khó: bản đồ 2.5D .npz + test case start/goal thiết kế sẵn.

Họ bản đồ (40 x 30 m, ô 0.2 m, giống lưới RELLIS):
  gap        : tường chắn ngang, khe rộng w (m) -> thử chân đế xe qua khe hẹp
  slope_band : dải dốc ngang góc a (độ) chắn giữa start và goal (dễ; 33° là case âm)
  saddle     : sống núi > 45° chắn ngang, chỉ có 1 đèo rộng w (m), sườn đèo dốc s (độ)
  corridor_s : hành lang uốn chữ S rộng w (m) -> thử rẽ liên tục sát tường
  maze       : mê cung tường, ô rộng c (m) -> thử tìm đường vòng
  boulders   : bãi đá ngẫu nhiên mật độ n -> thử né vật cản
  boulders_dense : bãi đá dày, khe giữa đá cỡ bề rộng xe
  blocked    : tường kín (case âm, không có đường) -> planner phải báo thất bại

Case có feasible=False khi không có đường kể cả với xe coi là 1 điểm, hoặc khe hẹp hơn bề rộng xe.

Ví dụ:
  python generate_synthetic_suite.py
  python run_testsuite.py --cases testsuite/testcases_synthetic.json --modes geo --name synth
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchnav.testsuite import (  # noqa: E402
    build_grid_graph, build_hmap, load_map_data, polyline_len_3d, reconstruct, shortcut_path,
)
from map_utils.hybrid_map import HybridMap  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

W, H, RES = 40.0, 30.0, 0.2
WALL_H = 1.5


def grid():
    nx, ny = int(round(W / RES)), int(round(H / RES))
    xs, ys = np.meshgrid(np.arange(nx) * RES, np.arange(ny) * RES)
    return xs, ys


def base_terrain(xs, ys, rng, amp=0.15):
    """Nền gợn nhẹ để bản đồ không phẳng tuyệt đối."""
    ph = rng.uniform(0, 2 * np.pi, 2)
    return amp * np.sin(xs * 0.3 + ph[0]) * np.cos(ys * 0.25 + ph[1])


def fam_gap(w, rng):
    xs, ys = grid()
    z = base_terrain(xs, ys, rng)
    wall = (np.abs(xs - 20.0) < 0.5) & (np.abs(ys - 15.0) > w / 2.0)
    z[wall] += WALL_H
    poses = [((6.0, 15.0), (34.0, 15.0)), ((6.0, 6.0), (34.0, 24.0))]
    return z, poses


def fam_slope_band(angle_deg, rng):
    xs, ys = grid()
    z = base_terrain(xs, ys, rng, amp=0.05)
    # Dải x in [12, 28] nghiêng theo y; hai mép làm mượt bằng cos để không tạo bậc
    w = np.clip((xs - 10.0) / 4.0, 0, 1) * np.clip((30.0 - xs) / 4.0, 0, 1)
    w = 0.5 - 0.5 * np.cos(np.pi * w)
    z += np.tan(np.radians(angle_deg)) * (ys - 15.0) * w
    poses = [((5.0, 15.0), (35.0, 15.0)), ((5.0, 8.0), (35.0, 22.0))]
    return z, poses


def fam_maze(cell, rng):
    xs, ys = grid()
    z = base_terrain(xs, ys, rng)
    nc, nr = int(W // cell), int(H // cell)
    # Mê cung DFS trên lưới nc x nr ô; tường dày 0.6 m
    visited = np.zeros((nr, nc), bool)
    walls_v = np.ones((nr, nc + 1), bool)   # tường dọc bên trái ô (r, c)
    walls_h = np.ones((nr + 1, nc), bool)   # tường ngang phía dưới ô (r, c)
    stack = [(0, 0)]
    visited[0, 0] = True
    while stack:
        r, c = stack[-1]
        nbrs = [(r + dr, c + dc) for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))
                if 0 <= r + dr < nr and 0 <= c + dc < nc and not visited[r + dr, c + dc]]
        if not nbrs:
            stack.pop()
            continue
        r2, c2 = nbrs[rng.integers(len(nbrs))]
        if r2 == r:
            walls_v[r, max(c, c2)] = False
        else:
            walls_h[max(r, r2), c] = False
        visited[r2, c2] = True
        stack.append((r2, c2))
    t = 0.3
    for r in range(nr):
        for c in range(nc + 1):
            if walls_v[r, c]:
                z[(np.abs(xs - c * cell) < t) & (ys >= r * cell - t) & (ys <= (r + 1) * cell + t)] += WALL_H
    for r in range(nr + 1):
        for c in range(nc):
            if walls_h[r, c]:
                z[(np.abs(ys - r * cell) < t) & (xs >= c * cell - t) & (xs <= (c + 1) * cell + t)] += WALL_H
    z = np.minimum(z, WALL_H + 0.3)
    half = cell / 2.0
    poses = [((half, half), ((nc - 1) * cell + half, (nr - 1) * cell + half)),
             ((half, (nr - 1) * cell + half), ((nc - 1) * cell + half, half))]
    return z, poses


def fam_boulders(n, rng):
    xs, ys = grid()
    z = base_terrain(xs, ys, rng, amp=0.3)
    for _ in range(n):
        bx, by = rng.uniform(8, 32), rng.uniform(2, 28)
        r = rng.uniform(0.8, 2.0)
        z += rng.uniform(0.6, 1.5) * np.exp(-((xs - bx) ** 2 + (ys - by) ** 2) / (2 * (r / 2) ** 2))
    poses = [((4.0, 15.0), (36.0, 15.0)), ((4.0, 4.0), (36.0, 26.0)), ((4.0, 26.0), (36.0, 4.0))]
    return z, poses


def fam_saddle(param, rng):
    """Sống núi dốc (> 45°) chắn ngang x = 20, chỉ có 1 đèo rộng w (m) với độ dốc sườn s (độ)."""
    w, s = param
    xs, ys = grid()
    z = base_terrain(xs, ys, rng, amp=0.05)
    sigma = 1.5
    h_ridge = 3.0
    h_pass = np.tan(np.radians(s)) * sigma * np.sqrt(np.e)  # sườn Gauss có dốc lớn nhất = s
    # Chuyển mượt từ đèo sang sống núi trong 1.5 m theo y
    t = np.clip((np.abs(ys - 15.0) - w / 2.0) / 1.5, 0, 1)
    h = h_pass + (h_ridge - h_pass) * (0.5 - 0.5 * np.cos(np.pi * t))
    z += h * np.exp(-((xs - 20.0) ** 2) / (2 * sigma ** 2))
    return z, [((6.0, 15.0), (34.0, 15.0)), ((6.0, 4.0), (34.0, 26.0))]


def fam_corridor_s(w, rng):
    """Hành lang uốn chữ S rộng w (m) giữa hai bên tường."""
    from scipy.spatial import cKDTree
    xs, ys = grid()
    z = base_terrain(xs, ys, rng)
    cx = np.linspace(0.0, W, 800)
    cy = 15.0 + 8.0 * np.sin(2 * np.pi * cx / 20.0 + rng.uniform(-0.3, 0.3))
    dist, _ = cKDTree(np.stack([cx, cy], 1)).query(np.stack([xs.ravel(), ys.ravel()], 1))
    z[(dist.reshape(xs.shape) > w / 2.0)] += WALL_H
    i0, i1 = np.searchsorted(cx, 2.0), np.searchsorted(cx, 38.0)
    return z, [((cx[i0], cy[i0]), (cx[i1], cy[i1]))]


def fam_boulders_dense(n, rng):
    """Bãi đá dày: đá cao 1.0–1.8 m, khe giữa đá cỡ bề rộng xe."""
    xs, ys = grid()
    z = base_terrain(xs, ys, rng, amp=0.2)
    for _ in range(n):
        bx, by = rng.uniform(8, 32), rng.uniform(1, 29)
        r = rng.uniform(0.6, 1.2)
        z += rng.uniform(1.0, 1.8) * np.exp(-((xs - bx) ** 2 + (ys - by) ** 2) / (2 * (r / 2) ** 2))
    return z, [((3.0, 15.0), (37.0, 15.0)), ((3.0, 5.0), (37.0, 25.0)), ((3.0, 25.0), (37.0, 5.0))]


def fam_blocked(_, rng):
    xs, ys = grid()
    z = base_terrain(xs, ys, rng)
    z[np.abs(xs - 20.0) < 0.5] += WALL_H
    return z, [((6.0, 15.0), (34.0, 15.0)), ((6.0, 5.0), (34.0, 25.0))]


FAMILIES = {
    "gap": (fam_gap, [1.0, 1.4, 1.8, 2.4, 3.2]),
    "slope_band": (fam_slope_band, [10, 28, 33]),
    "saddle": (fam_saddle, [(2.0, 20), (2.0, 27), (4.0, 20), (4.0, 27), (6.0, 27)]),
    "corridor_s": (fam_corridor_s, [1.2, 1.6, 2.0, 2.8]),
    "maze": (fam_maze, [4.0, 5.0, 6.0]),
    "boulders": (fam_boulders, [15, 30, 50]),
    "boulders_dense": (fam_boulders_dense, [80, 120, 160]),
    "blocked": (fam_blocked, [0]),
}
# Họ mà tham số là bề rộng lối đi: hẹp hơn thân xe thì xe thật không qua được
WIDTH_FAMILIES = {"gap": lambda p: p, "corridor_s": lambda p: p, "saddle": lambda p: p[0]}


def nearest_free(free, ix, iy, max_r=10):
    if free[iy, ix]:
        return ix, iy
    for r in range(1, max_r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = ix + dx, iy + dy
                if 0 <= x < free.shape[1] and 0 <= y < free.shape[0] and free[y, x]:
                    return x, y
    return ix, iy


def make_cases(map_path, rel_path, name, family, param, poses):
    data = load_map_data(map_path)
    hm = build_hmap(data, use_semantic=False)
    free = ~hm.occupancy_2d
    G, node_of_cell, cell_of_node = build_grid_graph(hm.Z, free, hm.resolution)
    cases = []
    for k, ((sx, sy), (gx, gy)) in enumerate(poses):
        si = nearest_free(free, int(round(sx / RES)), int(round(sy / RES)))
        gi = nearest_free(free, int(round(gx / RES)), int(round(gy / RES)))
        s, g = (si[0] * RES, si[1] * RES), (gi[0] * RES, gi[1] * RES)
        yaw = float(np.arctan2(g[1] - s[1], g[0] - s[0]))
        sn, gn = node_of_cell[si[1], si[0]], node_of_cell[gi[1], gi[0]]
        ref = ref_any = None
        if sn >= 0 and gn >= 0:
            dist, pred = dijkstra(G, indices=int(sn), return_predecessors=True)
            if np.isfinite(dist[gn]):
                nodes = reconstruct(pred, int(sn), int(gn))
                cy, cx = np.divmod(cell_of_node[nodes], hm.nx)
                pts = np.stack([cx * RES, cy * RES], axis=1).astype(float)
                ref = round(float(dist[gn]), 3)
                ref_any = round(polyline_len_3d(hm, shortcut_path(hm, pts)), 3)
        euclid = float(np.hypot(g[0] - s[0], g[1] - s[1]))
        cases.append({
            "id": f"{name}_{k:02d}", "map": name, "map_path": rel_path,
            "category": family, "family": family, "param": param, "feasible": ref is not None,
            "start": [round(s[0], 3), round(s[1], 3), round(yaw, 4)],
            "goal": [round(g[0], 3), round(g[1], 3), round(yaw, 4)],
            "euclid_m": round(euclid, 3), "ref_len_3d_m": ref, "ref_len_3d_anyangle_m": ref_any,
            "detour_ratio": None if ref is None else round(ref / max(euclid, 1e-6), 3),
        })
    return cases


def main():
    ap = argparse.ArgumentParser(description="Sinh bản đồ tổng hợp + test case có kiểm soát độ khó")
    ap.add_argument("--out-maps", default=os.path.join("map_data", "synthetic"))
    ap.add_argument("--out-cases", default=os.path.join("testsuite", "testcases_synthetic.json"))
    ap.add_argument("--seeds", type=int, default=2, help="Số bản đồ ngẫu nhiên cho mỗi tham số")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_maps = os.path.join(HERE, args.out_maps)
    os.makedirs(out_maps, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    all_cases, maps_info = [], []
    for family, (fn, params) in FAMILIES.items():
        for param in params:
            for s in range(args.seeds if family != "blocked" else 1):
                z, poses = fn(param, rng)
                ptag = "_".join(map(str, param)) if isinstance(param, tuple) else str(param)
                name = f"syn_{family}_{ptag.replace('.', 'p')}_s{s}"
                path = os.path.join(out_maps, name + ".npz")
                np.savez_compressed(path, elevation=z.astype(np.float32), resolution=RES)
                rel = os.path.relpath(path, HERE).replace("\\", "/")
                cases = make_cases(path, rel, name, family, param, poses)
                if family in WIDTH_FAMILIES and WIDTH_FAMILIES[family](param) < HybridMap.ROBOT_WIDTH:
                    for c in cases:  # lối đi hẹp hơn thân xe: xe thật không qua được
                        c["feasible"] = False
                all_cases += cases
                maps_info.append({"map": name, "map_path": rel, "family": family, "param": param,
                                  "n_cases": len(cases)})
        n_f = sum(c["family"] == family for c in all_cases)
        n_inf = sum(c["family"] == family and not c["feasible"] for c in all_cases)
        print(f"[{family:<10}] {n_f} case ({n_inf} không có đường)")

    out_cases = os.path.join(HERE, args.out_cases)
    with open(out_cases, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "generator": "generate_synthetic_suite.py", "seed": args.seed,
                   "categories": {k: None for k in FAMILIES}, "maps": maps_info, "cases": all_cases},
                  f, ensure_ascii=False, indent=2)
    print(f"[OK] {len(all_cases)} case / {len(maps_info)} bản đồ -> {os.path.relpath(out_cases, HERE)}")


if __name__ == "__main__":
    main()
