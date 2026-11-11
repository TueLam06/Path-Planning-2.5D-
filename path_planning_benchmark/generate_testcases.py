#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sinh test case start/goal: chỉ trên ô đo thật, cách vật cản >= --margin, luôn có đường đi
(Dijkstra lưới 8 hướng, trọng số 3D), chia nhóm short/medium/long theo độ dài đường tham chiếu.

Ví dụ:
  python generate_testcases.py
  python generate_testcases.py --maps "../rellis_25d_generator/output_25d/*.npz" --per-map 12
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import binary_erosion, label
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchnav.testsuite import (  # noqa: E402
    build_grid_graph, build_hmap, load_map_data, polyline_len_3d, reconstruct, shortcut_path,
)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_MAPS = [
    os.path.join("map_data", "*.npz"),
    os.path.join("..", "rellis_25d_generator", "samples", "*.npz"),
]
CATEGORIES = [("short", 8.0, 15.0), ("medium", 15.0, 25.0), ("long", 25.0, float("inf"))]


def heading(p_from, p_to):
    return float(np.arctan2(p_to[1] - p_from[1], p_to[0] - p_from[0]))


def point_along(path_xy, from_start, dist):
    """Điểm trên polyline cách đầu (hoặc cuối) khoảng dist (m)."""
    seq = path_xy if from_start else path_xy[::-1]
    acc = 0.0
    for a, b in zip(seq[:-1], seq[1:]):
        acc += float(np.hypot(*(b - a)))
        if acc >= dist:
            return b
    return seq[-1]


def generate_for_map(map_path, n_cases, rng, margin, border, min_euclid, max_tries,
                     allow_unobserved=False):
    data = load_map_data(map_path)
    hm = build_hmap(data, use_semantic=True)
    res = hm.resolution
    free = ~hm.occupancy_2d

    endpoint = binary_erosion(free, iterations=max(1, int(round(margin / res))), border_value=0)
    b = int(round(border / res))
    endpoint[:b, :] = endpoint[-b:, :] = False
    endpoint[:, :b] = endpoint[:, -b:] = False

    # Tỷ lệ ô đo thật trong vùng free lớn nhất
    lab, n_comp = label(free, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(lab.ravel())[1:]
    main = lab == (int(np.argmax(sizes)) + 1) if n_comp else np.zeros_like(free)
    obs = data["meta"].get("observed_mask")
    has_obs = obs is not None and obs.shape == free.shape
    obs_in_main = float(obs[main].mean()) if has_obs and main.any() else None
    if has_obs and not allow_unobserved:
        endpoint &= obs

    G, node_of_cell, cell_of_node = build_grid_graph(hm.Z, free, res)
    cand_cells = np.flatnonzero(endpoint.ravel())
    if len(cand_cells) < 2:
        return [], {"free_ratio": float(free.mean()), "endpoint_cells": int(len(cand_cells))}
    cand_nodes = node_of_cell.ravel()[cand_cells]
    cy, cx = np.divmod(cand_cells, hm.nx)

    base = n_cases // len(CATEGORIES)
    remaining = {c[0]: base for c in CATEGORIES}
    for c in CATEGORIES[: n_cases - base * len(CATEGORIES)]:
        remaining[c[0]] += 1

    cases = []
    used_starts, used_pairs = set(), set()
    tries = 0
    while sum(remaining.values()) > 0 and tries < max_tries:
        tries += 1
        k = int(rng.integers(len(cand_cells)))
        if k in used_starts:
            continue
        used_starts.add(k)
        s_node = int(cand_nodes[k])
        dist, pred = dijkstra(G, indices=s_node, return_predecessors=True)
        d_goal = dist[cand_nodes]
        eu = np.hypot(cx - cx[k], cy - cy[k]) * res
        ok = np.isfinite(d_goal) & (eu >= min_euclid)

        for cat, lo, hi in sorted(CATEGORIES, key=lambda c: -remaining[c[0]]):
            if remaining[cat] <= 0:
                continue
            m = np.flatnonzero(ok & (d_goal >= lo) & (d_goal < hi))
            if len(m) == 0:
                continue
            j = int(rng.choice(m))
            pair = (min(k, j), max(k, j))
            if pair in used_pairs:
                continue
            used_pairs.add(pair)
            nodes = reconstruct(pred, s_node, int(cand_nodes[j]))
            if nodes is None:
                continue
            gy, gx = np.divmod(cell_of_node[nodes], hm.nx)
            path_xy = np.stack([gx * res, gy * res], axis=1).astype(float)
            sxy, gxy = path_xy[0], path_xy[-1]
            ref_len = float(d_goal[j])
            euclid = float(eu[j])
            cases.append({
                "category": cat,
                "start": [round(sxy[0], 3), round(sxy[1], 3),
                          round(heading(sxy, point_along(path_xy, True, 2.0)), 4)],
                "goal": [round(gxy[0], 3), round(gxy[1], 3),
                         round(heading(point_along(path_xy, False, 2.0), gxy), 4)],
                "euclid_m": round(euclid, 3),
                "ref_len_3d_m": round(ref_len, 3),
                "ref_len_3d_anyangle_m": round(polyline_len_3d(hm, shortcut_path(hm, path_xy)), 3),
                "detour_ratio": round(ref_len / max(euclid, 1e-6), 3),
            })
            remaining[cat] -= 1
            break

    stats = {
        "free_ratio": round(float(free.mean()), 4),
        "observed_ratio": round(float(obs.mean()), 4) if has_obs else None,
        "observed_in_main_free_region": None if obs_in_main is None else round(obs_in_main, 4),
        "endpoint_cells": int(len(cand_cells)),
        "tries": tries,
        "missing": {k: v for k, v in remaining.items() if v > 0},
        "schema": data["meta"].get("schema"),
        "size_m": [round(hm.width, 2), round(hm.height, 2)],
    }
    return cases, stats


def main():
    ap = argparse.ArgumentParser(description="Sinh test case start/goal cho benchmark 2.5D")
    ap.add_argument("--maps", nargs="+", default=DEFAULT_MAPS,
                    help="Glob các file bản đồ (tương đối với thư mục này)")
    ap.add_argument("--maps-list", default=None,
                    help="File text, mỗi dòng 1 đường dẫn bản đồ (thay cho --maps)")
    ap.add_argument("--per-map", type=int, default=9, help="Số case mỗi bản đồ (chia đều 3 nhóm)")
    ap.add_argument("--margin", type=float, default=0.6, help="Khoảng cách tối thiểu tới vật cản (m)")
    ap.add_argument("--border", type=float, default=1.0, help="Khoảng cách tối thiểu tới mép bản đồ (m)")
    ap.add_argument("--min-euclid", type=float, default=6.0, help="Khoảng cách thẳng tối thiểu start-goal (m)")
    ap.add_argument("--max-tries", type=int, default=400, help="Số lần thử chọn start tối đa mỗi bản đồ")
    ap.add_argument("--allow-unobserved", action="store_true",
                    help="Cho phép start/goal ở ô chưa quan sát (mặc định: chỉ ô có điểm LiDAR)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join("testsuite", "testcases.json"))
    args = ap.parse_args()

    paths = []
    if args.maps_list:
        with open(args.maps_list, encoding="utf-8") as f:
            paths = [ln.strip() for ln in f if ln.strip()]
    else:
        for pat in args.maps:
            full = pat if os.path.isabs(pat) else os.path.join(HERE, pat)
            paths += sorted(glob.glob(full))
    if not paths:
        sys.exit("Không tìm thấy bản đồ nào khớp --maps")

    rng = np.random.default_rng(args.seed)
    all_cases, maps_info = [], []
    t0 = time.time()
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            rel = os.path.relpath(p, HERE).replace("\\", "/")
        except ValueError:  # khác ổ đĩa trên Windows
            rel = os.path.abspath(p).replace("\\", "/")
        try:
            cases, stats = generate_for_map(p, args.per_map, rng, args.margin, args.border,
                                            args.min_euclid, args.max_tries,
                                            args.allow_unobserved)
        except Exception as exc:  # bản đồ .pt khi thiếu torch, v.v.
            print(f"[BỎ QUA] {rel}: {exc}")
            continue
        for i, c in enumerate(cases):
            c.update({"id": f"{name}_{i:02d}", "map": name, "map_path": rel})
        all_cases += cases
        maps_info.append({"map": name, "map_path": rel, "n_cases": len(cases), **stats})
        cats = {k: sum(c["category"] == k for c in cases) for k, _, _ in CATEGORIES}
        miss = f" | THIẾU {stats['missing']}" if stats.get("missing") else ""
        oim = stats["observed_in_main_free_region"]
        oim_s = "" if oim is None else f" | đo thật trong vùng free chính {oim:.0%}"
        print(f"[{name:<26}] {stats.get('schema','?'):<6} free {stats['free_ratio']:.0%}{oim_s} "
              f"| {len(cases)} case {cats}{miss}")
        # Bản đồ jikay đã lấp lỗ nên chỉ cảnh báo bản đồ không lấp
        if oim is not None and oim < 0.2 and stats.get("schema") != "jikay":
            print("    [CẢNH BÁO] Vùng free liên thông lớn nhất gần như toàn ô CHƯA QUAN SÁT "
                  "(Z mặc định). Planner đi qua đó là đi trên địa hình bịa -> nên sinh lại "
                  "bản đồ có lấp lỗ (rellis_25d_generator chế độ jikay).")

    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "version": 1,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generator": {k: v for k, v in vars(args).items() if k != "out"},
            "categories": {c: [lo, hi if np.isfinite(hi) else None] for c, lo, hi in CATEGORIES},
            "maps": maps_info,
            "cases": [{k: c[k] for k in ("id", "map", "map_path", "category", "start", "goal",
                                         "euclid_m", "ref_len_3d_m", "ref_len_3d_anyangle_m",
                                         "detour_ratio")}
                      for c in all_cases],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] {len(all_cases)} test case / {len(maps_info)} bản đồ -> {os.path.relpath(out, HERE)}"
          f" ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
