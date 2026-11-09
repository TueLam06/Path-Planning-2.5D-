import os
import time
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for server environments
import matplotlib.pyplot as plt

from map_utils.map_loader import MapLoader
from map_utils.hybrid_map import HybridMap
from map_utils.point_picker import select_start_goal
from planners.thybrid_a_star import THybridAStar
from planners.rrt_star_2_5d import RRTStar25D
from planners.field_d_star_2_5d import FieldDStar25D
from planners.lazy_prm_star_2_5d import LazyPRMStar25D
from benchnav.metrics import PathEvaluator

DEFAULT_LOG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
)

def run_benchnav_map(
    map_name,
    start_pose=None,
    goal_pose=None,
    num_runs=1,
    map_data_dir=None,
    log_dir=None,
    use_semantic=True,
    seed=42,
    suffix=None,
):
    """
    Executes all 4 2.5D path planners on the specified map:
    1. T-Hybrid A* (Liu et al., 2023)
    2. 2.5D RRT* (Steinbauer & Koczka, 2025)
    3. 2.5D Field D* (Ferguson & Stentz, 2005)
    4. 2.5D LazyPRM* / ArtPlanner (Wellhausen & Hutter, 2023)
    """
    loader = MapLoader(map_data_dir=map_data_dir)
    filepath = loader.get_map_path(map_name)
    if not filepath:
        map_name, filepath = loader.select_map_interactive()
    else:
        map_name = os.path.splitext(os.path.basename(filepath))[0]

    X, Y, Z, res, meta = loader.load_map(filepath, return_metadata=True)
    hmap = HybridMap(
        X,
        Y,
        Z,
        resolution=res,
        semantic=meta.get("semantic"),
        costmap=meta.get("costmap"),
        observed_mask=meta.get("observed_mask"),
        use_semantic=use_semantic,
    )

    if start_pose is None or goal_pose is None:
        start_pose, goal_pose = select_start_goal(hmap, map_name)

    mode_label = "Semantic+Geo" if hmap.use_semantic else "Geometry-Only"
    print(f"\n==========================================================================")
    print(f"  CHẠY TẤT CẢ THUẬT TOÁN 2.5D TRÊN MAP: '{map_name}' [{meta.get('schema')} | {mode_label}]")
    print(f"  Kích thước: {hmap.width:.1f}m x {hmap.height:.1f}m ({hmap.nx}x{hmap.ny}) | Start: {start_pose[:2]} -> Goal: {goal_pose[:2]}")
    print(f"==========================================================================\n")

    costmap_ref = meta.get("costmap")
    semantic_ref = meta.get("semantic")

    def make_hmap():
        return HybridMap(
            X,
            Y,
            Z,
            resolution=res,
            semantic=meta.get("semantic"),
            costmap=meta.get("costmap"),
            observed_mask=meta.get("observed_mask"),
            use_semantic=use_semantic,
        )

    def build_planner(name, hm):
        if name == "T-Hybrid A*":
            return THybridAStar(hm, max_iterations=4000)
        if name == "2.5D RRT*":
            return RRTStar25D(hm, max_iterations=1200)
        if name == "2.5D Field D*":
            return FieldDStar25D(hm)
        if name == "2.5D LazyPRM*":
            return LazyPRMStar25D(hm, num_samples=1800)
        raise ValueError(name)

    results = []
    paths_dict = {}
    planner_names = ["T-Hybrid A*", "2.5D RRT*", "2.5D Field D*", "2.5D LazyPRM*"]
    per_algo_runs = {name: [] for name in planner_names}

    for run_idx in range(max(1, num_runs)):
        for p_idx, name in enumerate(planner_names):
            if seed is not None:
                np.random.seed(seed + run_idx)
            hm_run = make_hmap()
            planner = build_planner(name, hm_run)
            t0 = time.time()
            path, success = planner.plan(start_pose, goal_pose)
            t_plan = time.time() - t0

            metrics = PathEvaluator.evaluate_path(path, t_plan, success)
            metrics["map_name"] = map_name
            metrics["algorithm"] = name
            metrics["schema"] = meta.get("schema", "unknown")
            metrics["use_semantic"] = bool(hmap.use_semantic)
            metrics["run_id"] = run_idx + 1

            if costmap_ref is not None and path and len(path) > 0:
                cm_vals = []
                lethal_hits = 0
                soft_hits = 0
                for p in path:
                    g = hmap.world_to_grid(p[0], p[1])
                    if g is not None:
                        ix, iy = g
                        cm_vals.append(float(costmap_ref[iy, ix]))
                        if semantic_ref is not None:
                            sid = int(semantic_ref[iy, ix])
                            if sid in (4, 5, 6, 8, 9, 12, 15, 17, 18, 27):
                                lethal_hits += 1
                            elif sid in (19, 31, 33, 34):
                                soft_hits += 1
                if cm_vals:
                    metrics["mean_costmap"] = float(np.mean(cm_vals))
                    metrics["max_costmap"] = float(np.max(cm_vals))
                    metrics["lethal_waypoint_hits"] = int(lethal_hits)
                    metrics["soft_hazard_hits"] = int(soft_hits)
                    metrics["hazard_waypoint_hits"] = int(lethal_hits + soft_hits)

            per_algo_runs[name].append(metrics)
            if run_idx == 0:
                paths_dict[name] = path

    for name in planner_names:
        runs = per_algo_runs[name]
        if len(runs) == 1:
            m = runs[0]
        else:
            succ_rate = sum(1 for r in runs if r["success"]) / float(len(runs))
            valid_runs = [r for r in runs if r["success"]] or runs
            m = {
                "success": succ_rate >= 0.5,
                "success_rate": succ_rate,
                "num_runs": len(runs),
                "planning_time_sec": float(np.mean([r["planning_time_sec"] for r in runs])),
                "path_length_3d": float(np.mean([r["path_length_3d"] for r in valid_runs])),
                "path_length_2d": float(np.mean([r["path_length_2d"] for r in valid_runs])),
                "elevation_gain": float(np.mean([r["elevation_gain"] for r in valid_runs])),
                "mean_roll_deg": float(np.mean([r["mean_roll_deg"] for r in valid_runs])),
                "std_roll_deg": float(np.mean([r["std_roll_deg"] for r in valid_runs])),
                "max_roll_deg": float(np.mean([r["max_roll_deg"] for r in valid_runs])),
                "mean_pitch_deg": float(np.mean([r["mean_pitch_deg"] for r in valid_runs])),
                "std_pitch_deg": float(np.mean([r["std_pitch_deg"] for r in valid_runs])),
                "max_pitch_deg": float(np.mean([r["max_pitch_deg"] for r in valid_runs])),
                "mean_traversability": float(np.mean([r["mean_traversability"] for r in valid_runs])),
                "total_safety_cost": float(np.mean([r["total_safety_cost"] for r in valid_runs])),
                "map_name": map_name,
                "algorithm": name,
                "schema": meta.get("schema", "unknown"),
                "use_semantic": bool(hmap.use_semantic),
                "runs": runs,
            }
            if "mean_costmap" in valid_runs[0]:
                m["mean_costmap"] = float(np.mean([r.get("mean_costmap", 0.0) for r in valid_runs]))
                m["max_costmap"] = float(np.mean([r.get("max_costmap", 0.0) for r in valid_runs]))
                m["lethal_waypoint_hits"] = float(np.mean([r.get("lethal_waypoint_hits", 0) for r in valid_runs]))
                m["soft_hazard_hits"] = float(np.mean([r.get("soft_hazard_hits", 0) for r in valid_runs]))
                m["hazard_waypoint_hits"] = float(np.mean([r.get("hazard_waypoint_hits", 0) for r in valid_runs]))
                m["std_mean_costmap"] = float(np.std([r.get("mean_costmap", 0.0) for r in valid_runs]))
            m["std_planning_time_sec"] = float(np.std([r["planning_time_sec"] for r in runs]))
            m["std_path_length_3d"] = float(np.std([r["path_length_3d"] for r in valid_runs]))
            m["std_max_roll_deg"] = float(np.std([r["max_roll_deg"] for r in valid_runs]))

        results.append(m)
        extra = f" | MeanCost: {m['mean_costmap']:.1f}" if "mean_costmap" in m else ""
        print(f"  [{name:<18}] -> Thành công: {str(m['success']):<5} | Time: {m['planning_time_sec']:.3f}s | "
              f"3D Length: {m['path_length_3d']:.2f}m | "
              f"Std Roll: {m['std_roll_deg']:.2f}° | "
              f"Mean Tau: {m['mean_traversability']:.3f}{extra}")

    # Bảng tổng hợp
    print("\n======================================================================================================")
    print(f" BẢNG TỔNG HỢP SO SÁNH TẤT CẢ THUẬT TOÁN 2.5D (Map: {map_name} | {mode_label})")
    print("======================================================================================================")
    print(f"{'Algorithm':<18} | {'Success':<8} | {'Time (s)':<10} | {'3D Length':<10} | {'Std Roll':<10} | {'Max Roll':<10} | {'Mean Tau':<10}")
    print("-" * 96)
    for r in results:
        print(f"{r['algorithm']:<18} | {str(r['success']):<8} | {r['planning_time_sec']:<10.3f} | {r['path_length_3d']:<10.2f} | {r['std_roll_deg']:<10.2f} | {r['max_roll_deg']:<10.2f} | {r['mean_traversability']:<10.3f}")

    # Lưu log JSON & Ảnh
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    file_tag = f"{map_name}_{suffix}" if suffix else map_name
    json_path = os.path.join(log_dir, f"all_planners_{file_tag}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    img_path = os.path.join(log_dir, f"all_planners_{file_tag}.png")
    fig = plt.figure(figsize=(18, 7))

    colors = {
        'T-Hybrid A*': 'red',
        '2.5D RRT*': 'blue',
        '2.5D Field D*': 'purple',
        '2.5D LazyPRM*': 'darkorange'
    }
    styles = {
        'T-Hybrid A*': '-',
        '2.5D RRT*': '--',
        '2.5D Field D*': '-',
        '2.5D LazyPRM*': '-.'
    }

    # 1. Đồ thị 3D Surface
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='terrain', alpha=0.6, edgecolor='none')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Elevation Z (m)')

    for name, path in paths_dict.items():
        if path:
            pts = np.array(path)
            ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2] + 0.1, color=colors[name], linestyle=styles[name], linewidth=2.5, label=name)

    ax1.scatter([start_pose[0]], [start_pose[1]], [hmap.get_elevation(start_pose[0], start_pose[1])], color='green', s=120, label='Start')
    ax1.scatter([goal_pose[0]], [goal_pose[1]], [hmap.get_elevation(goal_pose[0], goal_pose[1])], color='gold', marker='*', s=220, label='Goal')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'3D Path Comparison ({map_name} | {mode_label})', fontsize=12, fontweight='bold')

    # 2. Đồ thị 2D Traversability
    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(hmap.static_traversability, origin='lower', extent=[0, hmap.width, 0, hmap.height], cmap='RdYlGn', vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax2, shrink=0.75, label='Traversability tau')
    ax2.contour(X, Y, hmap.occupancy_2d, levels=[0.5], colors='black', linewidths=1.2)

    for name, path in paths_dict.items():
        if path:
            pts = np.array(path)
            ax2.plot(pts[:, 0], pts[:, 1], color=colors[name], linestyle=styles[name], linewidth=2.5, label=name)

    ax2.plot(start_pose[0], start_pose[1], 'go', markersize=10, label='Start')
    ax2.plot(goal_pose[0], goal_pose[1], 'y*', markersize=15, label='Goal')
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'2D Map & Traversability ({map_name} | {mode_label})', fontsize=12, fontweight='bold')

    # Đặt Legend ra ngoài góc phải
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, bbox_to_anchor=(1.28, 1.0), loc='upper left', borderaxespad=0., fontsize=9.5, frameon=True)

    plt.tight_layout()
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[OK] Đã lưu log JSON vào: {json_path}")
    print(f"[OK] Đã lưu ảnh so sánh tổng hợp vào: {img_path}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark all 4 2.5D Path Planners")
    parser.add_argument("--map", type=str, default=None, help="Map name in map_data/ or full path to .npz/.pt/.npy")
    parser.add_argument("--map-dir", type=str, default=None, help="Custom map_data directory")
    parser.add_argument("--log-dir", type=str, default=None, help="Custom output log directory")
    parser.add_argument("--start", type=float, nargs="+", default=None, help="Start pose: x y [yaw]")
    parser.add_argument("--goal", type=float, nargs="+", default=None, help="Goal pose: x y [yaw]")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic/costmap fusion (geometry-only ablation)")
    parser.add_argument("--runs", type=int, default=1, help="Number of benchmark runs per planner (for multi-seed averaging)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for stochastic planners")
    parser.add_argument("--suffix", type=str, default=None, help="Optional filename suffix for saved JSON and PNG outputs")
    args, _ = parser.parse_known_args()

    print("==========================================================================")
    print("      CHẠY TẤT CẢ CÁC THUẬT TOÁN 2.5D (T-Hybrid A*, RRT*, Field D*, LazyPRM*)")
    print("==========================================================================")

    start_pose = None
    goal_pose = None
    if args.start is not None and len(args.start) >= 2:
        syaw = args.start[2] if len(args.start) >= 3 else 0.0
        start_pose = (args.start[0], args.start[1], syaw)
    if args.goal is not None and len(args.goal) >= 2:
        gyaw = args.goal[2] if len(args.goal) >= 3 else 0.0
        goal_pose = (args.goal[0], args.goal[1], gyaw)

    if args.map:
        target_map = args.map
    else:
        loader = MapLoader(map_data_dir=args.map_dir)
        _, target_map = loader.select_map_interactive()

    run_benchnav_map(
        target_map,
        start_pose=start_pose,
        goal_pose=goal_pose,
        num_runs=args.runs,
        map_data_dir=args.map_dir,
        log_dir=args.log_dir,
        use_semantic=not args.no_semantic,
        seed=args.seed,
        suffix=args.suffix,
    )

if __name__ == "__main__":
    main()
