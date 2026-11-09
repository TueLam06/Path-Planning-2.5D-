import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for server environments
import matplotlib.pyplot as plt

from map_utils.map_loader import MapLoader
from map_utils.hybrid_map import HybridMap
from map_utils.point_picker import select_start_goal
from planners.rrt_star_2_5d import RRTStar25D
from benchnav.metrics import PathEvaluator

def run_25d_rrt_standalone(map_name, start_pose=None, goal_pose=None):
    """Executes 2.5D RRT* on the given map name."""
    loader = MapLoader()
    filepath = loader.get_map_path(map_name)
    if not filepath:
        map_name, filepath = loader.select_map_interactive()

    X, Y, Z, res, meta = loader.load_map(filepath, return_metadata=True)
    hmap = HybridMap(
        X, Y, Z, resolution=res,
        semantic=meta.get("semantic"),
        costmap=meta.get("costmap"),
        observed_mask=meta.get("observed_mask"),
    )

    if start_pose is None or goal_pose is None:
        start_pose, goal_pose = select_start_goal(hmap, map_name)

    print(f"Executing 2.5D RRT* search on '{map_name}' from {start_pose[:2]} to {goal_pose[:2]}...")

    planner = RRTStar25D(hmap, max_iterations=1200)

    t0 = time.time()
    path, success = planner.plan(start_pose, goal_pose)
    t_plan = time.time() - t0

    metrics = PathEvaluator.evaluate_path(path, t_plan, success)
    metrics["map_name"] = map_name
    metrics["algorithm"] = "2.5D RRT*"

    print("\n==========================================================================")
    print(f" THÀNH CÔNG: KẾT QUẢ THỰC NGHIỆM 2.5D RRT* (Map: {map_name})")
    print("==========================================================================")
    print(f" - Trạng thái thành công: {metrics['success']}")
    print(f" - Thời gian tính toán:   {metrics['planning_time_sec']:.3f} s")
    print(f" - Độ dài đường đi 3D:    {metrics['path_length_3d']:.2f} m")
    print(f" - Độ lệch chuẩn Roll:    {metrics['std_roll_deg']:.2f}° (Max: {metrics['max_roll_deg']:.2f}°)")
    print(f" - Độ lệch chuẩn Pitch:   {metrics['std_pitch_deg']:.2f}° (Max: {metrics['max_pitch_deg']:.2f}°)")
    print(f" - Traversability (tau):  {metrics['mean_traversability']:.3f}")

    log_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs"))
    os.makedirs(log_dir, exist_ok=True)

    json_path = os.path.join(log_dir, f"25d_rrt_{map_name}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    img_path = os.path.join(log_dir, f"25d_rrt_{map_name}.png")
    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='terrain', alpha=0.7, edgecolor='none')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Độ cao Z (m)')
    if path:
        pts = np.array(path)
        ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2] + 0.1, 'b--', linewidth=3, label='2.5D RRT* Path')
    ax1.scatter([start_pose[0]], [start_pose[1]], [hmap.get_elevation(start_pose[0], start_pose[1])], color='green', s=120, label='Start')
    ax1.scatter([goal_pose[0]], [goal_pose[1]], [hmap.get_elevation(goal_pose[0], goal_pose[1])], color='gold', marker='*', s=220, label='Goal')
    ax1.set_title(f'2.5D RRT* 3D Trajectory ({map_name})', fontsize=12, fontweight='bold')

    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(hmap.static_traversability, origin='lower', extent=[0, hmap.width, 0, hmap.height], cmap='RdYlGn')
    fig.colorbar(im, ax=ax2, shrink=0.75, label='Traversability tau')
    ax2.contour(X, Y, hmap.occupancy_2d, levels=[0.5], colors='black', linewidths=1.5)
    if path:
        pts = np.array(path)
        ax2.plot(pts[:, 0], pts[:, 1], 'b--', linewidth=2.5, label='2.5D RRT*')
    ax2.plot(start_pose[0], start_pose[1], 'go', markersize=10, label='Start')
    ax2.plot(goal_pose[0], goal_pose[1], 'y*', markersize=15, label='Goal')
    ax2.set_title(f'2D Map & Traversability ({map_name})', fontsize=12, fontweight='bold')

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, bbox_to_anchor=(1.28, 1.0), loc='upper left', borderaxespad=0., fontsize=9.5, frameon=True)

    plt.tight_layout()
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[OK] Đã lưu log JSON vào: {json_path}")
    print(f"[OK] Đã lưu ảnh đường đi vào: {img_path}")
    return metrics

def main():
    print("==========================================================================")
    print("      TASK 2.2: 2.5D RRT* PLANNER (Steinbauer et al., 2025) - STANDALONE  ")
    print("==========================================================================")
    loader = MapLoader()
    map_name, filepath = loader.select_map_interactive()
    run_25d_rrt_standalone(map_name)

if __name__ == "__main__":
    main()
