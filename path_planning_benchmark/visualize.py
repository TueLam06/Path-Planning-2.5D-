import os
import matplotlib
matplotlib.use('Agg')  # Headless backend for server environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from map_utils.terrain_generator import TerrainGenerator
from map_utils.hybrid_map import HybridMap
from planners.thybrid_a_star import THybridAStar
from planners.rrt_star_2_5d import RRTStar25D

DEFAULT_SAVE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "comparison_plot.png")
)

def visualize_comparison(preset_name="steep_ridge", save_path=DEFAULT_SAVE_PATH):
    """
    Renders 3D Terrain Visualization and 2D Traversability Map comparing T-Hybrid A* and 2.5D RRT*.
    """
    print(f"Generating visualization for preset: {preset_name}...")
    gen = TerrainGenerator(width=50.0, height=50.0, resolution=0.25)
    X, Y, Z = gen.generate_preset(preset_name, seed=42)
    hmap = HybridMap(X, Y, Z, resolution=0.25)

    start_pose = (5.0, 5.0, 0.0)
    goal_pose = (43.0, 43.0, 0.0)

    # Run T-Hybrid A*
    planner_thybrid = THybridAStar(hmap, step_size=0.6, max_iterations=4000)
    path_thybrid, success_thybrid = planner_thybrid.plan(start_pose, goal_pose)

    # Run 2.5D RRT*
    planner_rrt = RRTStar25D(hmap, max_iterations=2000)
    path_rrt, success_rrt = planner_rrt.plan(start_pose, goal_pose)

    # Create multi-panel figure
    fig = plt.figure(figsize=(16, 7))

    # Panel 1: 3D Surface Plot with Trajectories
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='terrain', alpha=0.7, edgecolor='none')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Elevation z (m)')

    if path_thybrid:
        pts = np.array(path_thybrid)
        ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2] + 0.1, 'r-', linewidth=3, label='T-Hybrid A*')

    if path_rrt:
        pts = np.array(path_rrt)
        ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2] + 0.1, 'b--', linewidth=3, label='2.5D RRT*')

    ax1.scatter([start_pose[0]], [start_pose[1]], [hmap.get_elevation(start_pose[0], start_pose[1])],
                color='green', s=100, label='Start')
    ax1.scatter([goal_pose[0]], [goal_pose[1]], [hmap.get_elevation(goal_pose[0], goal_pose[1])],
                color='gold', marker='*', s=200, label='Goal')

    ax1.set_title(f'3D Terrain & Paths ({preset_name})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Elevation (m)')
    ax1.legend(loc='upper left')

    # Panel 2: 2D Static Traversability Map
    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(hmap.static_traversability, origin='lower',
                    extent=[0, hmap.width, 0, hmap.height], cmap='RdYlGn')
    fig.colorbar(im, ax=ax2, shrink=0.75, label='Traversability tau')


    # Overlay 2D obstacle contours
    ax2.contour(X, Y, hmap.occupancy_2d, levels=[0.5], colors='black', linewidths=1.5)

    if path_thybrid:
        pts = np.array(path_thybrid)
        ax2.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=2.5, label='T-Hybrid A*')

    if path_rrt:
        pts = np.array(path_rrt)
        ax2.plot(pts[:, 0], pts[:, 1], 'b--', linewidth=2.5, label='2.5D RRT*')

    ax2.plot(start_pose[0], start_pose[1], 'go', markersize=10, label='Start')
    ax2.plot(goal_pose[0], goal_pose[1], 'y*', markersize=15, label='Goal')

    ax2.set_title('2D Hybrid Map & Obstacle Contours', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    visualize_comparison()
