import os, time, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from map_utils.map_loader import MapLoader
from map_utils.hybrid_map import HybridMap
from planners.thybrid_a_star import THybridAStar
from planners.rrt_star_2_5d import RRTStar25D
from planners.baselines import BaselineAStar, BaselineRRTStar
from benchnav.metrics import PathEvaluator

loader = MapLoader()
filepath = os.path.join(loader.map_data_dir, 'from_benchnav', '000_001.pt')
map_name = '000_001'

X, Y, Z, res = loader.load_map(filepath)
hmap = HybridMap(X, Y, Z, resolution=res)

start_pose = (2.0, 2.0, 0.0)
goal_pose = (18.0, 18.0, 0.0)

planners = {
    'T-Hybrid A*': THybridAStar(hmap, max_iterations=4000),
    '2.5D RRT*': RRTStar25D(hmap, max_iterations=1200)
}

results = []
paths_dict = {}

for name, planner in planners.items():
    t0 = time.time()
    path, success = planner.plan(start_pose, goal_pose)
    t_plan = time.time() - t0

    metrics = PathEvaluator.evaluate_path(path, t_plan, success)
    metrics['map_name'] = map_name
    metrics['algorithm'] = name
    results.append(metrics)
    paths_dict[name] = path
    print(f"[{name}] Success: {success} | Time: {t_plan:.3f}s | Length 3D: {metrics['path_length_3d']:.2f}m")

log_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs'))
os.makedirs(log_dir, exist_ok=True)
json_path = os.path.join(log_dir, f'benchnav_{map_name}.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

img_path = os.path.join(log_dir, f'benchnav_{map_name}.png')
fig = plt.figure(figsize=(16, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='terrain', alpha=0.6, edgecolor='none')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Z (m)')

colors = {'T-Hybrid A*': 'red', '2.5D RRT*': 'blue'}
styles = {'T-Hybrid A*': '-', '2.5D RRT*': '--'}

for name, path in paths_dict.items():
    if path:
        pts = np.array(path)
        ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2] + 0.1, color=colors[name], linestyle=styles[name], linewidth=2.5, label=name)

ax1.scatter([start_pose[0]], [start_pose[1]], [hmap.get_elevation(start_pose[0], start_pose[1])], color='green', s=120, label='Start')
ax1.scatter([goal_pose[0]], [goal_pose[1]], [hmap.get_elevation(goal_pose[0], goal_pose[1])], color='gold', marker='*', s=220, label='Goal')
ax1.set_title(f'BenchNav 3D Path Comparison ({map_name})', fontsize=12, fontweight='bold')

ax2 = fig.add_subplot(1, 2, 2)
im = ax2.imshow(hmap.static_traversability, origin='lower', extent=[0, hmap.width, 0, hmap.height], cmap='RdYlGn')
fig.colorbar(im, ax=ax2, shrink=0.75, label='Traversability tau')
ax2.contour(X, Y, hmap.occupancy_2d, levels=[0.5], colors='black', linewidths=1.5)

for name, path in paths_dict.items():
    if path:
        pts = np.array(path)
        ax2.plot(pts[:, 0], pts[:, 1], color=colors[name], linestyle=styles[name], linewidth=2.5, label=name)

ax2.plot(start_pose[0], start_pose[1], 'go', markersize=10, label='Start')
ax2.plot(goal_pose[0], goal_pose[1], 'y*', markersize=15, label='Goal')
ax2.set_title(f'2D Map & Traversability Comparison ({map_name})', fontsize=12, fontweight='bold')

handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, bbox_to_anchor=(1.28, 1.0), loc='upper left', borderaxespad=0., fontsize=9.5, frameon=True)

plt.tight_layout()
plt.savefig(img_path, dpi=300, bbox_inches='tight')
plt.close()

print('Saved JSON:', json_path)
print('Saved PNG:', img_path)
