import os
import time
import json
import csv
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from map_utils.terrain_generator import TerrainGenerator
from map_utils.hybrid_map import HybridMap
from planners.thybrid_a_star import THybridAStar
from planners.rrt_star_2_5d import RRTStar25D
from planners.baselines import BaselineAStar, BaselineRRTStar
from benchnav.metrics import PathEvaluator

class BenchNavRunner:
    """
    BenchNav Automated Benchmark Suite for Group 2 Path Planning Algorithms.
    Compares Baseline 2D A*, Baseline 2D RRT*, T-Hybrid A* (Task 2.1), and 2.5D RRT* (Task 2.2).
    """
    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = os.path.abspath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
            )
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.presets = ["rolling_hills", "steep_ridge", "crater_hollow", "unstructured_offroad"]

    def run_benchmark(self, num_runs_per_map=2):
        """Runs batch evaluation of all planners on 2.5D terrain presets."""
        results = []
        print("==========================================================================")
        print(" Starting BenchNav Evaluation for Group 2 Path Planning Algorithms")
        print("==========================================================================")

        for preset in self.presets:
            print(f"\n--- Benchmark Scenario: {preset} ---")
            gen = TerrainGenerator(width=50.0, height=50.0, resolution=0.25)
            X, Y, Z = gen.generate_preset(preset, seed=101)
            hmap = HybridMap(X, Y, Z, resolution=0.25)

            # Standard start and goal queries
            start_pose = (5.0, 5.0, 0.0)
            goal_pose = (43.0, 43.0, 0.0)

            planners = {
                "T-Hybrid A*": THybridAStar(hmap, step_size=0.6, max_iterations=2500),
                "2.5D RRT*": RRTStar25D(hmap, max_iterations=1200)
            }

            for name, planner in planners.items():
                for run_idx in range(num_runs_per_map):
                    t0 = time.time()
                    path, success = planner.plan(start_pose, goal_pose)
                    t_plan = time.time() - t0

                    metrics = PathEvaluator.evaluate_path(path, t_plan, success)
                    metrics["preset"] = preset
                    metrics["algorithm"] = name
                    metrics["run_id"] = run_idx + 1

                    results.append(metrics)
                    print(f"  [{name}] Run {run_idx+1}/{num_runs_per_map} -> "
                          f"Success: {success} | Time: {t_plan:.3f}s | "
                          f"3D Length: {metrics['path_length_3d']:.2f}m | "
                          f"Std Roll: {metrics['std_roll_deg']:.2f}° | "
                          f"Mean Tau: {metrics['mean_traversability']:.3f}")

        # Save JSON results
        json_path = os.path.join(self.log_dir, "benchmark_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save CSV results
        csv_path = os.path.join(self.log_dir, "benchmark_results.csv")
        if results:
            keys = list(results[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)

        print("\n==========================================================================")
        print(" BenchNav Benchmark Summary")
        print("==========================================================================")

        if HAS_PANDAS:
            df = pd.DataFrame(results)
            summary = df.groupby("algorithm").agg({
                "success": "mean",
                "planning_time_sec": "mean",
                "path_length_3d": "mean",
                "elevation_gain": "mean",
                "std_roll_deg": "mean",
                "std_pitch_deg": "mean",
                "mean_traversability": "mean",
                "total_safety_cost": "mean"
            }).reset_index()
            print(summary.to_string(index=False))
        else:
            # Manual aggregation without pandas
            algo_stats = {}
            for r in results:
                algo = r["algorithm"]
                if algo not in algo_stats:
                    algo_stats[algo] = {"count": 0, "success": 0, "time": 0.0, "len3d": 0.0, "std_roll": 0.0, "tau": 0.0}
                algo_stats[algo]["count"] += 1
                algo_stats[algo]["success"] += 1 if r["success"] else 0
                algo_stats[algo]["time"] += r["planning_time_sec"]
                algo_stats[algo]["len3d"] += r["path_length_3d"]
                algo_stats[algo]["std_roll"] += r["std_roll_deg"]
                algo_stats[algo]["tau"] += r["mean_traversability"]

            print(f"{'Algorithm':<25} | {'Success':<8} | {'Time (s)':<10} | {'3D Length':<10} | {'Std Roll':<10} | {'Mean Tau':<10}")
            print("-" * 80)
            for algo, st in algo_stats.items():
                c = st["count"]
                print(f"{algo:<25} | {st['success']/c:<8.2f} | {st['time']/c:<10.3f} | {st['len3d']/c:<10.2f} | {st['std_roll']/c:<10.2f} | {st['tau']/c:<10.3f}")

        print(f"\nDetailed logs saved to:\n - JSON: {json_path}\n - CSV: {csv_path}")
        return results

