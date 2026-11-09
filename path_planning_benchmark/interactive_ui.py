import os
import sys
import argparse
import matplotlib
matplotlib.use('Agg')

from map_utils.map_loader import MapLoader
from run_thybrid import run_thybrid_standalone
from run_25d_rrt import run_25d_rrt_standalone
from run_field_d_star import run_field_d_star_standalone
from run_lazy_prm_star import run_lazy_prm_star_standalone
from run_benchmark import run_benchnav_map

def interactive_menu(map_selection=None, algo_selection=None):
    """
    Interactive Terminal UI for selecting 2.5D Path Planning Algorithms and Heightmap Datasets.
    """
    print("\n==========================================================================")
    print("      2.5D PATH PLANNING & BENCHNAV INTERACTIVE SUITE (GROUP 2)           ")
    print("==========================================================================")

    loader = MapLoader()
    map_files = loader.list_available_maps()

    if not map_files:
        print("No map files found in map_data! Generating default dataset...")
        from generate_map_dataset import generate_default_map_dataset
        generate_default_map_dataset()
        map_files = loader.list_available_maps()

    print("\n--- BƯỚC 1: Chọn tập dữ liệu bản đồ 2.5D từ map_data/ ---")
    for idx, (map_name, path) in enumerate(map_files, 1):
        print(f"  [{idx}] {map_name:<22} ({os.path.basename(path)})")

    selected_map_idx = 0
    if map_selection is not None:
        if isinstance(map_selection, int) and 1 <= map_selection <= len(map_files):
            selected_map_idx = map_selection - 1
        elif isinstance(map_selection, str):
            for i, (mname, _) in enumerate(map_files):
                if mname == map_selection:
                    selected_map_idx = i
                    break
    else:
        try:
            choice_str = input(f"\nNhập số thứ tự bản đồ [1-{len(map_files)}] (default 1): ").strip()
            if choice_str.isdigit() and 1 <= int(choice_str) <= len(map_files):
                selected_map_idx = int(choice_str) - 1
        except (EOFError, KeyboardInterrupt):
            selected_map_idx = 0

    selected_map_name = map_files[selected_map_idx][0]
    print(f">> Bản đồ đã chọn: '{selected_map_name}'")

    print("\n--- BƯỚC 2: Chọn thuật toán hoặc Chế độ chạy ---")
    print("  [1] T-Hybrid A* (Liu et al., 2023)")
    print("  [2] 2.5D RRT* (Steinbauer et al., 2025)")
    print("  [3] 2.5D Field D* (Ferguson & Stentz, 2005)")
    print("  [4] 2.5D LazyPRM* / ArtPlanner (ETH Zurich, 2023)")
    print("  [5] Chạy tất cả thuật toán (Run All Planners)")

    selected_algo_idx = 5
    if algo_selection is not None and 1 <= algo_selection <= 5:
        selected_algo_idx = algo_selection
    else:
        try:
            choice_str = input("\nChọn chế độ chạy [1-5] (default 5): ").strip()
            if choice_str.isdigit() and 1 <= int(choice_str) <= 5:
                selected_algo_idx = int(choice_str)
        except (EOFError, KeyboardInterrupt):
            selected_algo_idx = 5

    print(f"\n>> Thực thi Chế độ [{selected_algo_idx}] trên Bản đồ: '{selected_map_name}'...\n")

    if selected_algo_idx == 1:
        run_thybrid_standalone(selected_map_name)
    elif selected_algo_idx == 2:
        run_25d_rrt_standalone(selected_map_name)
    elif selected_algo_idx == 3:
        run_field_d_star_standalone(selected_map_name)
    elif selected_algo_idx == 4:
        run_lazy_prm_star_standalone(selected_map_name)
    elif selected_algo_idx == 5:
        run_benchnav_map(selected_map_name, num_runs=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive UI for selecting map and algorithm")
    parser.add_argument("--map", type=str, default=None, help="Map name or 1-based index")
    parser.add_argument("--algo", type=int, default=None, help="Algorithm choice [1-5]")
    args = parser.parse_args()

    map_val = int(args.map) if (args.map and args.map.isdigit()) else args.map
    interactive_menu(map_val, args.algo)
