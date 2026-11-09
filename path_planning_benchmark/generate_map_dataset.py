import os
import numpy as np
from map_utils.terrain_generator import TerrainGenerator
from map_utils.map_loader import MapLoader

def generate_default_map_dataset():
    """Generates and populates 3. code/map_data directory with 2.5D heightmaps."""
    loader = MapLoader()
    print("==========================================================")
    print(" Populating 3. code/map_data with 2.5D Heightmap Datasets")
    print("==========================================================")

    # 1. Standard Presets
    presets = ["rolling_hills", "steep_ridge", "crater_hollow", "unstructured_offroad"]
    for preset in presets:
        gen = TerrainGenerator(width=50.0, height=50.0, resolution=0.25)
        X, Y, Z = gen.generate_preset(preset, seed=42)
        loader.save_map(preset, X, Y, Z, resolution=0.25)

    # 2. RELLIS-3D Forest Map simulation (dense tree obstacles & elevation steps)
    gen_f = TerrainGenerator(width=50.0, height=50.0, resolution=0.25)
    X_f, Y_f, Z_f = gen_f.generate_preset("flat")
    # Add terrain slope
    Z_f += 0.05 * X_f + 0.08 * Y_f
    # Add dense trees/boulders
    np.random.seed(123)
    for _ in range(15):
        tx = np.random.uniform(5, 45)
        ty = np.random.uniform(5, 45)
        tr = np.random.uniform(1.0, 2.0)
        th = np.random.uniform(1.8, 3.5)
        dist = np.sqrt((X_f - tx)**2 + (Y_f - ty)**2)
        Z_f += th * np.exp(-(dist**2) / (2 * (tr / 2.0)**2))
    loader.save_map("rellis3d_forest", X_f, Y_f, Z_f, resolution=0.25)

    # 3. ETHZ Rocky Ground Map simulation (high local surface roughness)
    gen_r = TerrainGenerator(width=50.0, height=50.0, resolution=0.25)
    X_r, Y_r, Z_r = gen_r.generate_preset("rolling_hills", seed=999)
    # Add high-frequency continuous rocky ripples
    Z_r += 0.35 * np.sin(X_r * 0.8) * np.cos(Y_r * 0.8)
    Z_r += 0.20 * np.sin(X_r * 1.6 + 0.5) * np.cos(Y_r * 1.6 + 0.5)
    loader.save_map("ethz_rocky_ground", X_r, Y_r, Z_r, resolution=0.25)


    # 4. BenchNav Canyon Map simulation (narrow passage valley between steep walls)
    gen_c = TerrainGenerator(width=50.0, height=50.0, resolution=0.25)
    X_c, Y_c = gen_c.X, gen_c.Y
    # Canyon walls along Y = X line
    dist_canyon = np.abs(Y_c - X_c)
    Z_c = 4.0 * np.exp(-(dist_canyon**2) / 25.0)  # Steep walls on sides
    # Passable corridor along diagonal
    corridor_mask = dist_canyon < 3.5
    Z_c[corridor_mask] = 0.5 * np.sin(X_c[corridor_mask] * 0.2)
    loader.save_map("benchnav_canyon", X_c, Y_c, Z_c, resolution=0.25)

    print("\nDataset generation completed successfully!")
    maps = loader.list_available_maps()
    print(f"Total maps available in map_data: {len(maps)}")
    for name, path in maps:
        print(f" - {name} ({path})")

if __name__ == "__main__":
    generate_default_map_dataset()
