import numpy as np

class TerrainGenerator:
    """
    Generates synthetic 2.5D heightmaps for path planning evaluation.
    Supports procedural noise, hills, steep ridges, craters, and obstacles.
    """
    def __init__(self, width=50.0, height=50.0, resolution=0.2):
        """
        :param width: Physical width in meters (X direction)
        :param height: Physical height in meters (Y direction)
        :param resolution: Grid cell resolution in meters/cell
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.nx = int(np.ceil(width / resolution))
        self.ny = int(np.ceil(height / resolution))
        self.x_coords = np.linspace(0, width, self.nx)
        self.y_coords = np.linspace(0, height, self.ny)
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)

    def generate_preset(self, preset_name="rolling_hills", seed=42):
        """
        Generates heightmap grid based on preset scenarios.
        """
        np.random.seed(seed)
        Z = np.zeros((self.ny, self.nx))

        if preset_name == "flat":
            # Completely flat surface
            pass

        elif preset_name == "rolling_hills":
            # Smooth undulating terrain
            Z += 1.5 * np.sin(self.X * 0.15) * np.cos(self.Y * 0.15)
            Z += 0.8 * np.sin(self.X * 0.35 + 1.0) * np.cos(self.Y * 0.25 + 0.5)

        elif preset_name == "steep_ridge":
            # Central steep ridge (difficult terrain with narrow passable paths)
            ridge = 3.5 * np.exp(-((self.X - 25.0)**2) / 18.0) * np.cos((self.Y - 25.0) * 0.1)
            noise = 0.5 * np.sin(self.X * 0.5) * np.sin(self.Y * 0.5)
            Z = ridge + noise

        elif preset_name == "crater_hollow":
            # Bowl/crater with surrounding steep mounds
            dist_center = np.sqrt((self.X - 25.0)**2 + (self.Y - 25.0)**2)
            Z = 2.5 * np.sin(dist_center * 0.25) / (1.0 + 0.05 * dist_center)
            # Add steep boulders/blocks
            boulder_mask = (np.abs(self.X - 15.0) < 3.0) & (np.abs(self.Y - 30.0) < 3.0)
            Z[boulder_mask] += 2.0
            boulder_mask2 = (np.abs(self.X - 35.0) < 3.0) & (np.abs(self.Y - 20.0) < 3.0)
            Z[boulder_mask2] += 2.2

        elif preset_name == "unstructured_offroad":
            # Perlin-like multi-scale noise with random rocks and steep ramps
            freqs = [0.08, 0.18, 0.35, 0.7]
            weights = [2.5, 1.2, 0.5, 0.2]
            for f, w in zip(freqs, weights):
                phase_x = np.random.uniform(0, 2 * np.pi)
                phase_y = np.random.uniform(0, 2 * np.pi)
                Z += w * np.sin(self.X * f + phase_x) * np.cos(self.Y * f + phase_y)

            # Add random boulders/obstacles
            for _ in range(8):
                bx = np.random.uniform(5, self.width - 5)
                by = np.random.uniform(5, self.height - 5)
                radius = np.random.uniform(1.2, 2.5)
                height_val = np.random.uniform(1.5, 3.0)
                dist = np.sqrt((self.X - bx)**2 + (self.Y - by)**2)
                Z += height_val * np.exp(-(dist**2) / (2 * (radius / 2.0)**2))

        else:
            raise ValueError(f"Unknown preset: {preset_name}")

        return self.X, self.Y, Z
