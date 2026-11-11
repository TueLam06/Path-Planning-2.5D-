import numpy as np

RELLIS_SEMANTIC_COST_TABLE = {
    0: 100,    # void / unlabeled
    1: 15,     # dirt
    3: 35,     # grass
    4: 255,    # tree
    5: 255,    # pole
    6: 255,    # water
    7: 100,    # sky
    8: 255,    # vehicle
    9: 255,    # object (lethal obstacle)
    10: 10,    # asphalt
    12: 255,   # building
    15: 255,   # log
    17: 255,   # person
    18: 255,   # fence
    19: 120,   # bush
    23: 10,    # concrete
    27: 255,   # barrier
    31: 75,    # puddle
    33: 110,   # mud
    34: 90,    # rubble
}
RELLIS_LETHAL_CLASSES = {4, 5, 6, 8, 9, 12, 15, 17, 18, 27}


class HybridMap:
    """
    Combines a 2.5D Digital Elevation Map (DEM) with surface normals, roughness,
    optional RELLIS-3D semantic/costmap layers, a 2D occupancy grid for impassable areas,
    and orientation projection methods.
    """
    # Clearpath Warthog (xe thu RELLIS-3D): 1.52 x 1.38 x 0.83 m theo thông số Clearpath
    ROBOT_LENGTH = 1.52
    ROBOT_WIDTH = 1.38
    ROLL_MAX = 0.55   # rad, giới hạn lật chung cho mọi planner và bộ kiểm chứng
    PITCH_MAX = 0.55

    def __init__(self, X, Y, Z, resolution=0.2, robot_length=ROBOT_LENGTH, robot_width=ROBOT_WIDTH,
                 rho_max=0.52, h_max=0.15, wr=0.5, w_rho=0.5,
                 semantic=None, costmap=None, observed_mask=None,
                 use_semantic=True, obstacle_cost_thresh=240, w_sem=0.5):
        """
        :param X, Y, Z: 2D meshgrid coordinate matrices
        :param resolution: Grid resolution (m)
        :param robot_length, robot_width: Dimensions of UGV (m)
        :param rho_max: Maximum allowable terrain slope (radians, default ~30 deg)
        :param h_max: Maximum obstacle height step (m)
        :param semantic: Optional 2D array of RELLIS-3D semantic IDs (ny, nx)
        :param costmap: Optional 2D array of fused semantic+geometry costmap 0..255 (ny, nx)
        :param observed_mask: Optional 2D boolean mask of directly observed cells
        :param use_semantic: Whether to fuse semantic/costmap hazards into occupancy & traversability
        :param obstacle_cost_thresh: Threshold on costmap [0..255] treated as impassable (default 240)
        :param w_sem: Weight for semantic traversability penalty when fused with geometry (default 0.5)
        """
        self.X = X
        self.Y = Y
        self.Z = Z
        self.resolution = resolution
        self.ny, self.nx = Z.shape
        self.width = self.nx * resolution
        self.height = self.ny * resolution

        self.robot_length = robot_length
        self.robot_width = robot_width
        self.roll_max = self.ROLL_MAX
        self.pitch_max = self.PITCH_MAX
        self.rho_max = rho_max
        self.h_max = h_max

        # Weighting factors for static traversability
        self.wr = wr
        self.w_rho = w_rho

        self.semantic = semantic
        self.costmap = costmap
        self.observed_mask = observed_mask
        self.use_semantic = bool(use_semantic) and (semantic is not None or costmap is not None)
        self.obstacle_cost_thresh = obstacle_cost_thresh
        self.w_sem = w_sem
        self.semantic_risk_grid = None

        # Compute map layers: Normals, Slope, Roughness, 2D Occupancy, Static Traversability
        self._compute_map_layers()

    def _compute_map_layers(self):
        """
        Calculates gradients, normal vectors, slope, roughness, and 2D occupancy grid.
        """
        # Gradients along X and Y
        dZ_dy, dZ_dx = np.gradient(self.Z, self.resolution)

        # Normal vectors v = (-dZ/dx, -dZ/dy, 1) / sqrt(...)
        norm_factor = np.sqrt(dZ_dx**2 + dZ_dy**2 + 1.0)
        self.normal_x = -dZ_dx / norm_factor
        self.normal_y = -dZ_dy / norm_factor
        self.normal_z = 1.0 / norm_factor

        # Slope angle in radians (acute angle with horizontal plane: cos(rho) = normal_z)
        self.slope = np.arccos(np.clip(self.normal_z, -1.0, 1.0))

        # Terrain Roughness: local std dev of height in 3x3 neighborhood
        from scipy.ndimage import uniform_filter
        c1 = uniform_filter(self.Z, size=3)
        c2 = uniform_filter(self.Z**2, size=3)
        self.roughness = np.sqrt(np.maximum(c2 - c1**2, 0.0))
        self.rmax = np.max(self.roughness) if np.max(self.roughness) > 1e-5 else 1.0

        # 2D Occupancy Grid: 1 = Lethal obstacle, 0 = Drivable
        # Lethal if slope > rho_max or roughness > h_max
        self.occupancy_2d = np.zeros((self.ny, self.nx), dtype=bool)
        self.occupancy_2d[self.slope > self.rho_max] = True
        self.occupancy_2d[self.roughness > self.h_max] = True

        # Geometric Static traversability tau in [0, 1]
        r_norm = np.clip(self.roughness / self.rmax, 0.0, 1.0)
        rho_norm = np.clip(self.slope / self.rho_max, 0.0, 1.0)
        geom_traversability = np.clip(1.0 - self.wr * r_norm - self.w_rho * rho_norm, 0.0, 1.0)

        if self.use_semantic:
            sem_cost_grid = np.zeros((self.ny, self.nx), dtype=np.float32)
            if self.semantic is not None:
                for cls_id, c_val in RELLIS_SEMANTIC_COST_TABLE.items():
                    sem_cost_grid[self.semantic == cls_id] = float(c_val)
                sem_lethal = np.isin(self.semantic, list(RELLIS_LETHAL_CLASSES))
                self.occupancy_2d |= sem_lethal

            if self.costmap is not None:
                cm_float = self.costmap.astype(np.float32)
                self.occupancy_2d |= (cm_float >= self.obstacle_cost_thresh)
                combined_sem = np.maximum(cm_float, sem_cost_grid) if self.semantic is not None else cm_float
            else:
                combined_sem = sem_cost_grid

            self.semantic_risk_grid = np.clip(combined_sem / 255.0, 0.0, 1.0)
            self.static_traversability = np.clip(
                (1.0 - self.w_sem) * geom_traversability + self.w_sem * (1.0 - self.semantic_risk_grid),
                0.0,
                1.0,
            )
        else:
            self.static_traversability = geom_traversability

        self.static_traversability[self.occupancy_2d] = 0.0

    def clear_around_point(self, x, y, radius=1.2):
        """Clears occupancy in a circle around (x, y) so start/goal poses are drivable."""
        g = self.world_to_grid(x, y)
        if g is not None:
            tx, ty = g
            r_cells = int(round(radius / self.resolution))
            y_min, y_max = max(0, ty - r_cells), min(self.ny, ty + r_cells + 1)
            x_min, x_max = max(0, tx - r_cells), min(self.nx, tx + r_cells + 1)
            self.occupancy_2d[y_min:y_max, x_min:x_max] = False
            if self.semantic_risk_grid is not None:
                self.semantic_risk_grid[y_min:y_max, x_min:x_max] = np.minimum(
                    self.semantic_risk_grid[y_min:y_max, x_min:x_max], 0.3
                )
            if hasattr(self, 'static_traversability'):
                self.static_traversability[y_min:y_max, x_min:x_max] = np.maximum(
                    self.static_traversability[y_min:y_max, x_min:x_max], 0.7
                )


    def world_to_grid(self, x, y):
        """Converts world (x,y) to grid indices (ix, iy). Returns None if out of bounds."""
        ix = int(round(x / self.resolution))
        iy = int(round(y / self.resolution))
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return ix, iy
        return None

    def get_elevation(self, x, y):
        """Elevation of the cell containing (x, y)."""
        g = self.world_to_grid(x, y)
        if g is not None:
            return float(self.Z[g[1], g[0]])
        return 0.0

    def is_segment_free(self, x1, y1, x2, y2, step=None):
        """True nếu đoạn thẳng không cắt ô vật cản hay ra ngoài bản đồ (lấy mẫu mỗi 1/8 ô)."""
        if step is None:
            step = 0.125 * self.resolution
        n = max(int(np.ceil(np.hypot(x2 - x1, y2 - y1) / step)), 1)
        t = np.linspace(0.0, 1.0, n + 1)
        xs = x1 + t * (x2 - x1)
        ys = y1 + t * (y2 - y1)
        ix = np.round(xs / self.resolution).astype(int)
        iy = np.round(ys / self.resolution).astype(int)
        if np.any((ix < 0) | (ix >= self.nx) | (iy < 0) | (iy >= self.ny)):
            return False
        return not bool(np.any(self.occupancy_2d[iy, ix]))

    def is_occupied_2d(self, x, y):
        """Checks if (x,y) is in the lethal 2D obstacle layer."""
        g = self.world_to_grid(x, y)
        if g is None:
            return True
        ix, iy = g
        return self.occupancy_2d[iy, ix]

    def project_pose_so3(self, x, y, yaw):
        """
        Projects SE(2) pose (x, y, yaw) onto local 2.5D terrain surface.
        Computes 3D orientation matrix and roll/pitch angles (T-Hybrid A* approach, Liu 2023).
        :return: roll (rad), pitch (rad), normal_vec (3D)
        """
        g = self.world_to_grid(x, y)
        if g is None:
            return 0.0, 0.0, np.array([0.0, 0.0, 1.0])

        ix, iy = g
        v = np.array([self.normal_x[iy, ix], self.normal_y[iy, ix], self.normal_z[iy, ix]])

        # Horizontal heading vector frame
        y_mr = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        cross_y_v = np.cross(y_mr, v)
        norm_cross = np.linalg.norm(cross_y_v)

        if norm_cross < 1e-6:
            x_m_tilde = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        else:
            x_m_tilde = cross_y_v / norm_cross

        y_m_tilde = np.cross(v, x_m_tilde)
        R = np.column_stack((x_m_tilde, y_m_tilde, v))

        # Euler angles from rotation matrix (Z-Y-X order)
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        roll = np.arctan2(R[2, 1], R[2, 2])

        return float(roll), float(pitch), v

    def project_footprint_3point(self, x, y, yaw):
        """
        Projects triangular footprint (Steinbauer & Koczka 2025) onto 2.5D heightmap.
        :return: roll (rad), pitch (rad), normal_vec (3D)
        """
        half_l = self.robot_length / 2.0
        half_w = self.robot_width / 2.0

        # Unrotated relative coordinates of triangle footprint
        # P1: rear center, P2: front left, P3: front right
        pts_local = np.array([
            [-half_l, 0.0],
            [half_l, half_w],
            [half_l, -half_w]
        ])

        # Rotation matrix for yaw
        c, s = np.cos(yaw), np.sin(yaw)
        R_yaw = np.array([[c, -s], [s, c]])

        # Transform to world 3D coordinates
        pts_3d = []
        for p in pts_local:
            pw = np.dot(R_yaw, p) + np.array([x, y])
            pz = self.get_elevation(pw[0], pw[1])
            pts_3d.append([pw[0], pw[1], pz])

        p1, p2, p3 = [np.array(p) for p in pts_3d]

        edge1 = p2 - p1
        edge2 = p3 - p1
        normal = np.cross(edge1, edge2)
        norm_val = np.linalg.norm(normal)

        if norm_val < 1e-6:
            return 0.0, 0.0, np.array([0.0, 0.0, 1.0])

        normal /= norm_val
        if normal[2] < 0:
            normal = -normal

        # Pitch and roll from normal vector
        pitch = np.arctan2(normal[0], normal[2])
        roll = -np.arctan2(normal[1], normal[2])

        return float(roll), float(pitch), normal

    def project_footprint_3point_batch(self, xs, ys, yaws):
        """Bản vector hóa của project_footprint_3point cho nhiều pose. Trả về (roll, pitch) dạng mảng."""
        xs = np.atleast_1d(np.asarray(xs, dtype=float))
        ys = np.atleast_1d(np.asarray(ys, dtype=float))
        yaws = np.atleast_1d(np.asarray(yaws, dtype=float))
        half_l = self.robot_length / 2.0
        half_w = self.robot_width / 2.0
        pts_local = np.array([[-half_l, 0.0], [half_l, half_w], [half_l, -half_w]])

        # Dùng matmul (N,1,2,2)@(1,3,2,1) để làm tròn giống hệt np.dot của bản scalar
        c, s = np.cos(yaws), np.sin(yaws)
        r_yaw = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], 1)
        pw = (r_yaw[:, None] @ pts_local[None, :, :, None])[..., 0]
        px = pw[..., 0] + xs[:, None]
        py = pw[..., 1] + ys[:, None]

        ix = np.round(px / self.resolution).astype(int)
        iy = np.round(py / self.resolution).astype(int)
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        pz = np.zeros_like(px)
        pz[inside] = self.Z[iy[inside], ix[inside]]

        pts = np.stack([px, py, pz], axis=-1)
        normal = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
        norm_val = np.linalg.norm(normal, axis=1)
        degenerate = norm_val < 1e-6
        normal = normal / np.where(degenerate, 1.0, norm_val)[:, None]
        normal[normal[:, 2] < 0] *= -1.0

        pitch = np.arctan2(normal[:, 0], normal[:, 2])
        roll = -np.arctan2(normal[:, 1], normal[:, 2])
        roll[degenerate] = 0.0
        pitch[degenerate] = 0.0
        return roll, pitch

    def poses_stable(self, xs, ys, yaws):
        """Mảng bool: pose nào có |roll| <= roll_max và |pitch| <= pitch_max (chân đế 3 điểm)."""
        roll, pitch = self.project_footprint_3point_batch(xs, ys, yaws)
        return (np.abs(roll) <= self.roll_max) & (np.abs(pitch) <= self.pitch_max)

    def compute_dynamic_traversability(self, x, y, yaw,
                                       theta_x_max=0.35, theta_y_min=-0.45, theta_y_max=0.45,
                                       wr=0.3, w_rx=0.35, w_ry=0.35):
        """
        Calculates real-time traversability tau_tilde considering robot orientation on terrain
        Eq. (11) in Liu 2023, optionally fused with RELLIS-3D semantic risk.
        """
        roll, pitch, _ = self.project_pose_so3(x, y, yaw)
        g = self.world_to_grid(x, y)
        if g is None:
            return 0.0

        ix, iy = g
        if self.occupancy_2d[iy, ix]:
            return 0.0

        rsum = self.roughness[iy, ix]

        # Normalized roughness cost
        r_term = wr * (rsum / self.rmax)
        # Normalized roll cost
        rx_term = w_rx * (abs(roll) / theta_x_max)
        # Normalized pitch cost (asymmetric uphill/downhill)
        if pitch >= 0:
            ry_term = w_ry * (pitch / theta_y_max)
        else:
            ry_term = w_ry * (pitch / theta_y_min)

        tau_geom = float(np.clip(1.0 - r_term - rx_term - ry_term, 0.0, 1.0))
        if self.use_semantic and self.semantic_risk_grid is not None:
            sem_pen = float(self.semantic_risk_grid[iy, ix])
            tau_tilde = (1.0 - self.w_sem) * tau_geom + self.w_sem * (1.0 - sem_pen)
            return float(np.clip(tau_tilde, 0.0, 1.0))
        return tau_geom
