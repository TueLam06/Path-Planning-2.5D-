import numpy as np
import heapq
from scipy.spatial import cKDTree

class LazyPRMStar25DNode:
    def __init__(self, idx, x, y, yaw=0.0):
        self.idx = idx
        self.x = x
        self.y = y
        self.yaw = yaw
        self.neighbors = []

class LazyPRMStar25D:
    """
    2.5D LazyPRM* / ArtPlanner (Wellhausen & Hutter, 2023 - ANYbotics / ETH Zurich)
    Sampling-based roadmap planner designed for 2.5D elevation maps.
    Constructs a roadmap on the 2.5D manifold with fast lazy edge validation and footprint slope costing.
    """
    def __init__(self, hybrid_map, num_samples=2000, k_neighbors=18,
                 connection_radius=7.0, roll_max=0.55, pitch_max=0.55,
                 w_roll=1.2, w_pitch=1.0, w_slope=2.0):
        self.map = hybrid_map
        self.num_samples = num_samples
        self.k_neighbors = k_neighbors
        self.connection_radius = connection_radius
        self.roll_max = roll_max
        self.pitch_max = pitch_max
        self.w_roll = w_roll
        self.w_pitch = w_pitch
        self.w_slope = w_slope

        self.nodes = []
        self.edge_cache = {}  # edge_key -> (cost, is_valid)

    def _sample_free(self):
        """Samples state (x, y, yaw) strictly within non-lethal 2.5D map areas."""
        margin = 0.5
        for _ in range(30):
            x = np.random.uniform(margin, self.map.width - margin)
            y = np.random.uniform(margin, self.map.height - margin)
            if not self.map.is_occupied_2d(x, y):
                yaw = np.random.uniform(-np.pi, np.pi)
                return x, y, yaw
        return None

    def _eval_edge(self, n1, n2):
        """
        Fast vectorized collision & terrain slope check along edge (n1 -> n2).
        Returns (cost, is_valid).
        """
        key = (min(n1.idx, n2.idx), max(n1.idx, n2.idx))
        if key in self.edge_cache:
            return self.edge_cache[key]

        dist_2d = np.hypot(n2.x - n1.x, n2.y - n1.y)
        num_steps = max(int(np.ceil(dist_2d / (self.map.resolution * 0.7))), 3)

        pts_x = np.linspace(n1.x, n2.x, num_steps)
        pts_y = np.linspace(n1.y, n2.y, num_steps)

        # Fast vectorized grid indexing
        ixs = np.clip(np.round(pts_x / self.map.resolution).astype(int), 0, self.map.nx - 1)
        iys = np.clip(np.round(pts_y / self.map.resolution).astype(int), 0, self.map.ny - 1)

        # 1. 2D Occupancy check along entire continuous segment
        if np.any(self.map.occupancy_2d[iys, ixs]):
            self.edge_cache[key] = (float('inf'), False)
            return float('inf'), False

        # 2. Slope angle check
        slopes = self.map.slope[iys, ixs]
        if np.any(slopes > self.roll_max):
            self.edge_cache[key] = (float('inf'), False)
            return float('inf'), False

        # 3. 3D distance and terrain cost integration
        z_start = self.map.get_elevation(n1.x, n1.y)
        z_end = self.map.get_elevation(n2.x, n2.y)
        d3d = np.sqrt(dist_2d**2 + (z_end - z_start)**2)
        slope_cost = float(np.mean(slopes)) * self.w_slope * d3d
        tau_cost = float(np.mean(1.0 - self.map.static_traversability[iys, ixs])) * 1.5 * d3d

        cost = d3d + slope_cost + tau_cost
        self.edge_cache[key] = (cost, True)
        return cost, True

    def plan(self, start_pose, goal_pose):
        """
        Executes 2.5D LazyPRM* on 2.5D elevation manifold.
        """
        sx, sy, syaw = start_pose
        gx, gy, gyaw = goal_pose

        self.map.clear_around_point(sx, sy, radius=1.2)
        self.map.clear_around_point(gx, gy, radius=1.2)

        self.nodes = []
        self.edge_cache = {}

        # Node 0: Start, Node 1: Goal
        start_node = LazyPRMStar25DNode(0, sx, sy, syaw)
        goal_node = LazyPRMStar25DNode(1, gx, gy, gyaw)
        self.nodes.extend([start_node, goal_node])

        # Sample nodes across drivable 2.5D terrain
        for _ in range(self.num_samples):
            pt = self._sample_free()
            if pt is not None:
                x, y, yaw = pt
                self.nodes.append(LazyPRMStar25DNode(len(self.nodes), x, y, yaw))

        # Build initial Roadmap with k-NN query
        coords = np.array([[n.x, n.y] for n in self.nodes])
        tree = cKDTree(coords)
        dists, indices = tree.query(coords, k=min(self.k_neighbors + 1, len(self.nodes)))

        for i in range(len(self.nodes)):
            max_r = self.connection_radius * 1.8 if i < 2 else self.connection_radius
            for d, j in zip(dists[i], indices[i]):
                if j != i and j < len(self.nodes) and d <= max_r:
                    if j not in self.nodes[i].neighbors:
                        self.nodes[i].neighbors.append(j)
                    if i not in self.nodes[j].neighbors:
                        self.nodes[j].neighbors.append(i)

        # Single-Pass Lazy A* Search on Roadmap
        open_set = []
        heapq.heappush(open_set, (0.0, 0.0, 0))  # (f_score, g_score, node_idx)
        came_from = {}
        g_score = {0: 0.0}
        closed_set = set()

        path_found = False

        while open_set:
            f, g_curr, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == 1:  # Reached Goal
                path_found = True
                break

            curr_node = self.nodes[current]

            for nbr in curr_node.neighbors:
                if nbr in closed_set:
                    continue

                nbr_node = self.nodes[nbr]
                step_cost, is_valid = self._eval_edge(curr_node, nbr_node)
                if not is_valid:
                    continue

                tentative_g = g_score[current] + step_cost

                if nbr not in g_score or tentative_g < g_score[nbr]:
                    came_from[nbr] = current
                    g_score[nbr] = tentative_g
                    h = np.hypot(goal_node.x - nbr_node.x, goal_node.y - nbr_node.y)
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, nbr))

        if not path_found:
            return [], False

        # Reconstruct candidate node sequence
        candidate_indices = []
        curr = 1
        while curr in came_from:
            candidate_indices.append(curr)
            curr = came_from[curr]
        candidate_indices.append(0)
        candidate_indices.reverse()

        # Dense trajectory interpolation along the validated roadmap path
        path = []
        for k in range(len(candidate_indices) - 1):
            n1 = self.nodes[candidate_indices[k]]
            n2 = self.nodes[candidate_indices[k + 1]]
            d2 = np.hypot(n2.x - n1.x, n2.y - n1.y)
            steps = max(int(np.ceil(d2 / 0.25)), 1)
            yaw_seg = np.arctan2(n2.y - n1.y, n2.x - n1.x)

            for s in range(steps):
                alpha = s / float(steps)
                wx = (1.0 - alpha) * n1.x + alpha * n2.x
                wy = (1.0 - alpha) * n1.y + alpha * n2.y
                wz = self.map.get_elevation(wx, wy)
                roll, pitch, _ = self.map.project_footprint_3point(wx, wy, yaw_seg)
                tau = self.map.compute_dynamic_traversability(wx, wy, yaw_seg)
                path.append((wx, wy, wz, yaw_seg, roll, pitch, tau))

        gz = self.map.get_elevation(gx, gy)
        g_roll, g_pitch, _ = self.map.project_footprint_3point(gx, gy, gyaw)
        g_tau = self.map.compute_dynamic_traversability(gx, gy, gyaw)
        path.append((gx, gy, gz, gyaw, g_roll, g_pitch, g_tau))

        return path, True
