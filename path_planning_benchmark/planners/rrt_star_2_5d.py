import numpy as np

class RRTStar25DNode:
    def __init__(self, x, y, yaw, cost=0.0, parent=None):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.cost = cost
        self.parent = parent
        self.slope = None  # cache _slope_cost của pose node (pose không đổi)

class RRTStar25D:
    """
    Native 2.5D RRT* Planner (Task 2.2 - Steinbauer & Koczka, 2025)
    Projects robot footprint onto 2.5D elevation map to evaluate roll/pitch slope cost.
    """
    def __init__(self, hybrid_map, step_size=0.8, max_iterations=3000,
                 search_radius=3.5, goal_sample_rate=0.20,
                 roll_max=None, pitch_max=None,
                 w_roll=1.0, w_pitch=0.8, w_slope=2.0, N_norm=100.0):

        """
        :param hybrid_map: Instance of HybridMap
        :param step_size: Distance for RRT expansion step (m)
        :param max_iterations: Total sampling iterations
        :param search_radius: Neighborhood optimization radius for rewiring (m)
        :param goal_sample_rate: Goal bias ratio in [0, 1]
        :param roll_max, pitch_max: Hard safety limits (~27 degrees in rad)
        :param w_roll, w_pitch, w_slope: Cost weighting factors (Steinbauer 2025 Eq. 3.4)
        """
        self.map = hybrid_map
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.search_radius = min(search_radius, hybrid_map.width * 0.15)
        self.goal_sample_rate = goal_sample_rate


        self.roll_max = hybrid_map.roll_max if roll_max is None else roll_max
        self.pitch_max = hybrid_map.pitch_max if pitch_max is None else pitch_max
        self.w_roll = w_roll
        self.w_pitch = w_pitch
        self.w_slope = w_slope
        self.N_norm = N_norm


    def _slope_cost(self, x, y, yaw):
        """
        Steinbauer 2025 Cost Function Eq. (3.4):
        C(n) = ( (w_roll * |roll| / N) + (w_pitch * |pitch| / N) ) * w_slope
        """
        roll, pitch, _ = self.map.project_footprint_3point(x, y, yaw)
        if abs(roll) > self.roll_max or abs(pitch) > self.pitch_max:
            return float('inf')

        cost_val = ((abs(roll) / self.N_norm) * self.w_roll +
                    (abs(pitch) / self.N_norm) * self.w_pitch) * self.w_slope
        return cost_val

    def _node_slope(self, node):
        if node.slope is None:
            node.slope = self._slope_cost(node.x, node.y, node.yaw)
        return node.slope

    def _sample(self, goal_pose):
        """Samples random SE(2) state with goal biasing."""
        if np.random.rand() < self.goal_sample_rate:
            return goal_pose[0], goal_pose[1], goal_pose[2]
        else:
            rx = np.random.uniform(0, self.map.width)
            ry = np.random.uniform(0, self.map.height)
            ryaw = np.random.uniform(-np.pi, np.pi)
            return rx, ry, ryaw

    def _steer(self, from_node, rx, ry, ryaw):
        """Steers from_node towards (rx, ry, ryaw) by step_size."""
        dist = np.hypot(rx - from_node.x, ry - from_node.y)
        if dist <= self.step_size:
            nx, ny = rx, ry
            nyaw = ryaw
        else:
            theta = np.arctan2(ry - from_node.y, rx - from_node.x)
            nx = from_node.x + self.step_size * np.cos(theta)
            ny = from_node.y + self.step_size * np.sin(theta)
            nyaw = theta  # Align orientation with motion direction

        return RRTStar25DNode(nx, ny, nyaw)

    def _is_edge_valid(self, n1, n2, pose_spacing=0.4):
        """Edge is collision-free and within roll/pitch limits (pose checked every pose_spacing m)."""
        if not self.map.is_segment_free(n1.x, n1.y, n2.x, n2.y):
            return False
        length = np.hypot(n2.x - n1.x, n2.y - n1.y)
        num_checks = max(int(np.ceil(length / pose_spacing)) + 1, 2)
        dyaw = (n2.yaw - n1.yaw + np.pi) % (2 * np.pi) - np.pi
        t = np.linspace(0, 1, num_checks)
        roll, pitch = self.map.project_footprint_3point_batch(
            (1 - t) * n1.x + t * n2.x, (1 - t) * n1.y + t * n2.y, n1.yaw + t * dyaw)
        return not bool(np.any((np.abs(roll) > self.roll_max) | (np.abs(pitch) > self.pitch_max)))

    def plan(self, start_pose, goal_pose, goal_tolerance=2.5):
        """
        Executes 2.5D RRT* tree expansion and rewiring.
        :param start_pose: (x, y, yaw)
        :param goal_pose: (x, y, yaw)
        :return: path [(x, y, z, yaw, roll, pitch, tau)], success_flag
        """
        sx, sy, syaw = start_pose
        gx, gy, gyaw = goal_pose

        self.map.clear_around_point(sx, sy, radius=1.0)
        self.map.clear_around_point(gx, gy, radius=1.0)

        start_node = RRTStar25DNode(sx, sy, syaw, cost=0.0)


        nodes = [start_node]
        # Toạ độ node dạng mảng để tìm nearest/near bằng numpy
        node_xs = np.empty(self.max_iterations + 1)
        node_ys = np.empty(self.max_iterations + 1)
        node_xs[0], node_ys[0] = sx, sy

        best_goal_node = None
        min_goal_cost = float('inf')

        for i in range(self.max_iterations):
            rx, ry, ryaw = self._sample(goal_pose)
            n = len(nodes)
            nearest = nodes[int(np.argmin(np.hypot(node_xs[:n] - rx, node_ys[:n] - ry)))]
            new_node = self._steer(nearest, rx, ry, ryaw)

            if not self._is_edge_valid(nearest, new_node):
                continue

            dist = np.hypot(new_node.x - nearest.x, new_node.y - nearest.y)
            edge_slope_cost = self._node_slope(new_node)

            near_mask = np.hypot(node_xs[:n] - new_node.x, node_ys[:n] - new_node.y) <= self.search_radius
            near_nodes = [nodes[k] for k in np.flatnonzero(near_mask)]

            # Choose best parent
            min_cost = nearest.cost + dist + edge_slope_cost
            best_parent = nearest

            for near in near_nodes:
                d_near = np.hypot(new_node.x - near.x, new_node.y - near.y)
                if near.cost + d_near + edge_slope_cost < min_cost:
                    if self._is_edge_valid(near, new_node):
                        min_cost = near.cost + d_near + edge_slope_cost
                        best_parent = near

            new_node.cost = min_cost
            new_node.parent = best_parent
            nodes.append(new_node)
            node_xs[n], node_ys[n] = new_node.x, new_node.y

            # Rewire near neighbors
            for near in near_nodes:
                d_near = np.hypot(near.x - new_node.x, near.y - new_node.y)
                c_slope_near = self._node_slope(near)
                potential_cost = new_node.cost + d_near + c_slope_near
                if potential_cost < near.cost:
                    if self._is_edge_valid(new_node, near):
                        near.parent = new_node
                        near.cost = potential_cost

            # Check goal reachability
            dist_to_goal = np.hypot(gx - new_node.x, gy - new_node.y)
            if dist_to_goal <= goal_tolerance - 1e-6:
                if new_node.cost < min_goal_cost:
                    min_goal_cost = new_node.cost
                    best_goal_node = new_node

        if best_goal_node is not None:
            path = self._reconstruct_path(best_goal_node)
            # Append exact goal only if the last segment is valid
            goal_node = RRTStar25DNode(gx, gy, gyaw)
            if self._is_edge_valid(best_goal_node, goal_node):
                gz = self.map.get_elevation(gx, gy)
                roll_g, pitch_g, _ = self.map.project_footprint_3point(gx, gy, gyaw)
                tau_g = self.map.compute_dynamic_traversability(gx, gy, gyaw)
                path.append((gx, gy, gz, gyaw, roll_g, pitch_g, tau_g))
            return path, True

        # If goal not directly reached, find node closest to goal
        dists = [np.hypot(gx - n.x, gy - n.y) for n in nodes]
        closest_node = nodes[int(np.argmin(dists))]
        success = min(dists) <= goal_tolerance - 1e-6
        return self._reconstruct_path(closest_node), success







    def _reconstruct_path(self, node):
        """Reconstructs path from goal node back to root."""
        path = []
        curr = node
        while curr is not None:
            z = self.map.get_elevation(curr.x, curr.y)
            roll, pitch, _ = self.map.project_footprint_3point(curr.x, curr.y, curr.yaw)
            tau = self.map.compute_dynamic_traversability(curr.x, curr.y, curr.yaw)
            path.append((curr.x, curr.y, z, curr.yaw, roll, pitch, tau))
            curr = curr.parent
        path.reverse()
        return path
