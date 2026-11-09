import heapq
import numpy as np

class THybridAStarNode:
    def __init__(self, x, y, yaw, g=0.0, h=0.0, parent=None, steer=0.0, direction=1):
        self.x = x
        self.y = y
        self.yaw = yaw  # Radians [-pi, pi]
        self.g = g      # Cost-so-far
        self.h = h      # Heuristic cost
        self.f = g + h  # Total estimated cost
        self.parent = parent
        self.steer = steer
        self.direction = direction  # +1 for forward, -1 for reverse

    def __lt__(self, other):
        return self.f < other.f

class THybridAStar:
    """
    T-Hybrid A* Planner (Task 2.1 - Liu et al., 2023)
    Fuses 2D grid filtering with 2.5D pose projection & dynamic terrain traversability cost.
    """
    def __init__(self, hybrid_map, step_size=0.8, max_steer=0.55,
                 xy_resolution=0.6, yaw_resolution=np.radians(30.0),
                 kt=0.5, k_tau=1.5, theta_x_max=0.55, theta_y_max=0.60,
                 max_iterations=8000):
        """
        :param hybrid_map: Instance of HybridMap
        :param step_size: Distance for motion primitive step (m)
        :param max_steer: Maximum steering angle (rad)
        :param xy_resolution: Discretization grid size for closed set (m)
        :param yaw_resolution: Discretization yaw angle for closed set (rad)
        :param kt: Weight for turning cost penalty
        :param k_tau: Weight for traversability cost penalty
        :param max_iterations: Maximum search iterations
        """
        self.map = hybrid_map
        self.step_size = step_size
        self.max_steer = max_steer
        self.xy_resolution = xy_resolution
        self.yaw_resolution = yaw_resolution

        self.kt = kt
        self.k_tau = k_tau
        self.theta_x_max = theta_x_max
        self.theta_y_max = theta_y_max
        self.max_iterations = max_iterations

    def _state_key(self, x, y, yaw):
        """Discretizes continuous SE(2) state into discrete tuple for duplicate checking."""
        ix = int(round(x / self.xy_resolution))
        iy = int(round(y / self.xy_resolution))
        iyaw = int(round((yaw % (2 * np.pi)) / self.yaw_resolution))
        return (ix, iy, iyaw)

    def _precompute_2d_heuristic(self, gx, gy):
        """Precomputes 2D Dijkstra distance map from goal pose for 2.5D Hybrid A* heuristic."""
        goal_grid = self.map.world_to_grid(gx, gy)
        if goal_grid is None:
            return None

        dist_map = np.full((self.map.ny, self.map.nx), float('inf'))
        dist_map[goal_grid[1], goal_grid[0]] = 0.0

        pq = [(0.0, goal_grid)]
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]

        while pq:
            d, (cx, cy) = heapq.heappop(pq)
            if d > dist_map[cy, cx]:
                continue

            for dx, dy, step_cost in moves:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.map.nx and 0 <= ny < self.map.ny:
                    if self.map.occupancy_2d[ny, nx]:
                        continue
                    slope_penalty = 2.0 * float(self.map.slope[ny, nx])
                    new_d = d + (step_cost * self.map.resolution) + slope_penalty
                    if new_d < dist_map[ny, nx]:
                        dist_map[ny, nx] = new_d
                        heapq.heappush(pq, (new_d, (nx, ny)))
        return dist_map

    def _heuristic(self, x, y, yaw, goal_x, goal_y, goal_yaw):
        """Returns 2D-guided terrain heuristic if available, otherwise Euclidean distance."""
        if hasattr(self, 'dist_map') and self.dist_map is not None:
            g = self.map.world_to_grid(x, y)
            if g:
                val = self.dist_map[g[1], g[0]]
                if not np.isinf(val):
                    return val
        return 1.8 * np.hypot(goal_x - x, goal_y - y)



    def _get_motion_primitives(self, current_node):
        """Generates kinematically feasible child states using midpoint kinematic integration."""
        steer_inputs = np.linspace(-self.max_steer, self.max_steer, 7)
        directions = [1]  # Forward primitives

        children = []
        L = self.map.robot_length

        for d in directions:
            for steer in steer_inputs:
                ds = d * self.step_size
                d_theta = (ds / L) * np.tan(steer)
                yaw_mid = current_node.yaw + 0.5 * d_theta

                nx = current_node.x + ds * np.cos(yaw_mid)
                ny = current_node.y + ds * np.sin(yaw_mid)
                nyaw = (current_node.yaw + d_theta + np.pi) % (2 * np.pi) - np.pi

                children.append((nx, ny, nyaw, steer, d))
        return children


    def plan(self, start_pose, goal_pose, max_iterations=None, goal_tolerance=2.5):
        """
        Executes T-Hybrid A* graph search.
        :param start_pose: (x, y, yaw)
        :param goal_pose: (x, y, yaw)
        :return: path [(x, y, z, yaw, roll, pitch, tau)], success_flag
        """
        if max_iterations is None:
            max_iterations = self.max_iterations

        sx, sy, syaw = start_pose
        gx, gy, gyaw = goal_pose

        self.map.clear_around_point(sx, sy, radius=1.0)
        self.map.clear_around_point(gx, gy, radius=1.0)

        # Precompute 2D Dijkstra distance heuristic map for obstacle guidance
        self.dist_map = self._precompute_2d_heuristic(gx, gy)


        start_node = THybridAStarNode(sx, sy, syaw, g=0.0, h=self._heuristic(sx, sy, syaw, gx, gy, gyaw))

        open_set = []
        heapq.heappush(open_set, start_node)

        g_score = {}
        g_score[self._state_key(sx, sy, syaw)] = 0.0

        iterations = 0
        best_node = start_node
        min_dist_to_goal = np.hypot(gx - sx, gy - sy)

        while open_set and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(open_set)

            dist_to_goal = np.hypot(gx - current.x, gy - current.y)
            if dist_to_goal < min_dist_to_goal:
                min_dist_to_goal = dist_to_goal
                best_node = current

            # Check goal reachability condition
            if dist_to_goal <= goal_tolerance:
                return self._reconstruct_path(current), True

            # Analytical shortcut check when near goal
            if dist_to_goal < 15.0 and iterations % 3 == 0:
                direct_path = self._try_direct_connection(current, gx, gy, gyaw)
                if direct_path is not None:
                    return direct_path, True

            for nx, ny, nyaw, steer, direction in self._get_motion_primitives(current):
                # 1. Fast 2D Occupancy Check
                if self.map.is_occupied_2d(nx, ny):
                    continue

                # 2. SO(3) Pose Projection & Dynamic Traversability Check
                roll, pitch, _ = self.map.project_pose_so3(nx, ny, nyaw)
                if abs(roll) > self.theta_x_max or abs(pitch) > self.theta_y_max:
                    continue

                tau_tilde = self.map.compute_dynamic_traversability(nx, ny, nyaw, theta_x_max=self.theta_x_max)
                if tau_tilde <= 0.01:  # Impassable safety threshold
                    continue

                # 3. Node Cost Calculation: Distance + Turning Cost + Traversability Penalty
                dist_cost = self.step_size * (1.2 if direction < 0 else 1.0)
                turn_cost = self.kt * abs(steer - current.steer)
                traversability_cost = self.k_tau * (1.0 - tau_tilde)

                ng = current.g + dist_cost + turn_cost + traversability_cost
                key = self._state_key(nx, ny, nyaw)

                if key not in g_score or ng < g_score[key]:
                    g_score[key] = ng
                    nh = self._heuristic(nx, ny, nyaw, gx, gy, gyaw)
                    child_node = THybridAStarNode(nx, ny, nyaw, g=ng, h=nh, parent=current, steer=steer, direction=direction)
                    heapq.heappush(open_set, child_node)

        # Return partial path if max_iterations reached
        return self._reconstruct_path(best_node), False


    def _try_direct_connection(self, current, gx, gy, gyaw, step_check=0.4):
        """Attempts a straight analytical step connection from current node to goal pose."""
        dist = np.hypot(gx - current.x, gy - current.y)
        n_steps = int(np.ceil(dist / step_check))
        target_yaw = np.arctan2(gy - current.y, gx - current.x)

        path_nodes = []
        curr_node = current
        for i in range(1, n_steps + 1):
            t = i / n_steps
            cx = (1 - t) * current.x + t * gx
            cy = (1 - t) * current.y + t * gy
            cyaw = (1 - t) * current.yaw + t * gyaw

            if self.map.is_occupied_2d(cx, cy):
                return None

            roll, pitch, _ = self.map.project_pose_so3(cx, cy, cyaw)
            if abs(roll) > self.theta_x_max or abs(pitch) > self.theta_y_max:
                return None

            tau = self.map.compute_dynamic_traversability(cx, cy, cyaw, theta_x_max=self.theta_x_max)
            if tau <= 0.02:
                return None

            curr_node = THybridAStarNode(cx, cy, cyaw, g=curr_node.g + step_check, h=0.0, parent=curr_node)

        return self._reconstruct_path(curr_node)

    def _reconstruct_path(self, node):
        """Reconstructs trajectory path from goal node back to start node."""
        path = []
        curr = node
        while curr is not None:
            z = self.map.get_elevation(curr.x, curr.y)
            roll, pitch, _ = self.map.project_pose_so3(curr.x, curr.y, curr.yaw)
            tau = self.map.compute_dynamic_traversability(curr.x, curr.y, curr.yaw)
            path.append((curr.x, curr.y, z, curr.yaw, roll, pitch, tau))
            curr = curr.parent
        path.reverse()
        return path

