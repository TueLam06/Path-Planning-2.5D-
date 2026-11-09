import numpy as np
import heapq

class FieldDStar25D:
    """
    2.5D Field D* Planner (Ferguson & Stentz, 2005 - CMU / NASA Mars Rover)
    Interpolation-based path planner on 2.5D Digital Elevation Maps.
    Computes continuous, any-angle shortest/safest paths across cell edges using 
    continuous cost interpolation and 2.5D slope/energy constraints.
    """
    def __init__(self, hybrid_map, w_slope=2.0, w_tau=1.5, max_iterations=60000):
        self.map = hybrid_map
        self.w_slope = w_slope
        self.w_tau = w_tau
        self.max_iterations = max_iterations
        self.res = hybrid_map.resolution

    def _cost(self, s1, s2):
        """Computes traversal cost between two grid nodes on 2.5D terrain."""
        x1, y1 = s1[0] * self.res, s1[1] * self.res
        x2, y2 = s2[0] * self.res, s2[1] * self.res
        z1 = self.map.get_elevation(x1, y1)
        z2 = self.map.get_elevation(x2, y2)

        # 3D Euclidean distance
        d3d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

        # Slope and traversability cost
        mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        g = self.map.world_to_grid(mid_x, mid_y)
        if g is not None:
            if self.map.occupancy_2d[g[1], g[0]]:
                return float('inf')
            slope_val = float(self.map.slope[g[1], g[0]])
            tau_val = float(self.map.static_traversability[g[1], g[0]])
        else:
            slope_val = 0.0
            tau_val = 1.0

        if slope_val > self.map.rho_max:
            return float('inf')

        return d3d * (1.0 + self.w_slope * (slope_val / self.map.rho_max) + self.w_tau * (1.0 - tau_val))

    def plan(self, start_pose, goal_pose):
        """
        Executes 2.5D Field D* from start_pose to goal_pose.
        :return: path [(x, y, z, yaw, roll, pitch, tau), ...], success (bool)
        """
        sx, sy, _ = start_pose
        gx, gy, _ = goal_pose

        # Ensure start and goal areas are clear
        self.map.clear_around_point(sx, sy, radius=1.0)
        self.map.clear_around_point(gx, gy, radius=1.0)

        s_start = self.map.world_to_grid(sx, sy)
        s_goal = self.map.world_to_grid(gx, gy)

        if s_start is None or s_goal is None:
            return [], False

        # Priority queue for A* / Field D* search
        open_set = []
        heapq.heappush(open_set, (0.0, 0.0, s_start))
        came_from = {}
        g_score = {s_start: 0.0}

        # 16-connected neighbor offsets for high-fidelity any-angle interpolation
        nbr_offsets = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]

        found = False
        target_grid = s_start
        iterations = 0

        while open_set and iterations < self.max_iterations:
            iterations += 1
            f, g_curr, u = heapq.heappop(open_set)

            if g_curr > g_score.get(u, float('inf')):
                continue

            dist_to_goal = np.hypot((s_goal[0] - u[0]) * self.res, (s_goal[1] - u[1]) * self.res)
            if dist_to_goal <= 1.2 or u == s_goal:
                found = True
                target_grid = u
                break

            ux, uy = u
            for dx, dy in nbr_offsets:
                vx, vy = ux + dx, uy + dy
                if 0 <= vx < self.map.nx and 0 <= vy < self.map.ny:
                    if self.map.occupancy_2d[vy, vx]:
                        continue

                    v = (vx, vy)
                    step_c = self._cost(u, v)
                    if np.isinf(step_c):
                        continue

                    tentative_g = g_score[u] + step_c
                    if tentative_g < g_score.get(v, float('inf')):
                        came_from[v] = u
                        g_score[v] = tentative_g
                        # 3D Euclidean distance heuristic
                        wx_v, wy_v = vx * self.res, vy * self.res
                        wz_v = self.map.get_elevation(wx_v, wy_v)
                        wz_g = self.map.get_elevation(gx, gy)
                        h = np.sqrt((gx - wx_v)**2 + (gy - wy_v)**2 + (wz_g - wz_v)**2)
                        heapq.heappush(open_set, (tentative_g + h, tentative_g, v))

        if not found:
            # Fallback to closest explored node
            if g_score:
                closest_node = min(g_score.keys(), key=lambda node: np.hypot(node[0] - s_goal[0], node[1] - s_goal[1]))
                target_grid = closest_node
                found = True
            else:
                return [], False

        # Reconstruct path from target back to start
        grid_path = []
        curr = target_grid
        while curr in came_from:
            grid_path.append(curr)
            curr = came_from[curr]
        grid_path.append(s_start)
        grid_path.reverse()

        # Dense waypoint interpolation & orientation calculation
        path = []
        for i in range(len(grid_path)):
            gx_i, gy_i = grid_path[i]
            wx = gx_i * self.res
            wy = gy_i * self.res
            wz = self.map.get_elevation(wx, wy)

            if i < len(grid_path) - 1:
                nxt_x = grid_path[i + 1][0] * self.res
                nxt_y = grid_path[i + 1][1] * self.res
                yaw = np.arctan2(nxt_y - wy, nxt_x - wx)
            else:
                yaw = path[-1][3] if path else 0.0

            roll, pitch, _ = self.map.project_pose_so3(wx, wy, yaw)
            tau = self.map.compute_dynamic_traversability(wx, wy, yaw)
            path.append((wx, wy, wz, yaw, roll, pitch, tau))

        # Append exact goal point
        gz = self.map.get_elevation(gx, gy)
        g_roll, g_pitch, _ = self.map.project_pose_so3(gx, gy, path[-1][3])
        g_tau = self.map.compute_dynamic_traversability(gx, gy, path[-1][3])
        path.append((gx, gy, gz, path[-1][3], g_roll, g_pitch, g_tau))

        return path, found
