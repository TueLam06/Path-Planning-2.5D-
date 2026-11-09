import heapq
import numpy as np

class BaselineAStar:
    """Standard 2D A* Planner on 2D costmap grid."""
    def __init__(self, hybrid_map):
        self.map = hybrid_map

    def plan(self, start_pose, goal_pose, max_iterations=60000):
        sx, sy, _ = start_pose
        gx, gy, _ = goal_pose


        start_grid = self.map.world_to_grid(sx, sy)
        goal_grid = self.map.world_to_grid(gx, gy)

        if start_grid is None or goal_grid is None:
            return [], False

        open_set = []
        heapq.heappush(open_set, (0.0, start_grid))

        came_from = {}
        g_score = {start_grid: 0.0}

        # 8-neighbor movements
        moves = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]

        found = False
        target_grid = start_grid

        while open_set and max_iterations > 0:
            max_iterations -= 1
            _, current = heapq.heappop(open_set)

            dist_to_goal = np.hypot((goal_grid[0] - current[0]) * self.map.resolution,
                                    (goal_grid[1] - current[1]) * self.map.resolution)
            if dist_to_goal <= 1.2 or current == goal_grid:
                found = True
                target_grid = current
                break


            cx, cy = current
            for dx, dy, cost in moves:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.map.nx and 0 <= ny < self.map.ny:
                    if self.map.occupancy_2d[ny, nx]:
                        continue

                    # Slope gradient cost from 2D costmap
                    slope_val = float(self.map.slope[ny, nx])
                    move_cost = cost + 2.0 * slope_val

                    tentative_g = g_score[current] + move_cost
                    neighbor = (nx, ny)

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        h = np.hypot((goal_grid[0] - nx) * self.map.resolution,
                                     (goal_grid[1] - ny) * self.map.resolution)
                        heapq.heappush(open_set, (tentative_g + h, neighbor))


        # Reconstruct path
        path = []
        curr = target_grid
        while curr in came_from:
            wx = curr[0] * self.map.resolution
            wy = curr[1] * self.map.resolution
            wz = self.map.get_elevation(wx, wy)
            roll, pitch, _ = self.map.project_pose_so3(wx, wy, 0.0)
            tau = self.map.compute_dynamic_traversability(wx, wy, 0.0)
            path.append((wx, wy, wz, 0.0, roll, pitch, tau))
            curr = came_from[curr]

        wx = start_grid[0] * self.map.resolution
        wy = start_grid[1] * self.map.resolution
        wz = self.map.get_elevation(wx, wy)
        roll, pitch, _ = self.map.project_pose_so3(wx, wy, 0.0)
        tau = self.map.compute_dynamic_traversability(wx, wy, 0.0)
        path.append((wx, wy, wz, 0.0, roll, pitch, tau))

        path.reverse()
        return path, found


class BaselineRRTStarNode:
    def __init__(self, x, y, cost=0.0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent

class BaselineRRTStar:
    """Standard 2D RRT* Planner on 2D occupancy costmap."""
    def __init__(self, hybrid_map, step_size=0.8, max_iterations=2000, search_radius=3.0, goal_sample_rate=0.15):
        self.map = hybrid_map
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.search_radius = search_radius
        self.goal_sample_rate = goal_sample_rate

    def plan(self, start_pose, goal_pose, goal_tolerance=1.2):
        sx, sy, _ = start_pose
        gx, gy, _ = goal_pose

        start_node = BaselineRRTStarNode(sx, sy, cost=0.0)
        nodes = [start_node]
        best_goal_node = None
        min_cost = float('inf')

        for i in range(self.max_iterations):
            if np.random.rand() < self.goal_sample_rate:
                rx, ry = gx, gy
            else:
                rx = np.random.uniform(0, self.map.width)
                ry = np.random.uniform(0, self.map.height)

            # Nearest node
            dists = [np.hypot(n.x - rx, n.y - ry) for n in nodes]
            nearest = nodes[int(np.argmin(dists))]

            # Steer
            dist = np.hypot(rx - nearest.x, ry - nearest.y)
            if dist <= self.step_size:
                nx, ny = rx, ry
            else:
                theta = np.arctan2(ry - nearest.y, rx - nearest.x)
                nx = nearest.x + self.step_size * np.cos(theta)
                ny = nearest.y + self.step_size * np.sin(theta)

            if self.map.is_occupied_2d(nx, ny):
                continue

            new_node = BaselineRRTStarNode(nx, ny, cost=nearest.cost + self.step_size, parent=nearest)
            nodes.append(new_node)

            if np.hypot(gx - nx, gy - ny) <= goal_tolerance:
                if new_node.cost < min_cost:
                    min_cost = new_node.cost
                    best_goal_node = new_node

        target = best_goal_node if best_goal_node is not None else nodes[-1]
        found = best_goal_node is not None

        path = []
        curr = target
        while curr is not None:
            wz = self.map.get_elevation(curr.x, curr.y)
            roll, pitch, _ = self.map.project_pose_so3(curr.x, curr.y, 0.0)
            tau = self.map.compute_dynamic_traversability(curr.x, curr.y, 0.0)
            path.append((curr.x, curr.y, wz, 0.0, roll, pitch, tau))
            curr = curr.parent
        path.reverse()
        return path, found
