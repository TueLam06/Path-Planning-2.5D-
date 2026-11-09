import numpy as np

class PathEvaluator:
    """
    Evaluates generated trajectories against 2.5D off-road path planning metrics:
    - 3D Path length
    - Cumulative elevation gain
    - Roll and Pitch statistics (mean, std dev, max)
    - Dynamic traversability / safety cost score
    """
    @staticmethod
    def evaluate_path(path, planning_time, success):
        """
        :param path: List of tuples [(x, y, z, yaw, roll, pitch, tau), ...]
        :param planning_time: Time taken by planner in seconds
        :param success: Boolean flag indicating if goal was reached
        """
        if not path or len(path) < 2:
            return {
                "success": False,
                "planning_time_sec": planning_time,
                "path_length_3d": 0.0,
                "path_length_2d": 0.0,
                "elevation_gain": 0.0,
                "mean_roll_deg": 0.0,
                "std_roll_deg": 0.0,
                "max_roll_deg": 0.0,
                "mean_pitch_deg": 0.0,
                "std_pitch_deg": 0.0,
                "max_pitch_deg": 0.0,
                "mean_traversability": 0.0,
                "total_safety_cost": float('inf')
            }

        coords = np.array([[p[0], p[1], p[2]] for p in path])
        rolls = np.array([np.degrees(p[4]) for p in path])
        pitches = np.array([np.degrees(p[5]) for p in path])
        taus = np.array([p[6] for p in path])

        # 2D and 3D path lengths
        diffs_2d = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
        path_length_2d = np.sum(diffs_2d)

        diffs_3d = np.sqrt(np.diff(coords[:, 0])**2 + np.diff(coords[:, 1])**2 + np.diff(coords[:, 2])**2)
        path_length_3d = np.sum(diffs_3d)

        # Cumulative height gain
        dz = np.diff(coords[:, 2])
        elevation_gain = np.sum(np.maximum(dz, 0.0))

        # Roll / Pitch statistics
        mean_roll = float(np.mean(np.abs(rolls)))
        std_roll = float(np.std(rolls))
        max_roll = float(np.max(np.abs(rolls)))

        mean_pitch = float(np.mean(np.abs(pitches)))
        std_pitch = float(np.std(pitches))
        max_pitch = float(np.max(np.abs(pitches)))

        # Dynamic traversability / Safety cost
        mean_tau = float(np.mean(taus))
        total_safety_cost = float(np.sum(1.0 - taus))

        return {
            "success": bool(success),
            "planning_time_sec": float(planning_time),
            "path_length_3d": float(path_length_3d),
            "path_length_2d": float(path_length_2d),
            "elevation_gain": float(elevation_gain),
            "mean_roll_deg": mean_roll,
            "std_roll_deg": std_roll,
            "max_roll_deg": max_roll,
            "mean_pitch_deg": mean_pitch,
            "std_pitch_deg": std_pitch,
            "max_pitch_deg": max_pitch,
            "mean_traversability": mean_tau,
            "total_safety_cost": total_safety_cost
        }
