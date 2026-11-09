import os
import glob
import pickle
import sys
import zipfile
import numpy as np

# scripts/generate_terrain_dataset.py của BenchNav: grid_size=64, resolution=0.5 (m)
BENCHNAV_RESOLUTION = 0.5

_TORCH_STORAGE_DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
    "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
}


class _TorchStub:
    """Thay cho object torch không cần dùng (distributions, model...)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


def _load_torch_file_numpy(path):
    """Đọc file torch.save (định dạng zip) thành dict/ndarray numpy."""
    with zipfile.ZipFile(path) as zf:
        pkl_name = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        prefix = pkl_name[: -len("data.pkl")]

        def rebuild_tensor(storage, offset, size, stride, *args):
            itemsize = storage.dtype.itemsize
            return np.lib.stride_tricks.as_strided(
                storage[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)
            ).copy()

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name in ("_rebuild_tensor", "_rebuild_tensor_v2"):
                    return rebuild_tensor
                if name in _TORCH_STORAGE_DTYPES:
                    return name
                if module.startswith("torch"):
                    return _TorchStub
                return super().find_class(module, name)

            def persistent_load(self, pid):
                _, storage_type, key, _location, _numel = pid
                dtype = _TORCH_STORAGE_DTYPES.get(storage_type, np.float32)
                return np.frombuffer(zf.read(f"{prefix}data/{key}"), dtype=dtype)

        return _Unpickler(zf.open(pkl_name)).load()


class MapLoader:
    """
    Utility for scanning, listing, and loading raw BenchNav map datasets (.npz / .pt / .npy) from map_data folder.
    Directly parses raw PyTorch map checkpoints and NumPy compressed maps from masafumiendo/benchnav
    as well as RELLIS-3D 2.5D .npz maps.
    """
    def __init__(self, map_data_dir=None):
        if map_data_dir is None:
            map_data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "map_data"
            )
        self.map_data_dir = os.path.abspath(map_data_dir)
        os.makedirs(self.map_data_dir, exist_ok=True)

    def list_available_maps(self):
        """Scans map_data directory and returns list of raw map filenames (.npz, .pt, and .npy)."""
        pattern_npz = os.path.join(self.map_data_dir, "*.npz")
        pattern_pt = os.path.join(self.map_data_dir, "*.pt")
        pattern_npy = os.path.join(self.map_data_dir, "*.npy")

        map_files = sorted(glob.glob(pattern_npz)) + sorted(glob.glob(pattern_pt)) + sorted(glob.glob(pattern_npy))
        map_list = []
        for f in map_files:
            basename = os.path.basename(f)
            map_name = os.path.splitext(basename)[0]
            map_list.append((map_name, f))
        return map_list

    def get_map_path(self, map_name):
        """Returns the full filepath for a map name/path or None if not found."""
        if map_name and os.path.isfile(map_name):
            return os.path.abspath(map_name)
        for mname, path in self.list_available_maps():
            if mname == map_name:
                return path
        return None

    def select_map_interactive(self):
        """
        Scans map_data folder, prints numbered list of available raw maps,
        and prompts user to choose a map.
        :return: map_name, map_filepath
        """
        map_files = self.list_available_maps()
        if not map_files:
            raise FileNotFoundError(f"No raw map files (.npz, .pt, or .npy) found in {self.map_data_dir}!")

        print("\n==========================================================")
        print("    SELECT A RAW BENCHNAV MAP DATASET FROM map_data/       ")
        print("==========================================================")
        for idx, (map_name, path) in enumerate(map_files, 1):
            ext = os.path.splitext(path)[1]
            print(f"  [{idx}] {map_name:<25} ({os.path.basename(path)})")

        selected_idx = 0
        caller = os.path.basename(sys.argv[0]) if sys.argv else ""
        if caller != "main.py" and len(sys.argv) > 1 and sys.argv[1].isdigit():
            choice = int(sys.argv[1])
            if 1 <= choice <= len(map_files):
                selected_idx = choice - 1
        else:
            try:
                prompt_str = f"\nEnter map number [1-{len(map_files)}] (default 1): "
                user_input = input(prompt_str).strip()
                if user_input.isdigit() and 1 <= int(user_input) <= len(map_files):
                    selected_idx = int(user_input) - 1
            except (EOFError, KeyboardInterrupt):
                selected_idx = 0

        map_name, filepath = map_files[selected_idx]
        print(f">> Selected Map [{selected_idx+1}]: '{map_name}' ({os.path.basename(filepath)})\n")
        return map_name, filepath

    @staticmethod
    def _is_jikay_bev_layout(data, Z, resolution):
        """
        Detects whether an .npz archive uses Jikay71 BEV layout:
          shape = (H_x, W_y) with row: x_max -> x_min and col: y_max -> y_min.
        Canonical HybridMap layout uses shape = (ny, nx) with row: y_min -> y_max, col: x_min -> x_max.
        """
        if not all(k in data for k in ("x_min", "x_max", "y_min", "y_max")):
            return False
        nx_span = int(round((float(data["x_max"]) - float(data["x_min"])) / resolution))
        ny_span = int(round((float(data["y_max"]) - float(data["y_min"])) / resolution))
        if Z.shape != (nx_span, ny_span):
            return False
        return (
            nx_span != ny_span
            or "observed_mask" in data
            or "step_height" in data
            or "costmap_geom" in data
        )

    @staticmethod
    def _to_canonical_grid(arr):
        """Converts Jikay71 (H_x, W_y) grid into canonical (ny, nx) grid."""
        return arr[::-1, ::-1].T

    def load_map(self, filepath_or_name, return_metadata=False):
        """
        Loads raw 2.5D heightmap directly from BenchNav .npz, .pt, or .npy file.
        :return: (X, Y, Z, resolution) by default, or (X, Y, Z, resolution, metadata)
                 when return_metadata=True.
        """
        if not (filepath_or_name.endswith(".npz") or filepath_or_name.endswith(".pt") or filepath_or_name.endswith(".npy")):
            filepath_npz = os.path.join(self.map_data_dir, f"{filepath_or_name}.npz")
            filepath_pt = os.path.join(self.map_data_dir, f"{filepath_or_name}.pt")
            filepath_npy = os.path.join(self.map_data_dir, f"{filepath_or_name}.npy")
            if os.path.exists(filepath_npz):
                filepath = filepath_npz
            elif os.path.exists(filepath_pt):
                filepath = filepath_pt
            elif os.path.exists(filepath_npy):
                filepath = filepath_npy
            else:
                raise FileNotFoundError(f"Map file not found: {filepath_or_name}")
        else:
            filepath = filepath_or_name

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Map file not found: {filepath}")

        metadata = {"schema": "unknown", "transposed_from_jikay": False}

        # Parse .npz file
        if filepath.endswith(".npz"):
            data = np.load(filepath, allow_pickle=True)
            if "elevation" in data:
                Z = np.array(data["elevation"])
            elif "Z" in data:
                Z = np.array(data["Z"])
            elif "heights" in data:
                Z = np.array(data["heights"])
            else:
                raise ValueError(f"Unrecognized keys in npz map file: {list(data.keys())}")

            resolution = float(data["resolution"]) if "resolution" in data else 0.25
            is_jikay = self._is_jikay_bev_layout(data, Z, resolution)
            metadata["transposed_from_jikay"] = is_jikay
            metadata["schema"] = "jikay" if is_jikay else ("group1" if "valid_mask" in data else "npz")

            def _orient(arr):
                return self._to_canonical_grid(arr) if is_jikay else arr

            Z = _orient(Z)
            for layer_key in ("semantic", "costmap", "costmap_geom", "step_height"):
                if layer_key in data:
                    metadata[layer_key] = _orient(np.array(data[layer_key]))
            if "observed_mask" in data:
                metadata["observed_mask"] = _orient(np.array(data["observed_mask"]).astype(bool))
            elif "valid_mask" in data:
                metadata["observed_mask"] = _orient(np.array(data["valid_mask"]).astype(bool))

            for bound_key in ("x_min", "x_max", "y_min", "y_max"):
                if bound_key in data:
                    metadata[bound_key] = float(data[bound_key])

            ny, nx = Z.shape
            x = np.linspace(0, nx * resolution, nx)
            y = np.linspace(0, ny * resolution, ny)
            X, Y = np.meshgrid(x, y)
            return (X, Y, Z, resolution, metadata) if return_metadata else (X, Y, Z, resolution)

        # Parse raw PyTorch .pt file (BenchNav, đọc bằng numpy, không cần torch)
        if filepath.endswith(".pt"):
            data = _load_torch_file_numpy(filepath)
            tensors = data.get("tensors", {}) if isinstance(data, dict) else {}
            if "heights" in tensors:
                Z = np.array(tensors["heights"], dtype=np.float32)
            elif isinstance(data, np.ndarray):
                Z = np.array(data, dtype=np.float32)
            else:
                raise ValueError(f"Unrecognized PyTorch format in map file: {filepath}")

            metadata["schema"] = "benchnav"
            resolution = BENCHNAV_RESOLUTION
            ny, nx = Z.shape
            x = np.linspace(0, nx * resolution, nx)
            y = np.linspace(0, ny * resolution, ny)
            X, Y = np.meshgrid(x, y)
            return (X, Y, Z, resolution, metadata) if return_metadata else (X, Y, Z, resolution)

        # Parse .npy file
        data = np.load(filepath, allow_pickle=True).item()
        res = data.get("resolution", 0.25)
        metadata["schema"] = "npy"
        if return_metadata:
            return data["X"], data["Y"], data["Z"], res, metadata
        return data["X"], data["Y"], data["Z"], res

    def save_map(self, map_name, X, Y, Z, resolution=0.25):
        """Saves a 2.5D heightmap dataset in compressed .npz format under map_data_dir."""
        os.makedirs(self.map_data_dir, exist_ok=True)
        out_path = os.path.join(self.map_data_dir, f"{map_name}.npz")
        np.savez_compressed(
            out_path,
            elevation=np.asarray(Z, dtype=np.float32),
            X=np.asarray(X, dtype=np.float32),
            Y=np.asarray(Y, dtype=np.float32),
            resolution=float(resolution),
        )
        return out_path
