#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trình nạp dữ liệu thô RELLIS-3D (.bin point cloud + .label SemanticKITTI).

Hỗ trợ đầy đủ:
  1. Thư mục giải nén đầy đủ nhiều sequence: raw_data/Rellis-3D/<seq>/os1_cloud_node_*
  2. File nén .zip chính thức từ unmannedlab/RELLIS-3D:
       - Rellis_3D_os1_cloud_node_kitti_bin.zip
       - Rellis_3D_os1_cloud_node_semantickitti_label_id_20210614.zip
  3. Bộ dữ liệu mẫu chính thức (ouster_examples.zip / Rellis_3D_lidar_example)
     với tùy chọn tự động tải về bằng --download-sample.
"""

from pathlib import Path, PurePosixPath
import shutil
import subprocess
import threading
import urllib.request
import zipfile

import numpy as np

OFFICIAL_RELLIS_GITHUB = "https://github.com/unmannedlab/RELLIS-3D"
SAMPLE_DRIVE_ID = "1QikPnpmxneyCuwefr6m50fBOSB2ny4LC"
SAMPLE_DOWNLOAD_URL = (
    f"https://drive.usercontent.google.com/download?id={SAMPLE_DRIVE_ID}&export=download&confirm=t"
)

_local = threading.local()


def _get_thread_zip(zip_path: Path) -> zipfile.ZipFile:
    if not hasattr(_local, "zips"):
        _local.zips = {}
    key = str(zip_path)
    if key not in _local.zips:
        _local.zips[key] = zipfile.ZipFile(zip_path, "r")
    return _local.zips[key]


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_resolved = dest_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (dest_resolved / member.filename).resolve()
            if not str(member_path).startswith(str(dest_resolved)):
                raise RuntimeError(f"Đường dẫn không an toàn trong file zip: {member.filename}")
        zf.extractall(dest_resolved)


def download_sample(raw_dir=None, extract: bool = True, force: bool = False) -> Path:
    """Tải và giải nén bộ mẫu RELLIS-3D Ouster OS1 (8 frames: 000000..000007)."""
    if raw_dir is None:
        raw_dir = Path(__file__).resolve().parent / "raw_data"
    raw_dir = Path(raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "ouster_examples.zip"
    extracted_dir = raw_dir / "Rellis_3D_lidar_example"

    if not force and (extracted_dir / "os1_cloud_node_kitti_bin").is_dir():
        print(f"[Download] Bộ mẫu đã sẵn sàng tại: {extracted_dir}")
        return extracted_dir

    if force or not zip_path.is_file() or zip_path.stat().st_size < 100_000:
        print(f"[Download] Đang tải mẫu RELLIS-3D Ouster từ Google Drive về: {zip_path}")
        try:
            req = urllib.request.Request(
                SAMPLE_DOWNLOAD_URL,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp, open(zip_path, "wb") as out:
                shutil.copyfileobj(resp, out, length=1024 * 64)
        except Exception as err:
            if shutil.which("curl"):
                subprocess.run(
                    ["curl", "-fL", "-o", str(zip_path), SAMPLE_DOWNLOAD_URL],
                    check=True,
                )
            else:
                raise RuntimeError(
                    f"Không tải được dữ liệu mẫu ({err}). "
                    f"Bạn có thể tải thủ công từ {OFFICIAL_RELLIS_GITHUB}."
                ) from err

        with open(zip_path, "rb") as f:
            magic = f.read(2)
        if magic != b"PK":
            zip_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Tệp tải về không phải file .zip hợp lệ (có thể bị giới hạn Google Drive). "
                f"Vui lòng tải thủ công tại {OFFICIAL_RELLIS_GITHUB}."
            )

    if extract:
        print(f"[Download] Đang giải nén {zip_path.name} vào {raw_dir}...")
        _safe_extract_zip(zip_path, raw_dir)
        print(f"[Download] Hoàn tất! Thư mục dữ liệu mẫu: {extracted_dir}")

    return extracted_dir if extract else zip_path


class RellisDataLoader:
    """Quản lý tìm nguồn và đọc dữ liệu LiDAR .bin & nhãn .label RELLIS-3D."""

    BIN_ZIP_NAMES = ("Rellis_3D_os1_cloud_node_kitti_bin.zip",)
    LABEL_ZIP_NAMES = (
        "Rellis_3D_os1_cloud_node_semantickitti_label_id_20210614.zip",
        "Rellis_3D_os1_cloud_node_semantickitti_label_id.zip",
    )
    SAMPLE_ZIP_NAMES = ("ouster_examples.zip",)

    def __init__(self, workspace_dir=None, data_dir=None, bin_dir=None, label_dir=None):
        base_dir = Path(__file__).resolve().parent
        self.search_roots = []
        for candidate in [
            data_dir,
            workspace_dir,
            base_dir / "raw_data",
            base_dir / "data",
            base_dir,
            base_dir.parent / "raw_data",
        ]:
            if candidate is not None:
                p = Path(candidate).resolve()
                if p.exists() and p not in self.search_roots:
                    self.search_roots.append(p)

        self.explicit_bin = Path(bin_dir).resolve() if bin_dir else None
        self.explicit_label = Path(label_dir).resolve() if label_dir else None
        self.bin_source = None
        self.label_source = None
        self._find_data_sources()

    @staticmethod
    def _classify_zip_dirs(dirs_found, kind_ext):
        """Phân loại thư mục chứa *.bin hoặc *.label theo sequence_id."""
        by_seq = {}
        flat_dirs = []
        for d in dirs_found:
            if not any(d.glob(f"*{kind_ext}")):
                continue
            parent_name = d.parent.name
            if parent_name.isdigit() and len(parent_name) == 5:
                by_seq[parent_name] = d
            else:
                flat_dirs.append(d)
        return by_seq, flat_dirs

    def _find_data_sources(self):
        # --- 1. Nguồn BIN ---
        if self.explicit_bin is not None:
            if self.explicit_bin.is_file() and self.explicit_bin.suffix == ".zip":
                self.bin_source = ("zip", self.explicit_bin)
            elif self.explicit_bin.is_dir():
                if any(self.explicit_bin.glob("*.bin")):
                    self.bin_source = ("dir_map", {"_default": self.explicit_bin})
                else:
                    sub = list(self.explicit_bin.rglob("os1_cloud_node_kitti_bin"))
                    by_seq, flat = self._classify_zip_dirs(sub, ".bin")
                    if by_seq or flat:
                        mapping = dict(by_seq)
                        if flat:
                            mapping["_default"] = flat[0]
                        self.bin_source = ("dir_map", mapping)
            if self.bin_source is None:
                raise FileNotFoundError(
                    f"Không tìm thấy dữ liệu .bin tại đường dẫn --bin-dir đã chỉ định: {self.explicit_bin}"
                )

        if self.bin_source is None:
            for root in self.search_roots:
                if root.is_file() and root.suffix == ".zip":
                    self.bin_source = ("zip", root)
                    break
                for zname in self.BIN_ZIP_NAMES:
                    zp = root / zname
                    if zp.is_file():
                        self.bin_source = ("zip", zp)
                        break
                if self.bin_source:
                    break

        if self.bin_source is None:
            for root in self.search_roots:
                found = sorted(
                    d for d in root.rglob("os1_cloud_node_kitti_bin") if d.is_dir()
                )
                by_seq, flat = self._classify_zip_dirs(found, ".bin")
                if by_seq or flat:
                    mapping = dict(by_seq)
                    if flat:
                        mapping["_default"] = flat[0]
                    self.bin_source = ("dir_map", mapping)
                    break

        if self.bin_source is None:
            for root in self.search_roots:
                for zname in self.SAMPLE_ZIP_NAMES:
                    zp = root / zname
                    if zp.is_file():
                        self.bin_source = ("zip", zp)
                        break
                if self.bin_source:
                    break

        # --- 2. Nguồn LABEL ---
        if self.explicit_label is not None:
            if self.explicit_label.is_file() and self.explicit_label.suffix == ".zip":
                self.label_source = ("zip", self.explicit_label)
            elif self.explicit_label.is_dir():
                if any(self.explicit_label.glob("*.label")):
                    self.label_source = ("dir_map", {"_default": self.explicit_label})
                else:
                    sub = list(
                        self.explicit_label.rglob("os1_cloud_node_semantickitti_label_id")
                    )
                    by_seq, flat = self._classify_zip_dirs(sub, ".label")
                    if by_seq or flat:
                        mapping = dict(by_seq)
                        if flat:
                            mapping["_default"] = flat[0]
                        self.label_source = ("dir_map", mapping)
            if self.label_source is None:
                raise FileNotFoundError(
                    f"Không tìm thấy dữ liệu .label tại đường dẫn --label-dir đã chỉ định: {self.explicit_label}"
                )

        if self.label_source is None:
            for root in self.search_roots:
                if root.is_file() and root.suffix == ".zip":
                    self.label_source = ("zip", root)
                    break
                for zname in self.LABEL_ZIP_NAMES:
                    zp = root / zname
                    if zp.is_file():
                        self.label_source = ("zip", zp)
                        break
                if self.label_source:
                    break

        if self.label_source is None:
            for root in self.search_roots:
                found = sorted(
                    d
                    for d in root.rglob("os1_cloud_node_semantickitti_label_id")
                    if d.is_dir()
                )
                by_seq, flat = self._classify_zip_dirs(found, ".label")
                if by_seq or flat:
                    mapping = dict(by_seq)
                    if flat:
                        mapping["_default"] = flat[0]
                    self.label_source = ("dir_map", mapping)
                    break

        if self.label_source is None:
            for root in self.search_roots:
                for zname in self.SAMPLE_ZIP_NAMES:
                    zp = root / zname
                    if zp.is_file():
                        self.label_source = ("zip", zp)
                        break
                if self.label_source:
                    break

        if self.bin_source is None:
            raise FileNotFoundError(
                "Không tìm thấy dữ liệu thô RELLIS-3D (.bin)!\n"
                "Hãy chạy `python3 main.py --download-sample` để tải bộ mẫu (8 frames)\n"
                f"hoặc đặt bộ dữ liệu đầy đủ từ {OFFICIAL_RELLIS_GITHUB} vào thư mục `raw_data/`."
            )

        print(f"[DataLoader] Nguồn LiDAR BIN     : {self._format_source(self.bin_source)}")
        print(f"[DataLoader] Nguồn Semantics LABEL: {self._format_source(self.label_source)}")

    @staticmethod
    def _format_source(src):
        if src is None:
            return "None (chỉ dùng hình học / Void)"
        kind, payload = src
        if kind == "zip":
            return f"zip://{payload}"
        seqs = [k for k in payload.keys() if k != "_default"]
        if seqs:
            return f"dirs={sorted(seqs)} (gốc: {next(iter(payload.values())).parent.parent})"
        return str(payload.get("_default"))

    def _resolve_dir_for_seq(self, source, sequence_id: str) -> Path:
        _, mapping = source
        if sequence_id in mapping:
            return mapping[sequence_id]
        if "_default" in mapping:
            return mapping["_default"]
        first_key = sorted(mapping.keys())[0]
        return mapping[first_key]

    @staticmethod
    def _zip_matching_entries(zf: zipfile.ZipFile, sequence_id: str, ext: str):
        names = [
            n
            for n in zf.namelist()
            if n.endswith(ext) and not PurePosixPath(n).name.startswith(".")
        ]
        seq_matches = [
            n for n in names if f"/{sequence_id}/" in f"/{n}"
        ]
        return seq_matches if seq_matches else names

    def get_all_frame_ids(self, sequence_id: str = "00000"):
        """Trả về danh sách frame_id đã sắp xếp của sequence."""
        kind, payload = self.bin_source
        if kind == "zip":
            zf = _get_thread_zip(payload)
            entries = self._zip_matching_entries(zf, sequence_id, ".bin")
            frame_ids = [PurePosixPath(n).stem for n in entries]
        else:
            seq_dir = self._resolve_dir_for_seq(self.bin_source, sequence_id)
            frame_ids = [p.stem for p in seq_dir.glob("*.bin")]
        return sorted(set(frame_ids))

    def load_frame(self, sequence_id: str = "00000", frame_id: str = "000000"):
        """Đọc point cloud (N, 4) float32 và semantic labels (N,) uint16 cho 1 frame."""
        pts = None
        lbls = None

        # Đọc .bin
        kind, payload = self.bin_source
        if kind == "zip":
            zf = _get_thread_zip(payload)
            exact_candidates = [
                f"Rellis-3D/{sequence_id}/os1_cloud_node_kitti_bin/{frame_id}.bin",
                f"{sequence_id}/os1_cloud_node_kitti_bin/{frame_id}.bin",
                f"Rellis_3D_lidar_example/os1_cloud_node_kitti_bin/{frame_id}.bin",
            ]
            for path_in_zip in exact_candidates:
                try:
                    raw = zf.read(path_in_zip)
                    pts = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)
                    break
                except KeyError:
                    continue
            if pts is None:
                entries = [
                    n
                    for n in self._zip_matching_entries(zf, sequence_id, ".bin")
                    if PurePosixPath(n).stem == frame_id
                ]
                if entries:
                    pts = np.frombuffer(zf.read(entries[0]), dtype=np.float32).reshape(-1, 4)
        else:
            seq_dir = self._resolve_dir_for_seq(self.bin_source, sequence_id)
            bin_path = seq_dir / f"{frame_id}.bin"
            if bin_path.is_file():
                pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)

        if pts is None:
            raise FileNotFoundError(
                f"Không tìm thấy frame BIN '{frame_id}.bin' trong sequence '{sequence_id}'."
            )

        # Đọc .label
        if self.label_source is not None:
            lkind, lpayload = self.label_source
            if lkind == "zip":
                zf = _get_thread_zip(lpayload)
                exact_candidates = [
                    f"Rellis-3D/{sequence_id}/os1_cloud_node_semantickitti_label_id/{frame_id}.label",
                    f"{sequence_id}/os1_cloud_node_semantickitti_label_id/{frame_id}.label",
                    f"Rellis_3D_lidar_example/os1_cloud_node_semantickitti_label_id/{frame_id}.label",
                ]
                for path_in_zip in exact_candidates:
                    try:
                        raw = zf.read(path_in_zip)
                        lbls = (np.frombuffer(raw, dtype=np.uint32) & 0xFFFF).astype(np.uint16)
                        break
                    except KeyError:
                        continue
                if lbls is None:
                    entries = [
                        n
                        for n in self._zip_matching_entries(zf, sequence_id, ".label")
                        if PurePosixPath(n).stem == frame_id
                    ]
                    if entries:
                        lbls = (
                            np.frombuffer(zf.read(entries[0]), dtype=np.uint32) & 0xFFFF
                        ).astype(np.uint16)
            else:
                seq_dir = self._resolve_dir_for_seq(self.label_source, sequence_id)
                lbl_path = seq_dir / f"{frame_id}.label"
                if lbl_path.is_file():
                    lbls = (np.fromfile(str(lbl_path), dtype=np.uint32) & 0xFFFF).astype(
                        np.uint16
                    )

        if lbls is None:
            lbls = np.zeros(len(pts), dtype=np.uint16)
        elif len(lbls) != len(pts):
            raise ValueError(
                f"Số lượng điểm ({len(pts)}) không khớp số nhãn ({len(lbls)}) ở frame {frame_id}"
            )

        return pts, lbls
