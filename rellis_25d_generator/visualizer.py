#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trực quan hóa Dashboard 6 góc nhìn và xuất ảnh GIF cho bộ dữ liệu 2.5D RELLIS-3D.
"""

import glob
from pathlib import Path
import threading

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from config import RELLIS_COLORS, RELLIS_NAMES, RellisConfig

_plot_lock = threading.Lock()


def create_semantic_rgb_image(semantic_grid):
    """Chuyển ma trận semantic class ID sang ảnh màu RGB chuẩn RELLIS-3D."""
    H, W = semantic_grid.shape
    rgb_img = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, color in RELLIS_COLORS.items():
        rgb_img[semantic_grid == cls_id] = color
    return rgb_img


def visualize_dashboard(
    frame_id: str,
    pts_bev: dict,
    elevation,
    slope_rad,
    roughness,
    semantic_grid,
    costmap,
    path_semantic,
    path_geom,
    start_rc,
    goal_rc,
    out_fig_path,
    cfg: RellisConfig,
    mode: str = "jikay",
):
    """Tạo Dashboard tổng hợp 6 đồ thị trực quan hóa (thread-safe)."""
    with _plot_lock:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
        plt.subplots_adjust(wspace=0.25, hspace=0.30)
        fig.suptitle(
            f"RELLIS-3D to 2.5D Semantic Elevation Costmap & Path Planning - Frame [{frame_id}] ({mode})",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        if mode == "group1":
            extent = [cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX]
            origin = "lower"
            px, py = pts_bev["x"], pts_bev["y"]
            xlabel, ylabel = "X (Sau -> Trước, m)", "Y (Phải -> Trái, m)"
            xlim = (cfg.X_MIN, cfg.X_MAX)
            ylim = (cfg.Y_MIN, cfg.Y_MAX)

            def rc_to_plot_xy(rc_list):
                if not rc_list:
                    return [], []
                r = np.array([p[0] for p in rc_list])
                c = np.array([p[1] for p in rc_list])
                xw = cfg.X_MIN + (c + 0.5) * cfg.RES
                yw = cfg.Y_MIN + (r + 0.5) * cfg.RES
                return xw, yw

        else:
            extent = [cfg.Y_MAX, cfg.Y_MIN, cfg.X_MIN, cfg.X_MAX]
            origin = "upper"
            px, py = pts_bev["y"], pts_bev["x"]
            xlabel, ylabel = "Y (Trái -> Phải, m)", "X (Sau -> Trước, m)"
            xlim = (cfg.Y_MAX, cfg.Y_MIN)
            ylim = (cfg.X_MIN, cfg.X_MAX)

            def rc_to_plot_xy(rc_list):
                if not rc_list:
                    return [], []
                r = np.array([p[0] for p in rc_list])
                c = np.array([p[1] for p in rc_list])
                xw = cfg.X_MAX - (r + 0.5) * cfg.RES
                yw = cfg.Y_MAX - (c + 0.5) * cfg.RES
                return yw, xw

        # 1. Point Cloud BEV
        ax = axes[0, 0]
        sc = ax.scatter(px, py, c=pts_bev["z"], s=0.3, cmap="terrain", alpha=0.7)
        ax.set_title("1. Raw LiDAR Point Cloud (BEV)", fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.colorbar(sc, ax=ax, label="Cao độ Z (m)", fraction=0.046, pad=0.04)

        # 2. Elevation Grid
        ax = axes[0, 1]
        im = ax.imshow(elevation, extent=extent, cmap="gist_earth", origin=origin)
        ax.set_title(f"2. Elevation Grid (Res: {cfg.RES}m)", fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax, label="Mặt đất z (m)", fraction=0.046, pad=0.04)

        # 3. Slope Map
        ax = axes[0, 2]
        im = ax.imshow(np.degrees(slope_rad), extent=extent, cmap="hot", vmin=0, vmax=35, origin=origin)
        ax.set_title("3. Terrain Slope Map", fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax, label="Độ dốc (Độ)", fraction=0.046, pad=0.04)

        # 4. Roughness Map
        ax = axes[1, 0]
        im = ax.imshow(roughness, extent=extent, cmap="plasma", vmin=0, vmax=0.3, origin=origin)
        ax.set_title("4. Surface Roughness (Std Z)", fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax, label="Độ nhấp nhô (m)", fraction=0.046, pad=0.04)

        # 5. Semantic Grid
        ax = axes[1, 1]
        sem_rgb = create_semantic_rgb_image(semantic_grid)
        ax.imshow(sem_rgb, extent=extent, origin=origin)
        ax.set_title("5. 2.5D Semantic Grid", fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        present_classes = np.unique(semantic_grid)
        legend_patches = [
            mpatches.Patch(
                color=np.array(RELLIS_COLORS.get(int(c), (0, 0, 0))) / 255.0,
                label=RELLIS_NAMES.get(int(c), f"ID {c}"),
            )
            for c in present_classes
            if int(c) in RELLIS_NAMES
        ]
        if legend_patches:
            ax.legend(handles=legend_patches[:8], loc="lower right", fontsize=8, framealpha=0.75)

        # 6. Costmap & A* Path
        ax = axes[1, 2]
        im = ax.imshow(costmap, extent=extent, cmap="YlOrRd", vmin=0, vmax=255, origin=origin)
        ax.set_title("6. Costmap & A* Path Planning", fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax, label="Chi phí vượt địa hình (0-255)", fraction=0.046, pad=0.04)

        if path_geom:
            gx, gy = rc_to_plot_xy(path_geom)
            ax.plot(gx, gy, color="blue", linestyle="--", linewidth=2.0, label="Geo-Only A*")
        if path_semantic:
            sx, sy = rc_to_plot_xy(path_semantic)
            ax.plot(sx, sy, color="lime", linewidth=3.0, label="Semantic+Geo A*")

        sx0, sy0 = rc_to_plot_xy([start_rc])
        gx0, gy0 = rc_to_plot_xy([goal_rc])
        ax.scatter(sx0, sy0, color="cyan", s=130, edgecolors="black", zorder=5, label="Start")
        ax.scatter(gx0, gy0, color="magenta", s=160, marker="*", edgecolors="black", zorder=5, label="Goal")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)

        plt.tight_layout()
        plt.savefig(out_fig_path)
        plt.close(fig)
    print(f"[Visualizer] Đã lưu Dashboard tại: {out_fig_path}")


def create_gif(png_dir="output_25d", output_gif="demo_rellis_25d.gif", max_width: int = 1200, fps: int = 4):
    """Ghép chuỗi ảnh *_dashboard.png thành ảnh động GIF."""
    from PIL import Image

    png_files = sorted(glob.glob(str(Path(png_dir) / "*_dashboard.png")))
    if not png_files:
        print(f"[GIF] Không tìm thấy ảnh *_dashboard.png trong {png_dir}")
        return None

    print(f"[GIF] Đang ghép {len(png_files)} ảnh thành {output_gif}...")
    frames = []
    for p in png_files:
        with Image.open(p) as img:
            w, h = img.size
            if w > max_width:
                new_h = int(h * (max_width / w))
                img_resized = img.resize((max_width, new_h), Image.Resampling.LANCZOS)
            else:
                img_resized = img.copy()
            frames.append(img_resized.convert("P", palette=Image.Palette.ADAPTIVE))

    duration_ms = max(20, int(1000 / max(1, fps)))
    out_path = Path(output_gif)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[GIF] Hoàn tất: {out_path} ({size_mb:.2f} MB)")
    return out_path

