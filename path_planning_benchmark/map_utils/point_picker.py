import os
import sys
import numpy as np

def parse_coordinate_input(input_str, default_val, point_name="Point", max_w=100.0, max_h=100.0):
    """
    Parses user input string into (x, y) float tuple.
    Accepts: '10, 15', '10 15', '10.5,20.2', or empty string (falls back to default).
    """
    text = input_str.strip()
    if not text:
        print(f"  -> Dùng mặc định cho {point_name}: ({default_val[0]:.2f}, {default_val[1]:.2f})")
        return default_val[0], default_val[1]

    # Replace comma with space, then split
    parts = text.replace(',', ' ').split()
    if len(parts) >= 2:
        try:
            x = float(parts[0])
            y = float(parts[1])
            # Check bounds
            x = max(0.5, min(max_w - 0.5, x))
            y = max(0.5, min(max_h - 0.5, y))
            print(f"  [OK] Đã nhận {point_name}: ({x:.2f}, {y:.2f})")
            return x, y
        except ValueError:
            pass

    print(f"  [Cảnh báo] Nhập không đúng định dạng '{input_str}'. Dùng mặc định cho {point_name}: ({default_val[0]:.2f}, {default_val[1]:.2f})")
    return default_val[0], default_val[1]

def select_start_goal(hmap, map_name="map"):
    """
    Prompts user to select Start and Goal coordinates with interactive CLI or GUI window.
    Prints explicit confirmation and handles custom inputs with retry / graceful fallbacks.
    """
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    if has_display:
        try:
            import matplotlib
            if matplotlib.get_backend().lower() not in ['agg', 'template']:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 8))
                im = ax.imshow(hmap.static_traversability, origin='lower',
                               extent=[0, hmap.width, 0, hmap.height], cmap='RdYlGn')
                fig.colorbar(im, ax=ax, label='Traversability tau')
                ax.contour(hmap.X, hmap.Y, hmap.occupancy_2d, levels=[0.5], colors='black', linewidths=1.2)

                ax.set_title(f"[{map_name}] CLICK 1: Select START (Green) | CLICK 2: Select GOAL (Gold)",
                             fontsize=11, fontweight='bold', color='darkblue')
                ax.set_xlabel("X (meters)")
                ax.set_ylabel("Y (meters)")
                ax.grid(True, linestyle='--', alpha=0.5)

                print("\n>> CỬA SỔ CHỌN ĐIỂM: Click 2 điểm trên Bản đồ:")
                print("   - Click 1: Điểm START")
                print("   - Click 2: Điểm GOAL")

                pts = plt.ginput(2, timeout=30)
                plt.close(fig)

                if len(pts) == 2:
                    (sx, sy), (gx, gy) = pts[0], pts[1]
                    sx, sy = max(0.5, min(hmap.width - 0.5, sx)), max(0.5, min(hmap.height - 0.5, sy))
                    gx, gy = max(0.5, min(hmap.width - 0.5, gx)), max(0.5, min(hmap.height - 0.5, gy))

                    syaw = np.arctan2(gy - sy, gx - sx)
                    print(f"\n[OK] Đã chọn từ chuột -> START: ({sx:.2f}, {sy:.2f}) | GOAL: ({gx:.2f}, {gy:.2f})")
                    return (sx, sy, syaw), (gx, gy, syaw)
        except Exception:
            pass

    # Terminal input mode
    default_start = (round(hmap.width * 0.1, 2), round(hmap.height * 0.1, 2), 0.0)
    default_goal = (round(hmap.width * 0.85, 2), round(hmap.height * 0.85, 2), 0.0)

    print("\n----------------------------------------------------------")
    print(f" Tọa độ mặc định cho Bản đồ '{map_name}' (Kích thước: {hmap.width:.1f}m x {hmap.height:.1f}m):")
    print(f"  - START: ({default_start[0]:.2f}, {default_start[1]:.2f})")
    print(f"  - GOAL:  ({default_goal[0]:.2f}, {default_goal[1]:.2f})")
    print("----------------------------------------------------------")

    try:
        ans = input("Nhấn ENTER để dùng mặc định, hoặc nhập 'c' để nhập tọa độ tùy chỉnh: ").strip().lower()
        if ans == 'c':
            print("\n>> Nhập tọa độ (có thể dùng dấu phẩy hoặc dấu cách, vd: 10, 15 hoặc 10 15). Để trống = dùng mặc định:")
            
            s_raw = input(f"Nhập tọa độ START x,y [Mặc định: {default_start[0]}, {default_start[1]}]: ")
            sx, sy = parse_coordinate_input(s_raw, default_start, point_name="START", max_w=hmap.width, max_h=hmap.height)

            g_raw = input(f"Nhập tọa độ GOAL x,y  [Mặc định: {default_goal[0]}, {default_goal[1]}]: ")
            gx, gy = parse_coordinate_input(g_raw, default_goal, point_name="GOAL", max_w=hmap.width, max_h=hmap.height)

            syaw = np.arctan2(gy - sy, gx - sx)
            print(f"\n>> [XÁC NHẬN] Đã thiết lập thành công: START=({sx:.2f}, {sy:.2f}) -> GOAL=({gx:.2f}, {gy:.2f})\n")
            return (sx, sy, syaw), (gx, gy, syaw)
    except (EOFError, KeyboardInterrupt):
        pass

    print(f"\n>> [XÁC NHẬN] Dùng tọa độ mặc định: START=({default_start[0]:.2f}, {default_start[1]:.2f}) -> GOAL=({default_goal[0]:.2f}, {default_goal[1]:.2f})\n")
    return default_start, default_goal
