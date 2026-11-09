#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cấu hình chuẩn RELLIS-3D, bảng màu, chi phí ngữ nghĩa và tham số lưới 2.5D.
Hỗ trợ 2 pipeline:
  - jikay  : Pipeline đầy đủ từ Jikay71 (lưới BEV 200x150, lọc thân xe, nội suy EDT,
             độ dốc từ bề mặt Gaussian, inflation layer, A* Semantic vs Geo-only).
  - group1 : Pipeline theo báo cáo Nhóm 1 (lưới 150x200, Z phân vị 25%, valid_mask).
"""

RELLIS_NAMES = {
    0: "void",
    1: "dirt",
    3: "grass",
    4: "tree",
    5: "pole",
    6: "water",
    7: "sky",
    8: "vehicle",
    9: "object",
    10: "asphalt",
    12: "building",
    15: "log",
    17: "person",
    18: "fence",
    19: "bush",
    23: "concrete",
    27: "barrier",
    31: "puddle",
    33: "mud",
    34: "rubble",
}

RELLIS_COLORS = {
    0: (0, 0, 0),          # void
    1: (108, 64, 20),      # dirt
    3: (0, 102, 0),        # grass
    4: (0, 255, 0),        # tree
    5: (0, 153, 153),      # pole
    6: (0, 128, 255),      # water
    7: (0, 0, 255),        # sky
    8: (255, 255, 0),      # vehicle
    9: (255, 0, 127),      # object
    10: (64, 64, 64),      # asphalt
    12: (255, 0, 0),       # building
    15: (102, 0, 0),       # log
    17: (204, 153, 255),   # person
    18: (102, 0, 204),     # fence
    19: (255, 153, 204),   # bush
    23: (170, 170, 170),   # concrete
    27: (41, 121, 255),    # barrier
    31: (134, 255, 239),   # puddle
    33: (99, 66, 34),      # mud
    34: (110, 22, 138),    # rubble
}

# Bảng chi phí ngữ nghĩa theo Jikay71 (0: dễ đi nhất, 255: cấm đi)
SEMANTIC_COST_TABLE = {
    0: 100,    # void / unlabeled
    1: 15,     # dirt
    3: 35,     # grass
    4: 255,    # tree
    5: 255,    # pole
    6: 255,    # water
    7: 100,    # sky
    8: 255,    # vehicle
    9: 180,    # object (đưa vào LETHAL_CLASSES -> 255)
    10: 10,    # asphalt
    12: 255,   # building
    15: 255,   # log
    17: 255,   # person
    18: 255,   # fence
    19: 120,   # bush
    23: 10,    # concrete
    27: 255,   # barrier
    31: 75,    # puddle
    33: 110,   # mud
    34: 90,    # rubble
}

LETHAL_CLASSES = {4, 5, 6, 8, 9, 12, 15, 17, 18, 27}

# Bảng chi phí & vật cản theo chuẩn báo cáo Nhóm 1 (reproduce 000000..000005.npz)
GROUP1_SEMANTIC_COST_TABLE = {
    0: 150,    # void
    1: 20,     # dirt
    3: 40,     # grass
    10: 10,    # asphalt
    15: 100,   # log
    19: 90,    # bush
    23: 10,    # concrete
    31: 80,    # puddle
    33: 70,    # mud
    34: 90,    # rubble
}
GROUP1_OBSTACLE_IDS = {4, 5, 6, 8, 12, 15, 17, 18, 19, 27}


class RellisConfig:
    """Tham số cấu hình lưới 2.5D từ điểm LiDAR Ouster OS1-64."""

    # Độ phân giải ô lưới (m/ô)
    RES = 0.2

    # Vùng quan tâm (ROI) trong hệ trục gắn cảm biến/xe
    X_MIN = -10.0
    X_MAX = 30.0
    Y_MIN = -15.0
    Y_MAX = 15.0
    Z_MIN = -3.0
    Z_MAX = 5.0

    # Kích thước ma trận BEV kiểu Jikay71: (H=200 dọc X, W=150 ngang Y)
    GRID_H = int(round((X_MAX - X_MIN) / RES))  # 200
    GRID_W = int(round((Y_MAX - Y_MIN) / RES))  # 150

    # Kích thước ma trận chuẩn Group 1: (H=150 dọc Y, W=200 dọc X)
    GROUP1_H = int(round((Y_MAX - Y_MIN) / RES))  # 150
    GROUP1_W = int(round((X_MAX - X_MIN) / RES))  # 200

    # Hộp bao thân xe Polaris W-MRZR + người lái (lọc nhiễu quanh cảm biến)
    VEHICLE_X_MIN = -1.2
    VEHICLE_X_MAX = 2.4
    VEHICLE_Y_MIN = -1.3
    VEHICLE_Y_MAX = 1.3
    VEHICLE_Z_MIN = -1.3

    # Ngưỡng động học / hình học xe
    MAX_SLOPE_DEG = 30.0
    MAX_STEP_HEIGHT = 0.35
    MAX_ROUGHNESS = 0.25

    # Trọng số hợp nhất costmap (mode jikay)
    W_SEM = 0.50
    W_SLOPE = 0.30
    W_ROUGH = 0.20

    # Mở rộng vùng an toàn quanh vật cản (số ô)
    INFLATION_CELLS = 2

