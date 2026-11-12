# HANDOFF: Cải thiện thuật toán 2.5D bằng các bộ dữ liệu mới

> Bàn giao: 2026-11-09 (cập nhật sau GĐ 0 trên máy mới) · Repo: https://github.com/TueLam06/Path-Planning-2.5D- (nhánh `main`)
> Mục tiêu phiên tiếp theo: **chạy bộ auto-test trên các bộ dữ liệu mới → tìm điểm yếu → sửa thuật toán →
> chạy lại để chứng minh cải thiện**, không làm hỏng những gì đang đúng.

---

## 0. Định nghĩa "xong"
- [ ] Chạy được ít nhất **2 bộ dữ liệu mới** ngoài 8 frame RELLIS mẫu. BenchNav đã chạy thử nhưng quá dễ (§4.1); tiếp theo: bản đồ dốc tự sinh (§4.1b) → RELLIS đầy đủ (§4.2).
- [ ] Bộ test **phân biệt được thuật toán**: ít nhất 1 planner có valid < 90% ở nhóm khó (hiện tại mọi planner đạt 100% trên jikay8 nên chưa xếp hạng được).
- [ ] Mỗi thay đổi thuật toán có **số liệu trước/sau** trong `logs/testsuite/history.csv`.
- [ ] `false_success = 0` ở mọi lần chạy (đây là test hồi quy bắt buộc).
- [x] RRT\* trung bình < 3 s/case: 10.9 → 1.46 s trên jikay (GĐ 1).

---

## 0.5 Chuyển sang máy mới

**Có trong git (clone là có):** toàn bộ code, 6 bản đồ group1 + 100 bản đồ BenchNav (`map_data/`), 1 bản đồ RELLIS mẫu (`rellis_25d_generator/samples/`), bộ 40 case `testsuite/testcases.json`, `CLAUDE.md`, `HANDOFF.md`.

**KHÔNG có trong git (máy cũ, phải làm lại nếu cần):**
| Thứ | Ở máy cũ | Trên máy mới |
|---|---|---|
| Môi trường Python | venv tạm | `pip install -r requirements.txt` (torch **không** còn bắt buộc) |
| 8 frame RELLIS mẫu (24 MB) | `RELLIS 2.5D/3. code/rellis_25d_generator/raw_data/` | `python auto_pipeline.py ... --download-sample` tự tải |
| Log các lần chạy (`logs/`) | `Path-Planning-2.5D/logs/`, `RELLIS 2.5D/4. logs/testsuite/` | Không cần mang theo; số liệu chính đã ghi ở §1.3 |
| Lần chạy BenchNav dở (381/1.200 job) | `logs/testsuite/benchnav_base/` | Chạy lại từ đầu (lệnh ở §4.1) |
| Báo cáo LaTeX, `roadmap.md`, PDF bài báo | thư mục `RELLIS 2.5D/` (repo Whitenessly) | Không thuộc repo này |

Việc đầu tiên trên máy mới:
```bash
git clone https://github.com/TueLam06/Path-Planning-2.5D-.git && cd Path-Planning-2.5D-
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
python auto_pipeline.py --name smoke --download-sample --seqs 00000 --stride 1 --per-map 3 --workers 4
```
Đạt nếu `logs/testsuite/smoke/REPORT.md` báo **0 lần báo success sai**.

> Windows: nếu `python` báo "not found" (alias Microsoft Store) thì tạo venv bằng `py -3.14 -m venv .venv`. Chưa dùng 3.15 (có thể thiếu wheel scipy).

---

## 0.6 Lộ trình (đã chốt 2026-11-09)

Thứ tự: **chốt định nghĩa benchmark → dựng bộ test khó → sửa planner**. Sửa thuật toán khi bộ test còn dễ thì không chứng minh được cải thiện.

| GĐ | Việc | Mục §5 | Trạng thái |
|---|---|---|---|
| 0 | venv, smoke, baseline 40 case `--workers 1` trên máy mới | | ✅ xong (số liệu ở §1.3) |
| 1 | Tăng tốc RRT\* (KD-tree / lưới băm, cache cạnh), mục tiêu < 3 s/case | P1-b | ✅ xong (§1.4) |
| 2 | Chốt định nghĩa benchmark: dung sai goal chung, kích thước xe Warthog | P1-c, P2-g | ✅ xong (§1.5) |
| 3 | Bộ test khó: địa hình dốc tự sinh, tag độ khó, case âm; BenchNav `t_classes` → costmap | P1-a, §4.1, §4.1b | 🔶 đang làm (§1.6): bản đồ tổng hợp + case âm xong |
| 4 | Sửa planner theo điểm yếu: Field D\* roll, T-Hybrid chân đế + lùi, làm mượt Z, LazyPRM\* | P2-d/e/f, P3 | |
| song song | Tải RELLIS-3D đầy đủ (14 GB); chạy dev sau GĐ 1–2, test chỉ khi đã chốt | §4.2 | |

Lý do GĐ 2 đứng trước GĐ 3: đổi dung sai goal và kích thước xe làm thay đổi chính bài toán, mọi baseline đo trước đó phải chạy lại.

---

## 1. Trạng thái hiện tại

### 1.1 Repo
```
.
├── auto_pipeline.py / run_auto.bat   # 1 lệnh: LiDAR -> bản đồ -> test case -> planner -> REPORT.md
├── rellis_25d_generator/             # LiDAR (.bin/.label, thư mục hoặc .zip) -> bản đồ 2.5D .npz
├── path_planning_benchmark/
│   ├── planners/                     # T-Hybrid A*, RRT*, Field D*, LazyPRM* (+ baseline 2D)
│   ├── map_utils/hybrid_map.py       # lớp bản đồ dùng chung, is_segment_free(), chiếu chân đế
│   ├── map_utils/map_loader.py       # đọc .npz (jikay/group1), .pt (BenchNav), .npy
│   ├── benchnav/testsuite.py         # load map, build planner, đường tham chiếu, kiểm chứng độc lập
│   ├── generate_testcases.py         # sinh start/goal tự động
│   ├── run_testsuite.py              # chạy hàng loạt, ghi từng dòng, --resume
│   ├── map_data/                     # 6 bản đồ group1 (LỖI, chỉ đối chứng) + 100 bản đồ BenchNav .pt
│   └── testsuite/testcases.json      # 40 case trên 7 bản đồ cũ
└── tools/photo_to_lidar.py           # 1 ảnh RGB -> .bin/.label (demo)
```

### 1.2 Lịch sử commit quan trọng
- `f928430 fix: sửa kiểm tra va chạm và báo thành công sai của các planner`: xem diff commit này để hiểu các lỗi đã sửa (kiểm tra va chạm thưa, Field D\* đặt `found = True` khi chưa tới đích, RRT\* nối goal không kiểm tra, LazyPRM\* không kiểm tra góc lật).
- Commit cũ hơn là code gốc của nhóm.

### 1.3 Kết quả tham chiếu (baseline cần vượt)
| Bộ | Case | Valid (sem): T-Hybrid / RRT\* / Field D\* / LazyPRM\* | Thời gian TB (s) | Ghi chú |
|---|---|---|---|---|
| 7 bản đồ cũ, code gốc | 40 | 22% / 5% / 48% / 38% | 0.20 / 1.6 / 0.10 / 0.21 | 95 lần báo success sai |
| 7 bản đồ cũ, sau khi sửa | 40 | 75% / 55% / 100% / 45% | 0.25 / 2.2 / 0.81 / 0.58 | 0 lần báo sai |
| 8 bản đồ jikay (RELLIS mẫu) | 72 | 100% / 100% / 100% / 100% | 0.34 / 12.0 / 2.38 / 1.55 | **Quá dễ**, 1 cảnh duy nhất |
| BenchNav 100 bản đồ, chế độ geo (dừng ở 381/1.200 job, 5 worker) | 300 | 100% / 100% / 100% / 100% | 0.26 / 15.6 / 1.00 / 3.54 | Không có vật cản; max roll TB 14–17°, không vượt giới hạn; opt-ratio 0.96–1.01 |

> Log chi tiết của 3 lần chạy này nằm ở máy cũ: `Documents/RELLIS 2.5D/4. logs/testsuite/{rellis_full_v1, rellis_full_v3_fixed, jikay8_v1}/`.

**Baseline trên máy mới (GĐ 0, commit `9f3880c`, Python 3.14, numpy 2.5 / scipy 1.18).** Đây là mốc so sánh thời gian cho các thay đổi tiếp theo, không dùng số thời gian của máy cũ.

| Lần chạy (`logs/testsuite/`) | Mode | Valid: T-Hybrid / RRT\* / Field D\* / LazyPRM\* | Thời gian TB (s) | False success |
|---|---|---|---|---|
| `base40_gd0` (40 case cũ, `--workers 1`) | sem | 75% / 55% / 100% / 45% | 0.20 / 1.69 / 0.60 / 0.44 | 0 |
| `base40_gd0` | geo | 78% / 65% / 100% / 62% | 0.21 / 2.13 / 0.60 / 0.42 | 0 |
| `smoke` (8 frame jikay, 24 case, 4 worker) | sem | 100% / 100% / 100% / 100% | 0.35 / 13.0 / 2.60 / 1.67 | 0 |
| `smoke` | geo | 100% / 100% / 100% / 100% | 0.26 / 13.1 / 1.88 / 1.40 | 0 |

- Valid trên 40 case cũ **khớp đúng** máy cũ → môi trường mới tái lập được kết quả.
- Nhóm `long` trên 40 case: chỉ Field D\* đạt (100%), 3 planner còn lại 0%. Field D\* lại có max roll TB cao nhất (25–26°, vượt roll 22–25%) vì chưa kiểm tra roll (P2-e).
- Opt. ratio < 1 (0.84–0.86) ở T-Hybrid/RRT\* nhiều khả năng do dừng sớm trong dung sai goal 2.5 m, còn đường tham chiếu đi tới đúng goal (Field D\* dung sai 1.2 m → 0.95; LazyPRM\* tới đúng goal → 1.02). Opt. ratio chưa so sánh công bằng được cho tới khi xong P2-g.
- RRT\* trên jikay vẫn khoảng 13 s/case → GĐ 1.

### 1.4 GĐ 1: tăng tốc RRT\* (xong)
Profile 1 case jikay: **82% thời gian nằm trong `project_footprint_3point`** (khoảng 130.000 lần gọi/plan qua `_slope_cost`). Đã sửa, **giữ nguyên kết quả bit-exact**:
- `HybridMap.project_footprint_3point_batch`: chiếu chân đế cho nhiều pose cùng lúc; dùng `matmul` để làm tròn giống hệt `np.dot` của bản scalar (đã so 45.000 pose ngẫu nhiên: 0 khác biệt).
- `RRTStar25D`: `_is_edge_valid` dùng bản batch; cache slope cost theo node; bỏ tính lại slope bất biến trong vòng chọn parent; tìm nearest/near bằng mảng numpy.

Đo trên máy cũ (Windows 10, Python 3.13, numpy 2.4, `--workers 1`; log ở `logs/gd1/` + `logs/testsuite/gd1_*`):

| Bộ | Trước | Sau | Tăng tốc | Kết quả từng job |
|---|---|---|---|---|
| 40 case, RRT\* sem + geo (80 job) | 172.5 s (1.90 / 2.41 s/case) | 19.7 s (0.22 / 0.27 s/case) | **8.8×** | 80/80 giống hệt |
| 24 case jikay, RRT\* sem | 10.9 s/case | **1.46 s/case** | 7.5× | 24/24 giống hệt |

Hồi quy cả 4 planner (40 case + 24 case jikay, sem + geo): valid giữ nguyên 75/55/100/45% (sem) và 78/65/100/62% (geo) trên 40 case, 100% trên jikay; `false_success = 0`. Đo lại thời gian trên máy mới để cập nhật baseline của máy đó.

### 1.5 GĐ 2: chuẩn hoá benchmark (xong) → **baseline mới**
Mọi số liệu trước GĐ 2 **không còn so sánh được** với số liệu sau GĐ 2 (bài toán đã đổi).

| Chuẩn | Giá trị | Ở đâu |
|---|---|---|
| Kích thước xe | **Clearpath Warthog 1.52 × 1.38 m** (thông số Clearpath; không có số bề rộng vệt bánh nên dùng kích thước ngoài) | `HybridMap.ROBOT_LENGTH/ROBOT_WIDTH` |
| Giới hạn lật | roll = pitch = **0.55 rad**, chân đế 3 điểm, **cả 4 planner** | `HybridMap.ROLL_MAX/PITCH_MAX`, `poses_stable()` |
| Dung sai goal | **1.0 m**, chung cho planner và bộ kiểm chứng; planner so `<= tol − 1e-6` để không vượt do làm tròn | `benchnav/testsuite.py: GOAL_TOL`, `--goal-tol` |
| Đường tham chiếu | Dijkstra lưới 8 hướng → **kéo căng any-angle** → độ dài 3D theo địa hình | `shortcut_path`, `polyline_len_3d`; trường `ref_len_3d_anyangle_m` |

Thay đổi planner: T-Hybrid A\* kiểm tra chân đế (thay pháp tuyến 1 ô, pitch 0.60 → 0.55) và cố nối tới đúng goal; Field D\* có bảng ổn định 16 hướng tính trước, đo khoảng cách tới đúng goal; RRT\*/LazyPRM\* lấy giới hạn từ `HybridMap`.

Baseline mới (máy cũ, `--workers 1`, log `logs/testsuite/gd2_*`):

| Bộ | Mode | Valid: T-Hybrid / RRT\* / Field D\* / LazyPRM\* | Thời gian TB (s) | Opt. ratio (any-angle) | False success |
|---|---|---|---|---|---|
| 40 case | sem | 28% / 28% / 62% / 25% | 0.18 / 0.32 / 0.79 / 0.43 | 1.02–1.07 | 0 |
| 40 case | geo | 48% / 32% / 98% / 32% | 0.22 / 0.35 / 0.88 / 0.38 | 1.03–1.08 | 0 |
| 24 case jikay | sem | 100% / 100% / 100% / 100% | 0.29 / 1.78 / 3.28 / 1.18 | 1.00–1.02 | 0 |
| 24 case jikay | geo | 100% / 100% / 100% / 100% | 0.21 / 1.80 / 2.55 / 1.01 | 1.01–1.03 | 0 |

**Vì sao valid trên 40 case giảm mạnh** (ablation `logs/gd2/ablation.py`): do **kích thước xe** (giảm 20–40 điểm), dung sai goal chỉ 0–10 điểm. Khoảng 98–99.8% tư thế vượt lật có góc chân đế rơi vào **ô chưa đo** (Z = 0 trên bản đồ group1) → vách giả. Chân đế Warthog chạm vào vách giả nhiều gấp đôi (6.7% → 14.7% cặp ô–hướng). ⇒ **Bộ 40 case (34/40 case trên group1) không hợp làm bộ hồi quy nữa**; GĐ 3 cần bộ hồi quy mới trên bản đồ jikay / tự sinh.

Còn lệch, chưa sửa: cột "Vượt roll" (bộ kiểm chứng) vẫn 9–38% ở T-Hybrid/RRT\*/LazyPRM\* vì planner kiểm tra pose khác bộ đo (yaw nội suy giữa 2 node, khoảng cách mẫu 0.4 m, cache cạnh vô hướng của LazyPRM\* với chân đế không đối xứng). Field D\* = 0% vì kiểm tra đúng pose như bộ đo (ô + hướng bước). Cần thống nhất "pose = waypoint + hướng tới waypoint kế".

### 1.6 GĐ 3 (phần 1): bộ test tổng hợp có kiểm soát độ khó + case âm
- `generate_synthetic_suite.py`: sinh **53 bản đồ** 40 × 30 m (ô 0.2 m) + **110 case** thiết kế sẵn (12 case âm), seed cố định. Bản đồ ở `map_data/synthetic/` (**không commit**, chạy lại script để sinh); case ở `testsuite/testcases_synthetic.json`.

| Họ | Tham số | Thử điểm yếu gì |
|---|---|---|
| `gap` | khe 1.0–3.2 m | chân đế qua khe hẹp |
| `slope_band` | dốc ngang 10 / 28 / 33° | dễ (33° = case âm) |
| `saddle` | sống núi > 45°, đèo rộng 2–6 m, sườn 20–27° | tìm lối qua địa hình dốc |
| `corridor_s` | hành lang chữ S rộng 1.2–2.8 m (bán kính cua nhỏ nhất khoảng 1.3 m) | rẽ liên tục sát tường |
| `maze` | ô 4–6 m | đường vòng dài |
| `boulders` / `boulders_dense` | 15–50 / 80–160 tảng | né vật cản thưa / dày |
| `blocked` | tường kín | case âm |

- **Case âm** (`feasible: false`): không có đường kể cả coi xe là 1 điểm, hoặc lối đi (khe/hành lang/đèo) hẹp hơn bề rộng xe 1.38 m. Planner đúng khi **báo thất bại**.
- `run_testsuite.py`: thêm cột **"Từ chối đúng"** (case âm), valid chỉ tính trên case có đường; `auto_pipeline.py` xếp "case khó nhất" theo `correct`.
- Kết quả (`logs/testsuite/gd3_synth2`, geo, 4 worker nên thời gian chỉ tham khảo):

| Planner | Valid | Từ chối đúng | gap | corridor_s | maze | boulders_dense | saddle | Thời gian TB (s) |
|---|---|---|---|---|---|---|---|---|
| T-Hybrid A\* | 88% | 100% | 100% | **0%** | 67% | 89% | 100% | 1.1 |
| RRT\* | 68% | 100% | 69% | 17% | **0%** | 50% | 100% | 1.2 |
| Field D\* | 98% | 100% | 100% | 67% | 100% | 100% | 100% | **9.4** |
| LazyPRM\* | 90% | 100% | 69% | 17% | 100% | 100% | 100% | 3.0 |

- **Phân biệt được planner** (tiêu chí GĐ 3 đạt), `false_success = 0`:
  - `corridor_s`: rộng 1.6 m cả 4 đều hỏng (có thể bất khả thi thật với chân đế 1.52 × 1.38 m khi cua gắt); 2.0 m chỉ Field D\* qua; 2.8 m Field D\* 100%, RRT\*/LazyPRM\* 50%, **T-Hybrid 0%**.
  - `maze`: RRT\* 0%; T-Hybrid chỉ qua ô 6 m.
  - `boulders_dense` 160 tảng: RRT\* 0%, T-Hybrid 67%.
- **T-Hybrid A\* hỏng ở hành lang chữ S vì động học**: mô hình xe đạp dùng `robot_length` làm chiều dài cơ sở. Từ GĐ 2 giá trị này là 1.52 m → bán kính quay tối thiểu = 1.52 / tan(0.55) ≈ **2.5 m** (trước GĐ 2 là khoảng 1.1 m), lớn hơn bán kính cua của hành lang. Nhưng Warthog lái **skid-steer** (quay tại chỗ được), nên mô hình Ackermann không đúng với xe thật → P2-f.
- Còn dễ: `saddle`, `slope_band`, `boulders` đều 100%. Dốc trơn không bao giờ chạm giới hạn lật 31.5° vì `rho_max` (29.8°) chặn ô trước.
- **Field D\* chậm** (7–13 s/case trên bản đồ 40 × 30 m): bảng ổn định 16 hướng + `is_segment_free` mỗi cạnh + tối đa 60.000 vòng khi không có đường → P3-h.

### 1.7 RELLIS-3D sequence 00000: 57 cảnh đại diện (dữ liệu bạn trong nhóm sinh)
- Dữ liệu: `out_00000/` (2.847 `.npz` jikay của cả sequence, **không commit**, đã `.gitignore` `out_*/`). Cảnh đại diện = 59 frame có ảnh dashboard (0, 1, 2 rồi mỗi 50 frame); bỏ 1 và 2 (trùng 0) → **57 cảnh**, 159 case (57 short, 57 medium, 45 long; 12 bản đồ không đủ case long). Trung bình chỉ **36% ô đi được** (8 frame mẫu: 91%).
- Lệnh: `generate_testcases.py --maps-list ../logs/rellis00/maps_rep57_rel.txt --per-map 3`, rồi `run_testsuite.py --modes sem geo --workers 2` (log `logs/testsuite/rellis00_rep57*`).

| Mode | Planner | Valid | short / medium / long | Thời gian TB (s, 2 worker) | False success |
|---|---|---|---|---|---|
| sem | T-Hybrid A\* | 72% | 84 / 67 / 64% | 0.72 | 0 |
| sem | RRT\* | 61% | 74 / 58 / 49% | 0.55 | 0 |
| sem | Field D\* | **89%** | 96 / 88 / 80% | 2.26 | 0 |
| sem | LazyPRM\* | 82% | 93 / 81 / 69% | 3.58 | 0 |
| geo | T-Hybrid A\* | 91% | 91 / 91 / 89% | 0.28 | 0 |
| geo | RRT\* | 77% | 88 / 75 / 67% | 0.66 | 0 |
| geo | Field D\* | **97%** | 100 / 98 / 93% | 2.02 | 0 |
| geo | LazyPRM\* | 91% | 95 / 89 / 87% | 2.44 | 0 |

- Thứ hạng khớp bộ tổng hợp: **Field D\* > LazyPRM\* ≈ T-Hybrid A\* > RRT\***. Chế độ `sem` khó hơn `geo` 8–19 điểm (ô ngữ nghĩa nguy hiểm làm hẹp vùng đi được). Hầu hết thất bại là "không tìm được đường" (RRT\* 98 lần), không phải đi sai.
- 3/159 case không planner nào giải được (frame 2050–2150, cụm cảnh hẹp).
- Lần chạy đầu lộ **6 false success** (LazyPRM\* 5, T-Hybrid A\* 1), mỗi lần chạm vật cản đúng 1 mẫu: planner kiểm tra cả cạnh dài, bộ kiểm chứng kiểm tra từng đoạn ngắn giữa các điểm xuất ra → lấy mẫu lệch vài cm. Đã sửa: LazyPRM\* kiểm tra đúng các đoạn 0.25 m sẽ xuất ra (cả 2 chiều cạnh); T-Hybrid A\* kiểm tra từng bước 0.4 m của đoạn nối thẳng tới goal. Hồi quy bộ tổng hợp + 40 case: valid giữ nguyên, `false_success = 0`. Đổi lại LazyPRM\* chậm hơn (khoảng 1.5–2× do kiểm tra cạnh 2 chiều).
- **Lọc trùng theo nội dung (`--dedup`) gần như vô dụng khi xe đang chạy**: 2.847 → 2.183 bản đồ ở ngưỡng 0.25 m. Bản đồ gắn theo xe nên lưới bị dịch theo chuyển động, so khối 2 m không chịu được dịch. Cần lọc theo pose (`--min-move`) hoặc so sánh chịu được dịch chuyển (căn chỉnh trước khi so).

### 1.8 Xem kết quả trực quan
- `run_testsuite.py --save-paths`: lưu toạ độ đường đi + điểm cắt vật cản + waypoint vượt lật của từng job vào `logs/testsuite/<run>/paths/`.
- `plot_results.py <run> [--only-failures]`: mỗi (case, chế độ) 1 ảnh 2×2 (mỗi ô 1 planner, cùng bản đồ) + trang `viewer/index.html` lọc theo chế độ / nhóm / trạng thái / planner hỏng. Một khung 1 planner vì 4 màu chồng nhau không phân biệt được với người mù màu (đã kiểm bằng validator của skill dataviz).
- `auto_pipeline.py --plot` (hoặc `--plot failures`): tự lưu đường đi và vẽ sau bước chạy; link ghi trong `REPORT.md`.
- Mở trang: chạy `python -m http.server 8765 --directory logs/testsuite` rồi vào `http://localhost:8765/<run>/viewer/index.html` (hoặc mở thẳng file `index.html`). Cấu hình xem trong app: `.claude/launch.json` (tên `test-viewer`, không commit).
- **Phát hiện nhờ ảnh**: 3 case RELLIS "không planner nào giải được" (frame 2050–2150) đều kẹt ở cùng **một lối thắt hẹp hơn thân xe**. Bộ sinh case coi xe là 1 điểm nên đánh dấu "có đường", nhưng Warthog 1.38 m không qua được → khả năng cao là case âm thật. Cần thêm kiểm tra **khoảng trống dọc đường tham chiếu ≥ nửa bề rộng xe** vào `generate_testcases.py` (P1-a).

---

## 2. Cài đặt
```bash
git clone https://github.com/TueLam06/Path-Planning-2.5D-.git
cd Path-Planning-2.5D-
python -m venv .venv && .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# torch chỉ cần cho tools/photo_to_lidar.py; bản đồ BenchNav .pt đã đọc được bằng numpy
```
Kiểm tra nhanh (khoảng 2 phút, tự tải 24 MB mẫu RELLIS):
```bash
python auto_pipeline.py --name smoke --download-sample --seqs 00000 --stride 1 --per-map 3 --workers 4
```
Xem `logs/testsuite/smoke/REPORT.md`.

---

## 3. Vòng lặp làm việc (lặp cho mỗi cải tiến)

```
 (1) baseline   ->  (2) chạy bộ dữ liệu  ->  (3) đọc REPORT.md  ->  (4) sửa 1 planner
      ^                                         "10 case khó nhất"          |
      |                                                                     v
 (7) commit  <-  (6) so history.csv  <-  (5) smoke: false_success = 0 ?  <--+
```

1. **Baseline:** chạy bộ dữ liệu với code hiện tại, đặt tên rõ ràng (ví dụ `benchnav_base`).
2. **Chạy:** `auto_pipeline.py` (dữ liệu LiDAR) hoặc `generate_testcases.py` + `run_testsuite.py` (bản đồ có sẵn).
3. **Phân tích:** `REPORT.md` §5 liệt kê 10 case khó nhất; `results.csv` lọc theo `planner`, `valid`, `collision_samples`, `goal_error_m`, `max_roll_deg`.
4. **Sửa đúng 1 vấn đề**, ghi rõ giả thuyết.
5. **Hồi quy:** chạy lại smoke và bộ 40 case cũ → `false_success` phải bằng 0, valid không được giảm.
6. **So sánh:** cùng `testcases.json`, cùng seed → mở `logs/testsuite/history.csv` (1 dòng/planner/lần chạy, có cột git commit).
7. **Commit** theo chuẩn, 1 dòng tiếng Việt, ví dụ `perf: tăng tốc RRT* bằng KD-tree`, `fix: ...`.

**Quy tắc chống "học thuộc" bộ test:** chỉ chỉnh tham số trên tập **dev**. Tập **test** chỉ chạy khi đã chốt (xem §4.2).

Lệnh chạy lại cùng bộ case cho một planner cụ thể:
```bash
cd path_planning_benchmark
python run_testsuite.py --cases ../logs/testsuite/<run>/testcases.json --planners "2.5D RRT*" --modes sem --workers 1 --name rrt_after_fix
```

---

## 4. Bộ dữ liệu (theo thứ tự nên làm)

### 4.1 BenchNav: đã có trong repo, ĐÃ CHẠY THỬ
- `path_planning_benchmark/map_data/from_benchnav/*.pt`: 100 bản đồ tổng hợp (nguồn: https://github.com/masafumiendo/benchnav), **4 nhóm** (`000`–`003`) × 25 bản đồ.
- Mỗi file: lưới **64 × 64**, `heights` (chênh khoảng 1.4–1.65 m), `slopes` (độ, max khoảng 19–24°), `t_classes` (lớp địa hình), `colors`, và dự đoán độ trượt (`distributions`).
- **Đã sửa** (`map_utils/map_loader.py`): đọc `.pt` bằng numpy (không cần torch); độ phân giải **0.5 m** theo `scripts/generate_terrain_dataset.py` của BenchNav (trước đó gán sai 0.25 m → dốc gấp đôi, sinh vật cản giả). Bản đồ = **32 × 32 m**, độ dốc tính lại khớp lớp `slopes` của file.
- **Kết quả:** bản đồ **100% ô đi được** (dốc < ngưỡng 30°) → mọi planner đạt 100%. BenchNav **chưa giúp phân biệt planner** ở dạng hiện tại.
- **Độ khó thật của BenchNav là độ trượt theo loại địa hình** (10 lớp `t_classes`, mỗi nhóm dùng 4 lớp: `000`: 3,4,5,7 · `001`: 3,4,8,9 · `002`: 0,2,5,8 · `003`: 4,7,8,9). Planner hiện không dùng thông tin này.
- **Việc tiếp theo với BenchNav:** đọc `src/environments/terrain_properties.py` và slip model trong repo BenchNav → ánh xạ `t_classes` thành chi phí (hoặc cấm đi với lớp trượt cao) → đưa vào `HybridMap` như một lớp `costmap` (`MapLoader` cần trả `metadata["costmap"]`), rồi chạy chế độ `sem`.
```bash
cd path_planning_benchmark
python generate_testcases.py --maps "map_data/from_benchnav/*.pt" --per-map 3 --out ../logs/benchnav/testcases_benchnav.json
python run_testsuite.py --cases ../logs/benchnav/testcases_benchnav.json --modes geo --workers 4 --name benchnav_base
```
> 300 case × 4 planner ≈ 1.200 job; RRT\* khoảng 15 s/case nên mất khoảng 1 giờ với 5 worker. Tối ưu RRT\* (P1-b) trước nếu cần chạy nhiều lần.

### 4.1b Bản đồ dốc tự sinh (không cần tải gì)
- `path_planning_benchmark/map_utils/terrain_generator.py` có sẵn bộ sinh địa hình. Sinh bản đồ có dốc 25–35° và vật cản để thử ràng buộc roll/pitch và khe hẹp. Đây là cách nhanh nhất để có bộ test **khó** trước khi tải RELLIS đầy đủ.

### 4.2 RELLIS-3D đầy đủ (dữ liệu chính)
- Nguồn: https://github.com/unmannedlab/RELLIS-3D (mục *LiDAR Download*). Link Google Drive:
  - *Ouster LiDAR SemanticKITTI Format* (14 GB): https://drive.google.com/file/d/1lDSVRf_kZrD0zHHMsKJ0V1GN9QATR4wH/view?usp=sharing
  - *Ouster LiDAR Annotation SemanticKITTI Format* (174 MB): https://drive.google.com/file/d/12bsblHXtob60KrjV7lGXUQTdC5PhV8Er/view?usp=sharing
  - *Ouster LiDAR Scan Poses files*: https://drive.google.com/file/d/1V3PT_NJhA41N7TBLp5AbW31d0ztQDQOX/view?usp=sharing (cho `--min-move`)
  - *Ouster LiDAR Split File* (75 KB): https://drive.google.com/file/d/1raQJPySyqDaHpc53KPnJVl3Bln6HlcVS/view?usp=sharing
  - Velodyne 32 kênh (tuỳ chọn): `.bin` 5.58 GB https://drive.google.com/file/d/1PiQgPQtJJZIpXumuHSig5Y6kxhAzz1cz/view?usp=sharing · nhãn 143.6 MB https://drive.google.com/file/d/1n-9FkpiH4QUP7n0PnQBp-s7nzbSzmxp8/view?usp=sharing
  - Baidu (dự phòng): https://pan.baidu.com/s/1akqSm7mpIMyUJhn_qwg3-w?pwd=4gk3 (mã `4gk3`)
- **Lọc cảnh trùng** (`auto_pipeline.py`, đã thử trên 8 frame mẫu: 8 → 1 bản đồ):
  - `--min-move 10 --min-turn 30`: chọn frame theo pose, chỉ giữ khi xe đã đi ≥ 10 m hoặc quay ≥ 30° (cần `poses.txt`, đặt trong `raw_data/` hoặc `--poses-dir`). Bỏ được đoạn xe đứng yên, tốt hơn `--stride`.
  - `--dedup`: bỏ bản đồ có lưới độ cao khối 2 m gần giống (RMSE < `--dedup-z` 0.15 m) một trong 20 bản đồ giữ gần nhất. Mặc định **không so nhãn** (`--dedup-sem 0`) vì nhãn lấp lỗ EDT không ổn định giữa các frame cùng cảnh.
  - Gợi ý: `python auto_pipeline.py --name rellis_dev --seqs 00000 00001 00002 --min-move 10 --dedup --workers 6`.
- Đặt **nguyên file .zip** vào `rellis_25d_generator/raw_data/` (hoặc ổ khác + `--data-dir`). **Không giải nén** (`data_loader` đọc thẳng zip; giải nén sẽ cần thêm 14 GB).
- Google Drive có thể giới hạn lượt tải → thử lại sau 24 giờ hoặc dùng link Baidu trong README RELLIS.
- 5 sequence: 00000 (2.847 scan) · 00001 (2.319) · 00002 (4.147) · 00003 (2.184) · 00004 (2.059). LiDAR 10 Hz → các frame liền nhau gần như trùng; dùng `--stride 50` (5 s) ≈ 270 bản đồ.
- **Chia tập theo sequence:** dev = 00000–00002, test = 00003–00004.
```bash
python auto_pipeline.py --name rellis_dev  --seqs 00000 00001 00002 --stride 50 --workers 6
python auto_pipeline.py --name rellis_test --seqs 00003 00004       --stride 50 --workers 1   # chỉ khi đã chốt
```
- Ước tính: 270 bản đồ × 9 case × 4 planner × 2 chế độ ≈ 19.000 job ≈ 21 giờ tuần tự với RRT\* hiện tại (khoảng 4–5 giờ với 6 worker). **Tối ưu RRT\* trước** (§5, P1-b).
- Máy hiện tại còn khoảng 26 GB trống trên ổ C:.

### 4.3 GOOSE (kiểm tra tổng quát hóa: khác nước, khác cảm biến)
- Paper: https://arxiv.org/abs/2310.16788 · Trang dữ liệu: goose-dataset.de
- 10.000 cặp ảnh + LiDAR **128 kênh** có nhãn từng điểm, **64 lớp** (GOOSE-Ex thêm 5.000 cặp).
- **Chưa xác minh định dạng file** (`.bin`? cấu trúc thư mục? cách mã hoá nhãn?) → đọc tài liệu trên goose-dataset.de trước.
- Cần viết `tools/adapters/goose_to_rellis.py` (hợp đồng ở §4.6) + **bảng ánh xạ 64 lớp → ID RELLIS** (đặc biệt: lớp mặt đất, thảm thực vật, nước, vật cản).
- Xe và cảm biến khác RELLIS → kiểm tra lại ROI, chiều cao cảm biến và hộp lọc thân xe trong `rellis_25d_generator/config.py`.

### 4.4 WildScenes (rừng tự nhiên, tuỳ chọn)
- Trang: https://csiro-robotics.github.io/WildScenes · Dữ liệu: https://data.csiro.au/collection/csiro:61541
- Theo README repo, script tải dữ liệu tạo **file pickle** cho phần 3D → adapter phải đọc pickle rồi xuất `.bin`/`.label`. Chưa xác minh hệ trục và cách mã hoá nhãn.

### 4.5 TartanDrive 2.0 (tuỳ chọn, chỉ hình học)
- https://arxiv.org/abs/2402.01913: khoảng 7 giờ lái ATV off-road, nhiều LiDAR. **Chưa xác nhận có nhãn từng điểm** → nếu không có thì chỉ chạy chế độ `geo`.

### 4.6 Hợp đồng adapter (cho mọi bộ dữ liệu mới)
Xuất ra đúng cấu trúc mà `rellis_25d_generator` đọc được:
```
<root>/<seq>/os1_cloud_node_kitti_bin/<frame>.bin                    # float32, N x 4: x y z intensity
<root>/<seq>/os1_cloud_node_semantickitti_label_id/<frame>.label     # uint32, 16 bit thấp = ID lớp RELLIS
```
- Hệ trục: **x tiến về trước, y sang trái, z hướng lên, gốc tại cảm biến** (mặt đất có z âm). ROI hiện tại: x ∈ [−10, 30), y ∈ [−15, 15), z ∈ [−3, 5).
- Không có nhãn → ghi toàn 0 (void) và chỉ chạy `--modes geo`.
- Sau đó: `python auto_pipeline.py --name <bo>_dev --data-dir <root> --seqs <seq...> --stride <n>`.
- Ví dụ mẫu đã có: `tools/photo_to_lidar.py` (hàm `save_outputs`) ghi đúng định dạng này.

---

## 5. Backlog cải thiện thuật toán (theo ưu tiên)

> Số dòng tính theo commit `4936970`. Mỗi mục: **vấn đề → cách sửa gợi ý → cách kiểm chứng**.

### P1: làm trước
**a. Bộ test quá dễ** · `generate_testcases.py:35` (`CATEGORIES`), `:53` (`generate_for_map`)
- Thêm tag độ khó: `detour_ratio` (đường tham chiếu / khoảng cách thẳng), `min_clearance` (khoảng trống nhỏ nhất dọc đường tham chiếu, dùng EDT trên `occupancy_2d`), `max_slope_on_ref`.
- Sinh case có chủ đích cho từng tag (quota), thêm **case âm** (start/goal khác vùng liên thông → planner phải báo thất bại) và **case semantic-critical** (đường ngắn nhất theo `costmap_geom` đi qua bùn/vũng/bụi, nhưng vẫn có đường an toàn theo `costmap`).
- Kiểm chứng: trên RELLIS dev, ít nhất 1 planner < 90% valid ở nhóm khó; bật/tắt ngữ nghĩa khác nhau rõ ở nhóm semantic-critical.

**b. ~~RRT\* chậm~~: xong ở GĐ 1** (§1.4). Còn tồn tại, chưa sửa (thay đổi kết quả nên để riêng):
- Rewire không cập nhật cost cho **các node con** của node vừa đổi parent → cost cây bị cũ, chọn đường chưa tối ưu.
- Chưa dừng sớm khi cost tới goal đã hội tụ (luôn chạy đủ `max_iterations`).

**c. ~~Kích thước xe sai~~: phía planner xong ở GĐ 2** (Warthog 1.52 × 1.38 m). Còn lại:
- Hộp lọc thân xe trong `rellis_25d_generator/config.py:122-126` vẫn theo "Polaris + người lái" 3.6 × 2.6 m. Đổi cần biết vị trí gắn LiDAR trên Warthog, và phải sinh lại bản đồ.
- Va chạm vẫn coi xe là **một điểm** (`is_segment_free` dọc tâm xe). Nếu muốn tính cả thân xe: phình `occupancy_2d` thêm nửa bề rộng xe (khoảng 0.69 m). Thay đổi này làm nhiều case hiện tại thành bất khả thi → quyết định riêng.

### P2
**d. Độ dốc tính trên Z chưa làm mượt** · `map_utils/hybrid_map.py:79-106` (`_compute_map_layers`: `slope > rho_max`, `roughness > h_max=0.15` → vật cản)
- Bản đồ jikay lấp ô trống bằng EDT nên có bậc thang → bị coi là vách. Ví dụ: bản đồ mẫu `samples/00000_000500` có 72% ô thành vật cản.
- Gợi ý: dùng lớp `slope` đã làm mượt trong `.npz` nếu có, hoặc làm mượt Gaussian giống generator; tính roughness chỉ trên ô `observed_mask`.
- Kiểm chứng: % ô vật cản giảm, valid tăng, `false_success` vẫn bằng 0.

**e. Field D\*: ~~không kiểm tra roll~~ (xong ở GĐ 2); tên chưa đúng bản chất** · `planners/field_d_star_2_5d.py`
- Đây thực chất là A\* trên lưới 16 hướng (không nội suy chi phí qua cạnh ô, không replanning). Hoặc hiện thực nội suy Field D\* thật, hoặc đổi tên trong báo cáo.

**f. T-Hybrid A\*: động học không khớp xe thật, chỉ đi tiến** · `planners/thybrid_a_star.py` (`_get_motion_primitives`: `L = self.map.robot_length`; `directions = [1]`). Kiểm tra roll bằng chân đế đã xong ở GĐ 2.
- Bán kính quay tối thiểu hiện khoảng 2.5 m (L = 1.52 m, `max_steer` 0.55 rad) → hỏng 100% ở `corridor_s`. Warthog là skid-steer: thêm primitive **quay tại chỗ** (hoặc tách chiều dài cơ sở khỏi `robot_length`).
- Thử primitive lùi (direction = −1, đã có hệ số phạt 1.2).
- Kiểm chứng: `corridor_s` 2.0–2.8 m qua được, không giảm ở các họ khác.

**g. ~~Dung sai goal khác nhau giữa planner~~: xong ở GĐ 2** (`GOAL_TOL = 1.0 m`). LazyPRM\* vẫn đi tới đúng goal (goal là node roadmap) nên chặt hơn 3 planner còn lại; chấp nhận được.

### P3
**h. `is_segment_free` tốn thời gian** · `map_utils/hybrid_map.py:173` (lấy mẫu mỗi 2.5 cm). Field D\* chậm từ 0.13 lên 0.8 s sau khi thêm kiểm tra này.
- Planner lưới: tính trước danh sách ô bị cắt cho 16 offset cố định (supercover), tra bảng thay vì lấy mẫu.

**i. LazyPRM\* chỉ đạt 45% trên bộ 40 case cũ** · `planners/lazy_prm_star_2_5d.py:19` (`num_samples`, `connection_radius`), `:46` (`_eval_edge`)
- Thử lấy mẫu tập trung gần vùng hẹp (bridge sampling / Gaussian sampling), tăng mẫu thích ứng khi không tìm thấy đường.

**j. `clear_around_point` sửa trực tiếp bản đồ** · `map_utils/hybrid_map.py:139`: planner tự xoá vật cản 1.0–1.2 m quanh start/goal. Bộ kiểm chứng dùng 1.2 m (`benchnav/testsuite.py:19`). Nếu đổi bán kính thì đổi cả hai.

---

## 6. Đọc kết quả
| File (`logs/testsuite/<run>/`) | Nội dung |
|---|---|
| `REPORT.md` | Tóm tắt: thông tin chạy, bảng kết quả, cảnh báo, 10 case khó nhất |
| `results.csv` | Từng job: `valid, reported_success, false_success, collision_samples, goal_error_m, planning_time_sec, path_length_3d, optimality_ratio, max_roll_deg, roll_violation, lethal_waypoint_hits…` |
| `summary_table.tex` | Bảng LaTeX cho báo cáo (không commit file .tex) |
| `STATUS.txt` / `pipeline.log` | Tiến độ / log |
| `../history.csv` | 1 dòng/planner/lần chạy, có git commit → so trước/sau |

Định nghĩa: **valid** = planner báo thành công **và** điểm cuối cách goal ≤ dung sai **và** không mẫu nào dọc đường (bước 2.5 cm) rơi vào ô vật cản. `max_roll_deg` đo **thống nhất bằng chân đế 3 điểm** cho mọi planner (`max_roll_reported_deg` là giá trị planner tự báo).

---

## 7. Lưu ý / bẫy đã gặp
- **Không dùng 6 bản đồ group1** (`map_data/000000`–`000005`) để kết luận: chỉ 23% ô có dữ liệu thật, ô trống Z = 0 → vật cản và vách giả.
- 8 frame RELLIS mẫu là **cùng một cảnh** (cách nhau 0.1 s).
- `--workers > 1` nhanh nhưng **thời gian đo bị nhiễu**; số liệu báo cáo dùng `--workers 1`.
- Chạy lại cùng `--name` sẽ **tự resume**; muốn chạy mới thì đổi tên.
- `.gitignore` chặn `*.md`, `*.tex`, `*.pdf`, `logs/`, dữ liệu thô (riêng `HANDOFF.md` được phép commit). Không commit dữ liệu tải về.
- Windows: các script đã tự chuyển console sang UTF-8; nếu chạy qua công cụ khác vẫn lỗi font thì đặt `PYTHONIOENCODING=utf-8`.
- Commit: chuẩn conventional, **1 dòng tiếng Việt**, không thêm dòng co-author.

---

## 8. Nguồn
- RELLIS-3D: https://github.com/unmannedlab/RELLIS-3D · https://arxiv.org/abs/2011.12954
- BenchNav: https://github.com/masafumiendo/benchnav
- GOOSE: https://arxiv.org/abs/2310.16788 (dữ liệu: goose-dataset.de)
- WildScenes: https://csiro-robotics.github.io/WildScenes · https://data.csiro.au/collection/csiro:61541
- TartanDrive 2.0: https://arxiv.org/abs/2402.01913
- Pipeline gốc chế độ jikay: https://github.com/Jikay71/Building-a-2.5D-Dataset-from-RELLIS-3D-for-Path-Planning
