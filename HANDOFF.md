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
| 2 | Chốt định nghĩa benchmark: dung sai goal chung, kích thước xe Warthog | P1-c, P2-g | ⏭ tiếp theo |
| 3 | Bộ test khó: địa hình dốc tự sinh, tag độ khó, case âm; BenchNav `t_classes` → costmap | P1-a, §4.1, §4.1b | |
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
- Nguồn: https://github.com/unmannedlab/RELLIS-3D (mục *LiDAR Download*):
  - *Ouster LiDAR SemanticKITTI Format* (14 GB) → `Rellis_3D_os1_cloud_node_kitti_bin.zip`
  - *Ouster LiDAR Annotation SemanticKITTI Format* (174 MB) → `Rellis_3D_os1_cloud_node_semantickitti_label_id_20210614.zip`
  - (tuỳ chọn) *Ouster LiDAR Scan Poses files* → cho phương án ghép nhiều frame
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

**c. Kích thước xe sai** · `map_utils/hybrid_map.py:34` (`robot_length=0.7, robot_width=0.5`), `rellis_25d_generator/config.py:122-126` (hộp lọc thân xe "Polaris + người lái" 3.6 × 2.6 m)
- RELLIS-3D thu bằng **Clearpath Warthog không người lái** (khoảng 1.5 × 1.4 m, **cần tra thông số chính thức** của Clearpath). Chân đế nhỏ làm roll đo được nhiễu; hộp lọc quá to xoá mất mặt đất thật quanh xe.
- Kiểm chứng: % vượt roll giảm; bản đồ quanh gốc không còn lỗ do lọc nhầm (xem dashboard).

### P2
**d. Độ dốc tính trên Z chưa làm mượt** · `map_utils/hybrid_map.py:79-106` (`_compute_map_layers`: `slope > rho_max`, `roughness > h_max=0.15` → vật cản)
- Bản đồ jikay lấp ô trống bằng EDT nên có bậc thang → bị coi là vách. Ví dụ: bản đồ mẫu `samples/00000_000500` có 72% ô thành vật cản.
- Gợi ý: dùng lớp `slope` đã làm mượt trong `.npz` nếu có, hoặc làm mượt Gaussian giống generator; tính roughness chỉ trên ô `observed_mask`.
- Kiểm chứng: % ô vật cản giảm, valid tăng, `false_success` vẫn bằng 0.

**e. Field D\* không kiểm tra roll; tên chưa đúng bản chất** · `planners/field_d_star_2_5d.py:18` (`_cost`)
- Thêm kiểm tra roll/pitch bằng `project_footprint_3point` theo hướng bước di chuyển.
- Đây thực chất là A\* trên lưới 16 hướng (không nội suy chi phí qua cạnh ô, không replanning). Hoặc hiện thực nội suy Field D\* thật, hoặc đổi tên trong báo cáo.

**f. T-Hybrid A\* chỉ dùng pháp tuyến 1 ô và chỉ đi tiến** · `planners/thybrid_a_star.py:102` (`directions = [1]`), kiểm tra roll trong `plan()` (`:121`) dùng `project_pose_so3`
- Thử primitive lùi (direction = −1, đã có hệ số phạt 1.2); kiểm tra roll bằng chân đế cho nhất quán với bộ đo.

**g. Dung sai goal khác nhau giữa planner** · T-Hybrid 2.5 m, RRT\* 2.5 m, Field D\* 1.2 m (dừng khi ≤ 1.2 m), LazyPRM\* coi goal là 1 node của roadmap (phải tới đúng goal)
- Đưa thành tham số chung truyền từ `benchnav/testsuite.py:41` (`build_planner`), bằng `--goal-tol` của runner.

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
