# HANDOFF: Cải thiện thuật toán 2.5D bằng các bộ dữ liệu mới

> Bàn giao: 2026-11-09 · Repo: https://github.com/TueLam06/Path-Planning-2.5D- (nhánh `main`)
> Mục tiêu phiên tiếp theo: **chạy bộ auto-test trên các bộ dữ liệu mới → tìm điểm yếu → sửa thuật toán →
> chạy lại để chứng minh cải thiện**, không làm hỏng những gì đang đúng.

---

## 0. Định nghĩa "xong"
- [ ] Chạy được ít nhất **2 bộ dữ liệu mới** ngoài 8 frame RELLIS mẫu (ưu tiên: BenchNav → RELLIS đầy đủ).
- [ ] Bộ test **phân biệt được thuật toán**: ít nhất 1 planner có valid < 90% ở nhóm khó (hiện tại mọi planner đạt 100% trên jikay8 nên chưa xếp hạng được).
- [ ] Mỗi thay đổi thuật toán có **số liệu trước/sau** trong `logs/testsuite/history.csv`.
- [ ] `false_success = 0` ở mọi lần chạy (đây là test hồi quy bắt buộc).
- [ ] RRT\* trung bình < 3 s/case (hiện khoảng 12 s).

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

> Log chi tiết của 3 lần chạy này nằm ở máy cũ: `Documents/RELLIS 2.5D/4. logs/testsuite/{rellis_full_v1, rellis_full_v3_fixed, jikay8_v1}/`.

---

## 2. Cài đặt
```bash
git clone https://github.com/TueLam06/Path-Planning-2.5D-.git
cd Path-Planning-2.5D-
python -m venv .venv && .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # cần cho bản đồ BenchNav .pt
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

### 4.1 BenchNav: đã có sẵn trong repo, làm ĐẦU TIÊN
- `path_planning_benchmark/map_data/from_benchnav/*.pt`: 100 bản đồ độ cao tổng hợp (nguồn: https://github.com/masafumiendo/benchnav).
- `MapLoader` đọc `tensors["heights"]`, **gán cứng độ phân giải 0.25 m** (`map_loader.py`, nhánh `.pt`), **không có lớp ngữ nghĩa** → chỉ chạy chế độ `geo`.
- Giá trị: địa hình **dốc và gồ ghề** hơn RELLIS → kiểm tra được ràng buộc roll/pitch và khả năng tìm đường vòng, đúng chỗ jikay8 còn thiếu.
- **Kiểm tra trước khi tin số liệu:** đơn vị của `heights` (m?), kích thước lưới, độ phân giải thật của BenchNav. Mở 1 file, in `Z.min()`, `Z.max()`, `Z.shape`; nếu sai thì sửa `resolution` trong `map_loader.py`.
```bash
cd path_planning_benchmark
python generate_testcases.py --maps "map_data/from_benchnav/*.pt" --per-map 6 --out ../logs/benchnav_cases.json
python run_testsuite.py --cases ../logs/benchnav_cases.json --modes geo --workers 4 --name benchnav_base
```
> `generate_testcases.py` chỉ đặt start/goal trên ô đã quan sát khi bản đồ có `observed_mask`; bản đồ `.pt` không có lớp này nên mọi ô free đều hợp lệ.

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

> Số dòng tính theo commit `dd37897`. Mỗi mục: **vấn đề → cách sửa gợi ý → cách kiểm chứng**.

### P1: làm trước
**a. Bộ test quá dễ** · `generate_testcases.py:46` (`CATEGORIES`), `:64` (`generate_for_map`)
- Thêm tag độ khó: `detour_ratio` (đường tham chiếu / khoảng cách thẳng), `min_clearance` (khoảng trống nhỏ nhất dọc đường tham chiếu, dùng EDT trên `occupancy_2d`), `max_slope_on_ref`.
- Sinh case có chủ đích cho từng tag (quota), thêm **case âm** (start/goal khác vùng liên thông → planner phải báo thất bại) và **case semantic-critical** (đường ngắn nhất theo `costmap_geom` đi qua bùn/vũng/bụi, nhưng vẫn có đường an toàn theo `costmap`).
- Kiểm chứng: trên RELLIS dev, ít nhất 1 planner < 90% valid ở nhóm khó; bật/tắt ngữ nghĩa khác nhau rõ ở nhóm semantic-critical.

**b. RRT\* chậm (khoảng 12 s/case)** · `planners/rrt_star_2_5d.py:68` (`_nearest_node` duyệt list O(n)), vòng tìm `near_nodes` trong `plan()` (`:110`) cũng O(n); `_is_edge_valid` (`:88`) gọi lại nhiều lần cho cùng cạnh.
- Dùng mảng numpy cho toạ độ node + `scipy.spatial.cKDTree` (dựng lại định kỳ) hoặc lưới băm; cache kết quả kiểm tra cạnh; dừng sớm khi đã tới goal và cost hội tụ.
- Kiểm chứng: cùng `testcases.json`, valid không giảm, thời gian TB < 3 s.

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

**f. T-Hybrid A\* chỉ dùng pháp tuyến 1 ô và chỉ đi tiến** · `planners/thybrid_a_star.py:99` (`directions = [1]`), kiểm tra roll trong `plan()` (`:121`) dùng `project_pose_so3`
- Thử primitive lùi (direction = −1, đã có hệ số phạt 1.2); kiểm tra roll bằng chân đế cho nhất quán với bộ đo.

**g. Dung sai goal khác nhau giữa planner** · T-Hybrid 2.5 m, RRT\* 2.5 m, Field D\* 1.2 m (dừng khi ≤ 1.2 m), LazyPRM\* coi goal là 1 node của roadmap (phải tới đúng goal)
- Đưa thành tham số chung truyền từ `benchnav/testsuite.py:49` (`build_planner`), bằng `--goal-tol` của runner.

### P3
**h. `is_segment_free` tốn thời gian** · `map_utils/hybrid_map.py:173` (lấy mẫu mỗi 2.5 cm). Field D\* chậm từ 0.13 lên 0.8 s sau khi thêm kiểm tra này.
- Planner lưới: tính trước danh sách ô bị cắt cho 16 offset cố định (supercover), tra bảng thay vì lấy mẫu.

**i. LazyPRM\* chỉ đạt 45% trên bộ 40 case cũ** · `planners/lazy_prm_star_2_5d.py:19` (`num_samples`, `connection_radius`), `:46` (`_eval_edge`)
- Thử lấy mẫu tập trung gần vùng hẹp (bridge sampling / Gaussian sampling), tăng mẫu thích ứng khi không tìm thấy đường.

**j. `clear_around_point` sửa trực tiếp bản đồ** · `map_utils/hybrid_map.py:139`: planner tự xoá vật cản 1.0–1.2 m quanh start/goal. Bộ kiểm chứng dùng 1.2 m (`benchnav/testsuite.py:27`). Nếu đổi bán kính thì đổi cả hai.

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
