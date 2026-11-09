# CLAUDE.md

Benchmark 4 planner 2.5D (T-Hybrid A\*, RRT\*, Field D\*, LazyPRM\*) trên bản đồ 2.5D sinh từ LiDAR off-road (RELLIS-3D).
Việc đang làm: xem `HANDOFF.md`.

## Cấu trúc
- `rellis_25d_generator/`: LiDAR `.bin`/`.label` → bản đồ 2.5D `.npz`
- `path_planning_benchmark/`: `planners/`, `map_utils/`, `benchnav/`, `generate_testcases.py`, `run_testsuite.py`
- `auto_pipeline.py`: chạy toàn bộ một lệnh → `logs/testsuite/<name>/REPORT.md`
- `tools/`: công cụ phụ (ảnh → LiDAR, adapter dữ liệu)

## Git
- Commit theo Conventional Commits, **1 dòng tiếng Việt, ngắn gọn**: `feat: ...`, `fix: ...`, `perf: ...`, `refactor: ...`, `test: ...`, `chore: ...`, `docs: ...`
- **Không** thêm dòng `Co-Authored-By` hay bất kỳ dòng ký tên nào.
- Mỗi commit một thay đổi logic; không gộp sửa lỗi với tính năng.
- **Không commit**: tài liệu, báo cáo LaTeX (`*.tex`), `*.pdf`, file `*.md` (trừ `CLAUDE.md`, `HANDOFF.md`), `logs/`, dữ liệu thô / đầu ra sinh lại được, file rác hệ điều hành (`.DS_Store`, `*:Zone.Identifier`, `__pycache__`).
- Chỉ commit/push khi được yêu cầu.

## Code
- Comment **ngắn**: chỉ ghi khối code làm gì khi không tự hiển nhiên, hoặc `TODO:` / `FIXME:`. Không giải thích dài dòng, không lặp lại điều code đã nói.
- Docstring tối đa vài dòng: mục đích, tham số quan trọng, giá trị trả về.
- Đường dẫn tương đối theo vị trí file (`os.path.dirname(__file__)`), log ghi vào `logs/` ở gốc repo.
- Script có `print` tiếng Việt phải bật UTF-8 cho stdout (console Windows mặc định cp1252).

## Kiểm thử thuật toán
- Sau mỗi thay đổi planner: chạy smoke và bộ `testsuite/testcases.json`. **`false_success` phải bằng 0**, valid không được giảm.
- So trước/sau bằng cùng `testcases.json` + cùng seed, đọc `logs/testsuite/history.csv`.
- Chỉ chỉnh tham số trên tập dev; tập test chỉ chạy khi đã chốt.
- Số liệu báo cáo chạy `--workers 1` (song song làm sai thời gian đo).
