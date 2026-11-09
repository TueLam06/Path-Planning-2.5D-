#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_pipeline.py - Chạy TỰ ĐỘNG toàn bộ quy trình đánh giá, một lệnh từ đầu tới cuối:

  [1] maps   : LiDAR RELLIS (.bin/.label, thư mục hoặc .zip) -> bản đồ 2.5D .npz
               (rellis_25d_generator, lấy 1 frame mỗi --stride frame)
  [2] cases  : sinh test case start/goal (generate_testcases.py)
  [3] run    : chạy 4 planner (run_testsuite.py, ghi từng kết quả, tự resume)
  [4] report : ghi REPORT.md (kết quả cần cho báo cáo) + nối 1 dòng/planner vào history.csv

Mọi bước đều IDEMPOTENT: chạy lại cùng --name sẽ bỏ qua phần đã xong, nên máy tắt / mất điện
giữa chừng chỉ cần chạy lại đúng lệnh cũ.

Kết quả: "logs/testsuite/<name>/"
  REPORT.md            <- file tổng hợp để đọc / dán vào báo cáo
  STATUS.txt           <- trạng thái hiện tại (running / done / failed + bước)
  pipeline.log         <- toàn bộ log
  summary_table.tex, summary.md, results.csv, results.json, testcases.json
"logs/testsuite/history.csv" <- lịch sử mọi lần chạy (so sánh giữa các phiên bản code)

Ví dụ:
  # Thử nhanh trên bộ mẫu 8 frame (tự tải nếu chưa có)
  python auto_pipeline.py --name sample_quick --download-sample --seqs 00000 --stride 1 --per-map 3 --workers 4

  # Bộ đầy đủ (đã đặt các file .zip RELLIS vào rellis_25d_generator/raw_data/ hoặc --data-dir)
  python auto_pipeline.py --name rellis_full --seqs 00000 00001 00002 00003 00004 --stride 50 --workers 6
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
GEN_DIR = os.path.join(CODE_DIR, "rellis_25d_generator")
BENCH_DIR = os.path.join(CODE_DIR, "path_planning_benchmark")
LOG_ROOT = os.path.abspath(os.path.join(CODE_DIR, "logs", "testsuite"))


# ----------------------------------------------------------------------------
# Log ra cả màn hình và file
# ----------------------------------------------------------------------------
class Logger:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg=""):
        line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def raw(self, line):
        print(line, end="", flush=True)
        self.f.write(line)
        self.f.flush()


def set_status(run_dir, state, step, extra=""):
    with open(os.path.join(run_dir, "STATUS.txt"), "w", encoding="utf-8") as f:
        f.write(f"state: {state}\nstep: {step}\nupdated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}\n{extra}\n")


def run_cmd(cmd, log, cwd):
    """Chạy lệnh con, stream log ra màn hình + file. Ném lỗi nếu exit code != 0."""
    log("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", env=env)
    for line in p.stdout:
        log.raw(line)
    if p.wait() != 0:
        raise RuntimeError(f"Lệnh thất bại (exit {p.returncode}): {cmd[1] if len(cmd) > 1 else cmd}")


def git_info():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=CODE_DIR,
                                      text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "."], cwd=CODE_DIR,
                                        text=True, stderr=subprocess.DEVNULL).strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------
# [1] Sinh bản đồ
# ----------------------------------------------------------------------------
def step_maps(args, maps_dir, log):
    sys.path.insert(0, GEN_DIR)
    from config import RellisConfig          # noqa: E402  (module của rellis_25d_generator)
    from data_loader import RellisDataLoader, download_sample  # noqa: E402
    from pipeline import process_single_frame  # noqa: E402

    if args.download_sample:
        download_sample(raw_dir=os.path.join(GEN_DIR, "raw_data"), extract=True)

    cfg = RellisConfig()
    loader = RellisDataLoader(workspace_dir=GEN_DIR, data_dir=args.data_dir)
    os.makedirs(maps_dir, exist_ok=True)

    plan = []
    for seq in args.seqs:
        try:
            ids = loader.get_all_frame_ids(seq)
        except Exception as exc:
            log(f"  [BỎ QUA] sequence {seq}: {exc}")
            continue
        picked = ids[:: args.stride]
        if args.max_frames_per_seq:
            picked = picked[: args.max_frames_per_seq]
        log(f"  sequence {seq}: {len(ids)} frame -> lấy {len(picked)} (stride {args.stride})")
        plan += [(seq, f) for f in picked]
    if not plan:
        raise RuntimeError("Không tìm thấy frame nào. Kiểm tra --data-dir / raw_data/.")

    def work(item):
        seq, frame = item
        return process_single_frame(loader, seq, frame, __import__("pathlib").Path(maps_dir), cfg,
                                    mode=args.method, save_preview=False, skip_existing=True,
                                    verbose=False)

    t0, n_new, n_skip, n_err = time.time(), 0, 0, 0
    map_files = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(work, it): it for it in plan}
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                res = fu.result()
                map_files.append(str(res["npz_path"]))
                n_skip += bool(res.get("skipped"))
                n_new += not res.get("skipped")
            except Exception as exc:
                n_err += 1
                log(f"  [LỖI] frame {futs[fu]}: {exc}")
            if k % 25 == 0 or k == len(plan):
                log(f"  bản đồ {k}/{len(plan)} (mới {n_new}, có sẵn {n_skip}, lỗi {n_err}) "
                    f"{time.time() - t0:.0f}s")
    # Chỉ dùng đúng các bản đồ của lần chạy này (thư mục bản đồ dùng chung có thể chứa bản đồ cũ)
    with open(maps_list_path(args), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(map_files)) + "\n")
    if args.method == "group1" and len(set(os.path.basename(p) for p in map_files)) < len(map_files):
        log("  [CẢNH BÁO] group1 đặt tên file theo frame, các sequence khác nhau sẽ ghi đè nhau.")
    return {"frames_planned": len(plan), "maps_new": n_new, "maps_existing": n_skip, "map_errors": n_err}


def maps_list_path(args):
    return os.path.join(LOG_ROOT, args.name, "maps_list.txt")


# ----------------------------------------------------------------------------
# [4] Báo cáo
# ----------------------------------------------------------------------------
def _pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def _num(x, fmt="{:.2f}"):
    return "–" if x is None else fmt.format(x)


def step_report(args, run_dir, stage_info, t_start, log):
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        summ = json.load(f)
    with open(os.path.join(run_dir, "results.json"), encoding="utf-8") as f:
        results = json.load(f)["results"]
    with open(os.path.join(run_dir, "testcases.json"), encoding="utf-8") as f:
        tc = json.load(f)

    meta, rows = summ["meta"], summ["summary"]
    cats = list(tc.get("categories", {}).keys())
    cases = tc["cases"]
    git = git_info()

    # Case khó nhất: nhiều planner thất bại nhất
    by_case = {}
    for r in results:
        by_case.setdefault(r["case_id"], []).append(r)
    hardest = sorted(by_case.items(), key=lambda kv: -sum(not x["valid"] for x in kv[1]))[:10]
    false_succ = [r for r in results if r["false_success"]]

    maps = tc.get("maps", [])
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    L = []
    L += [f"# Báo cáo auto-test: `{args.name}`", ""]
    L += ["## 1. Thông tin chạy", "",
          "| Mục | Giá trị |", "|---|---|",
          f"| Bắt đầu / kết thúc | {dt.datetime.fromtimestamp(t_start):%Y-%m-%d %H:%M} → {dt.datetime.now():%Y-%m-%d %H:%M} |",
          f"| Tổng thời gian | {(time.time() - t_start) / 3600:.2f} giờ |",
          f"| Git commit | `{git}` |",
          f"| Máy | {platform.platform()} · {os.cpu_count()} CPU · Python {platform.python_version()} |",
          f"| Phương án bản đồ | `{args.method}` · stride {args.stride} · sequences {', '.join(args.seqs)} |",
          f"| Planner / chế độ | {', '.join(meta['planners'])} / {', '.join(meta['modes'])} |",
          f"| Lần chạy mỗi case · seed · dung sai goal · workers | {meta['runs']} · {meta['seed']} · {meta['goal_tol']} m · {meta['workers']} |",
          ""]
    for k, v in stage_info.items():
        L.append(f"- Bước **{k}**: {v}")
    L += ["", "## 2. Dữ liệu", "",
          f"- Bản đồ: **{len(maps)}** · test case: **{len(cases)}** "
          f"({', '.join(f'{c}: {sum(x['category'] == c for x in cases)}' for c in cats)})",
          f"- Tỷ lệ ô đi được TB: {_pct(mean([m['free_ratio'] for m in maps]))} · "
          f"ô đo thật TB: {_pct(mean([m['observed_ratio'] for m in maps if m.get('observed_ratio') is not None]))}",
          f"- Bản đồ thiếu case (không đủ quota nhóm khó): "
          f"{sum(1 for m in maps if m.get('missing'))}/{len(maps)}", ""]
    L += ["## 3. Kết quả chính", "",
          "| Chế độ | Planner | N | Valid | False success | Thời gian TB (s) | Median (s) | Opt. ratio | Max roll TB (°) | Vượt roll | "
          + " | ".join(f"Valid {c}" for c in cats) + " |",
          "|" + "---|" * (10 + len(cats))]
    for r in rows:
        L.append(f"| {r['mode']} | {r['planner']} | {r['n']} | **{_pct(r['valid_rate'])}** | {r['false_success']} | "
                 f"{_num(r['mean_time_s'], '{:.3f}')} | {_num(r['median_time_s'], '{:.3f}')} | {_num(r['mean_opt_ratio'])} | "
                 f"{_num(r['mean_max_roll_deg'], '{:.1f}')} | {_pct(r['roll_violation_rate'])} | "
                 + " | ".join(_pct(r.get(f'valid_rate_{c}')) for c in cats) + " |")
    L += ["", "Valid = báo thành công **và** tới đích (≤ dung sai) **và** không cắt ô vật cản (kiểm chứng độc lập). "
          "Max roll đo thống nhất bằng chân đế 3 điểm cho mọi planner.", ""]
    L += ["## 4. Cảnh báo", ""]
    L.append(f"- ⚠️ **{len(false_succ)} lần báo success sai** — planner có lỗi kiểm tra va chạm!" if false_succ
             else "- ✅ Không có lần báo success sai.")
    all_valid = all(r["valid_rate"] >= 0.99 for r in rows)
    if all_valid:
        L.append("- ⚠️ Mọi planner đạt ≥ 99%: bộ test **quá dễ**, không phân biệt được planner "
                 "(cần case khó hơn / bản đồ đa dạng hơn).")
    L += ["", "## 5. 10 case khó nhất", "", "| Case | Nhóm | Ref (m) | Số job thất bại | Planner thất bại |", "|---|---|---|---|---|"]
    if not hardest or all(x["valid"] for x in hardest[0][1]):
        L.append("| (không có case nào thất bại) | | | | |")
    for cid, rs in hardest:
        fails = [f"{x['planner']} ({x['mode']})" for x in rs if not x["valid"]]
        if not fails:
            break
        L.append(f"| {cid} | {rs[0]['category']} | {rs[0]['ref_len_3d']:.1f} | {len(fails)}/{len(rs)} | {', '.join(fails)} |")
    L += ["", "## 6. File", "",
          "- `summary_table.tex`: bảng LaTeX dán vào báo cáo",
          "- `results.csv`: từng lần chạy (mở bằng Excel)",
          "- `testcases.json`: bộ test case dùng trong lần chạy này (tái lập được)",
          "- `pipeline.log`: log đầy đủ", ""]
    with open(os.path.join(run_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    hist = os.path.join(LOG_ROOT, "history.csv")
    new = not os.path.isfile(hist)
    with open(hist, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["finished", "run", "git", "method", "n_maps", "n_cases", "mode", "planner", "n",
                        "valid_rate", "false_success", "mean_time_s", "mean_opt_ratio",
                        "mean_max_roll_deg", "roll_violation_rate"])
        for r in rows:
            w.writerow([f"{dt.datetime.now():%Y-%m-%d %H:%M}", args.name, git, args.method, len(maps), len(cases),
                        r["mode"], r["planner"], r["n"], round(r["valid_rate"], 4), r["false_success"],
                        _num(r["mean_time_s"], "{:.4f}"), _num(r["mean_opt_ratio"], "{:.4f}"),
                        _num(r["mean_max_roll_deg"], "{:.2f}"),
                        "" if r["roll_violation_rate"] is None else round(r["roll_violation_rate"], 4)])
    log(f"  REPORT.md + history.csv đã ghi.")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Chạy tự động: LiDAR -> bản đồ 2.5D -> test case -> planner -> báo cáo")
    ap.add_argument("--name", required=True, help="Tên lần chạy (dùng lại tên này để resume)")
    ap.add_argument("--data-dir", default=None, help="Thư mục/zip RELLIS (mặc định: rellis_25d_generator/raw_data)")
    ap.add_argument("--download-sample", action="store_true", help="Tải bộ mẫu 8 frame (24 MB) nếu chưa có")
    ap.add_argument("--seqs", nargs="+", default=["00000", "00001", "00002", "00003", "00004"])
    ap.add_argument("--stride", type=int, default=50, help="Lấy 1 frame mỗi N frame (50 = 5 giây)")
    ap.add_argument("--max-frames-per-seq", type=int, default=None)
    ap.add_argument("--method", default="jikay", choices=["jikay", "group1"])
    ap.add_argument("--maps-dir", default=None,
                    help="Nơi lưu bản đồ (mặc định rellis_25d_generator/output_auto/<method>, dùng chung giữa các run)")
    ap.add_argument("--per-map", type=int, default=9)
    ap.add_argument("--case-seed", type=int, default=2026)
    ap.add_argument("--modes", nargs="+", default=["sem", "geo"])
    ap.add_argument("--planners", nargs="+", default=None)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1, help="Song song (nhanh) / 1 = thời gian đo chính xác")
    ap.add_argument("--goal-tol", type=float, default=2.5)
    ap.add_argument("--skip", nargs="*", default=[], choices=["maps", "cases", "run"],
                    help="Bỏ qua bước (vd. đã có bản đồ)")
    ap.add_argument("--regen-cases", action="store_true", help="Sinh lại test case dù đã có")
    args = ap.parse_args()

    run_dir = os.path.join(LOG_ROOT, args.name)
    os.makedirs(run_dir, exist_ok=True)
    log = Logger(os.path.join(run_dir, "pipeline.log"))
    maps_dir = args.maps_dir or os.path.join(GEN_DIR, "output_auto", args.method)
    cases_path = os.path.join(run_dir, "testcases.json")
    py = sys.executable
    t_start = time.time()
    info = {}
    step = "init"
    log(f"===== AUTO PIPELINE '{args.name}' | bản đồ: {maps_dir}")
    with open(os.path.join(run_dir, "pipeline_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    try:
        step = "maps"
        set_status(run_dir, "running", step)
        if "maps" not in args.skip:
            log("[1/4] Sinh bản đồ 2.5D")
            t = time.time()
            s = step_maps(args, maps_dir, log)
            info["maps"] = (f"{s['frames_planned']} frame → mới {s['maps_new']}, có sẵn {s['maps_existing']}, "
                            f"lỗi {s['map_errors']} ({time.time() - t:.0f} s)")

        step = "cases"
        set_status(run_dir, "running", step)
        if "cases" not in args.skip and (args.regen_cases or not os.path.isfile(cases_path)):
            log("[2/4] Sinh test case")
            t = time.time()
            src = (["--maps-list", maps_list_path(args)] if os.path.isfile(maps_list_path(args))
                   else ["--maps", os.path.join(maps_dir, "*.npz")])
            run_cmd([py, "generate_testcases.py", *src,
                     "--per-map", str(args.per_map), "--seed", str(args.case_seed), "--out", cases_path],
                    log, BENCH_DIR)
            info["cases"] = f"{time.time() - t:.0f} s"
        elif os.path.isfile(cases_path):
            log("[2/4] Dùng lại testcases.json đã có")
            info["cases"] = "dùng lại file có sẵn"

        step = "run"
        set_status(run_dir, "running", step, "Tiến độ: xem pipeline.log hoặc đếm dòng results.partial.jsonl")
        if "run" not in args.skip:
            log("[3/4] Chạy planner (tự resume nếu đã chạy dở)")
            t = time.time()
            cmd = [py, "run_testsuite.py", "--cases", cases_path, "--name", args.name, "--resume",
                   "--log-root", LOG_ROOT, "--modes", *args.modes, "--runs", str(args.runs),
                   "--workers", str(args.workers), "--goal-tol", str(args.goal_tol)]
            if args.planners:
                cmd += ["--planners", *args.planners]
            run_cmd(cmd, log, BENCH_DIR)
            info["run"] = f"{(time.time() - t) / 60:.1f} phút (phiên này)"

        step = "report"
        set_status(run_dir, "running", step)
        log("[4/4] Ghi báo cáo")
        step_report(args, run_dir, info, t_start, log)
        set_status(run_dir, "done", step, f"Báo cáo: {os.path.join(run_dir, 'REPORT.md')}")
        log(f"===== XONG sau {(time.time() - t_start) / 60:.1f} phút -> {os.path.join(run_dir, 'REPORT.md')}")
    except KeyboardInterrupt:
        set_status(run_dir, "stopped", step, "Chạy lại đúng lệnh cũ để tiếp tục.")
        log("Đã dừng bởi người dùng. Chạy lại đúng lệnh để tiếp tục.")
        sys.exit(130)
    except Exception as exc:
        set_status(run_dir, "failed", step, f"{type(exc).__name__}: {exc}")
        log(f"LỖI ở bước {step}: {exc}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
