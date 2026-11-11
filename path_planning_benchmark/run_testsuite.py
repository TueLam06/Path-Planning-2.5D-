#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chạy hàng loạt 4 planner trên testsuite/testcases.json và kiểm chứng độc lập kết quả.

valid = báo thành công + tới đích (<= --goal-tol) + không cắt ô vật cản.
Đầu ra: logs/testsuite/<name>/ (results.json|csv, summary.json|md, summary_table.tex).

Ví dụ:
  python run_testsuite.py --modes sem geo --workers 1 --name rellis_v1
  python run_testsuite.py --name rellis_v1 --resume
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchnav.metrics import PathEvaluator  # noqa: E402
from benchnav.testsuite import (  # noqa: E402
    GOAL_TOL, PLANNER_NAMES, ROLL_LIMIT_DEG, build_hmap, build_planner, load_map_data, verify_path,
)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_LOG_ROOT = os.path.abspath(os.path.join(HERE, "..", "logs", "testsuite"))
_MAP_CACHE = {}


def _get_map(map_path):
    if map_path not in _MAP_CACHE:
        _MAP_CACHE[map_path] = load_map_data(map_path)
    return _MAP_CACHE[map_path]


def run_job(job):
    """Chạy 1 job (case, planner, mode, run)."""
    case, planner_name, mode, run_idx, seed, goal_tol = job
    use_sem = mode == "sem"
    data = _get_map(os.path.join(HERE, case["map_path"]))
    start, goal = tuple(case["start"]), tuple(case["goal"])

    np.random.seed(seed)
    hm = build_hmap(data, use_sem)
    planner = build_planner(planner_name, hm)
    err = None
    t0 = time.perf_counter()
    try:
        path, success = planner.plan(start, goal, goal_tolerance=goal_tol)
    except Exception as exc:
        path, success, err = [], False, f"{type(exc).__name__}: {exc}"
    t_plan = time.perf_counter() - t0

    m = PathEvaluator.evaluate_path(path, t_plan, success)
    v = verify_path(path, start, goal, data, use_sem, goal_tol)
    collision_free = v["collision_samples"] == 0
    valid = bool(success) and v["reached"] and collision_free
    ref = case.get("ref_len_3d_anyangle_m") or case["ref_len_3d_m"]
    rec = {
        "case_id": case["id"], "map": case["map"], "category": case["category"],
        "planner": planner_name, "mode": mode, "run": run_idx, "seed": seed,
        "reported_success": bool(success), "reached": v["reached"],
        "collision_free": collision_free, "valid": valid,
        "false_success": bool(success) and not valid,
        "planning_time_sec": m["planning_time_sec"],
        "path_length_3d": m["path_length_3d"], "ref_len_3d": ref,
        "optimality_ratio": (m["path_length_3d"] / ref) if valid and ref > 0 else None,
        "max_roll_deg": v["max_roll_fp_deg"], "max_pitch_deg": v["max_pitch_fp_deg"],
        "max_roll_reported_deg": m["max_roll_deg"], "max_pitch_reported_deg": m["max_pitch_deg"],
        "mean_traversability": m["mean_traversability"],
        "roll_violation": (bool(v["max_roll_fp_deg"] > ROLL_LIMIT_DEG)
                           if v["max_roll_fp_deg"] is not None else None),
        "goal_error_m": v["goal_error_m"], "collision_samples": v["collision_samples"],
        "lethal_waypoint_hits": v["lethal_waypoint_hits"],
        "soft_hazard_hits": v["soft_hazard_hits"], "mean_costmap": v["mean_costmap"],
        "n_waypoints": len(path) if path else 0, "error": err,
    }
    return rec


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def summarize(records, categories):
    out = []
    for mode in sorted({r["mode"] for r in records}, reverse=True):
        for p in PLANNER_NAMES:
            rs = [r for r in records if r["mode"] == mode and r["planner"] == p]
            if not rs:
                continue
            ok = [r for r in rs if r["valid"]]
            row = {
                "mode": mode, "planner": p, "n": len(rs),
                "valid_rate": len(ok) / len(rs),
                "reported_success_rate": sum(r["reported_success"] for r in rs) / len(rs),
                "false_success": sum(r["false_success"] for r in rs),
                "crashes": sum(r["error"] is not None for r in rs),
                "mean_time_s": _mean([r["planning_time_sec"] for r in rs]),
                "median_time_s": _median([r["planning_time_sec"] for r in rs]),
                "mean_opt_ratio": _mean([r["optimality_ratio"] for r in ok]),
                "mean_max_roll_deg": _mean([r["max_roll_deg"] for r in ok]),
                "roll_violation_rate": (sum(bool(r["roll_violation"]) for r in ok) / len(ok)) if ok else None,
                "mean_lethal_hits": _mean([r["lethal_waypoint_hits"] for r in ok]),
                "mean_costmap": _mean([r["mean_costmap"] for r in ok]),
            }
            for c in categories:
                rc = [r for r in rs if r["category"] == c]
                row[f"valid_rate_{c}"] = (sum(r["valid"] for r in rc) / len(rc)) if rc else None
            out.append(row)
    return out


def _f(x, fmt="{:.2f}", none="–"):
    return none if x is None else fmt.format(x)


def write_reports(out_dir, records, summary, categories, meta):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": records}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary}, f, ensure_ascii=False, indent=2)

    cat_hdr = " | ".join(f"Valid {c}" for c in categories)
    lines = [
        f"# Test suite: {meta['suite']}",
        "",
        f"- Test case: {meta['n_cases']} trên {meta['n_maps']} bản đồ | chế độ: {', '.join(meta['modes'])} "
        f"| {meta['runs']} lần/case | goal_tol {meta['goal_tol']} m | workers {meta['workers']}",
        "- **Valid** = planner báo thành công VÀ điểm cuối tới đích VÀ không cắt ô vật cản (kiểm chứng độc lập).",
        "- **Opt. ratio** = độ dài 3D / đường tham chiếu Dijkstra trên lưới (chỉ tính case valid).",
        "",
        f"| Mode | Planner | N | Valid | Báo success | False success | Time TB (s) | Opt. ratio | Max roll TB (°) | Vượt roll | {cat_hdr} |",
        "|" + "---|" * (10 + len(categories)),
    ]
    for r in summary:
        cats = " | ".join(_f(r[f"valid_rate_{c}"], "{:.0%}") for c in categories)
        lines.append(
            f"| {r['mode']} | {r['planner']} | {r['n']} | {r['valid_rate']:.0%} | "
            f"{r['reported_success_rate']:.0%} | {r['false_success']} | {_f(r['mean_time_s'], '{:.3f}')} | "
            f"{_f(r['mean_opt_ratio'])} | {_f(r['mean_max_roll_deg'], '{:.1f}')} | "
            f"{_f(r['roll_violation_rate'], '{:.0%}')} | {cats} |"
        )
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    tex = [
        "% Sinh tự động bởi run_testsuite.py",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Kết quả bộ test tự động ({meta['n_cases']} cặp start/goal, {meta['n_maps']} bản đồ RELLIS-3D 2.5D).}}",
        "\\label{tab:testsuite}",
        "\\begin{tabular}{llrrrrr}",
        "\\hline",
        "Chế độ & Thuật toán & Valid (\\%) & False succ. & Thời gian (s) & Opt. ratio & Max roll ($^\\circ$) \\\\",
        "\\hline",
    ]
    for r in summary:
        tex.append(
            f"{'Sem+Geo' if r['mode'] == 'sem' else 'Geo-only'} & {r['planner'].replace('*', '$^*$')} & "
            f"{100 * r['valid_rate']:.1f} & {r['false_success']} & {_f(r['mean_time_s'], '{:.3f}')} & "
            f"{_f(r['mean_opt_ratio'])} & {_f(r['mean_max_roll_deg'], '{:.1f}')} \\\\"
        )
    tex += ["\\hline", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(out_dir, "summary_table.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Chạy hàng loạt planner 2.5D trên bộ test case")
    ap.add_argument("--cases", default=os.path.join("testsuite", "testcases.json"))
    ap.add_argument("--planners", nargs="+", default=PLANNER_NAMES, choices=PLANNER_NAMES)
    ap.add_argument("--modes", nargs="+", default=["sem"], choices=["sem", "geo"],
                    help="sem = dung hợp ngữ nghĩa, geo = chỉ hình học (ablation)")
    ap.add_argument("--maps", nargs="+", default=None, help="Chỉ chạy các bản đồ này")
    ap.add_argument("--categories", nargs="+", default=None, help="Chỉ chạy nhóm short/medium/long")
    ap.add_argument("--runs", type=int, default=1, help="Số lần chạy mỗi case (seed khác nhau)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--goal-tol", type=float, default=GOAL_TOL,
                    help="Dung sai goal (m), dùng chung cho planner và bộ kiểm chứng")
    ap.add_argument("--workers", type=int, default=1,
                    help="Số tiến trình song song. >1 nhanh hơn nhưng thời gian đo bị nhiễu do tranh CPU")
    ap.add_argument("--name", default=None, help="Tên thư mục kết quả (mặc định theo thời gian)")
    ap.add_argument("--resume", action="store_true",
                    help="Tiếp tục run dở (cùng --name): bỏ qua job đã có trong results.partial.jsonl")
    ap.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    args = ap.parse_args()

    cases_path = args.cases if os.path.isabs(args.cases) else os.path.join(HERE, args.cases)
    with open(cases_path, encoding="utf-8") as f:
        suite = json.load(f)
    cases = suite["cases"]
    if args.maps:
        cases = [c for c in cases if c["map"] in args.maps]
    if args.categories:
        cases = [c for c in cases if c["category"] in args.categories]
    if not cases:
        sys.exit("Không còn test case nào sau khi lọc.")
    categories = [c for c in suite.get("categories", {"short": 0, "medium": 0, "long": 0})
                  if any(x["category"] == c for x in cases)]

    jobs = []
    for ci, case in enumerate(cases):
        for mode in args.modes:
            for p in args.planners:
                for r in range(args.runs):
                    jobs.append((case, p, mode, r + 1, args.seed + 1000 * ci + r, args.goal_tol))

    if args.resume and not args.name:
        sys.exit("--resume cần --name của run đang dở.")
    name = args.name or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.log_root, name)
    os.makedirs(out_dir, exist_ok=True)
    partial_path = os.path.join(out_dir, "results.partial.jsonl")

    # Ghi từng kết quả để có thể resume
    records = []
    if args.resume and os.path.isfile(partial_path):
        with open(partial_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        # Ghi lại file, bỏ dòng hỏng
        with open(partial_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        done = {(r["case_id"], r["planner"], r["mode"], r["run"]) for r in records}
        jobs = [j for j in jobs if (j[0]["id"], j[1], j[2], j[3]) not in done]
        print(f"[Resume] Đã có {len(records)} kết quả, còn {len(jobs)} job.")
    elif os.path.isfile(partial_path):
        os.remove(partial_path)

    print(f"[Test suite] {len(cases)} case × {len(args.planners)} planner × {len(args.modes)} chế độ "
          f"× {args.runs} lần = {len(jobs) + len(records)} job (cần chạy {len(jobs)}) | workers={args.workers}")
    if args.workers > 1:
        print("  (Lưu ý: chạy song song làm thời gian đo kém chính xác; dùng --workers 1 cho số liệu báo cáo)")

    t0 = time.time()
    n_total = len(jobs)
    partial = open(partial_path, "a", encoding="utf-8")

    def _done(rec, k):
        records.append(rec)
        partial.write(json.dumps(rec, ensure_ascii=False) + "\n")
        partial.flush()
        flag = "OK " if rec["valid"] else ("FS!" if rec["false_success"] else "-- ")
        eta = (time.time() - t0) / k * (n_total - k)
        print(f"  [{k:>5}/{n_total}] {flag} {rec['case_id']:<30} {rec['mode']:<3} {rec['planner']:<14} "
              f"{rec['planning_time_sec']:6.2f}s  L={rec['path_length_3d']:6.2f}m "
              f"(ref {rec['ref_len_3d']:.2f})  ETA {eta / 60:5.1f} phút"
              + (f"  ERR {rec['error']}" if rec["error"] else ""),
              flush=True)

    try:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(run_job, j) for j in jobs]
                for k, fu in enumerate(as_completed(futs), 1):
                    _done(fu.result(), k)
        else:
            for k, j in enumerate(jobs, 1):
                _done(run_job(j), k)
    finally:
        partial.close()

    if not records:
        sys.exit("Không có kết quả nào.")

    order = {c["id"]: i for i, c in enumerate(cases)}
    records.sort(key=lambda r: (order[r["case_id"]], r["mode"], PLANNER_NAMES.index(r["planner"]), r["run"]))
    summary = summarize(records, categories)
    meta = {
        "suite": name, "cases_file": os.path.relpath(cases_path, HERE).replace("\\", "/"),
        "n_cases": len(cases), "n_maps": len({c["map"] for c in cases}),
        "modes": args.modes, "planners": args.planners, "runs": args.runs, "seed": args.seed,
        "goal_tol": args.goal_tol, "workers": args.workers,
        "elapsed_sec": round(time.time() - t0, 1), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    table = write_reports(out_dir, records, summary, categories, meta)
    print("\n" + table)
    print(f"\n[OK] Kết quả lưu tại: {out_dir}  ({meta['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
