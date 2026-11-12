#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vẽ kết quả một lần chạy run_testsuite.py (cần --save-paths) và tạo trang HTML để duyệt.

Mỗi (case, chế độ) -> 1 ảnh 2x2, mỗi ô 1 planner trên cùng bản đồ:
  nền xám = độ cao (càng tối càng cao), ô đen = vật cản, đường xanh = ĐẠT, đỏ nét đứt = HỎNG,
  × đỏ = điểm cắt vật cản, chấm vàng = waypoint vượt giới hạn lật, S/G = start/goal (vòng tròn = dung sai).

Ví dụ:
  python run_testsuite.py --cases testsuite/testcases_synthetic.json --modes geo --save-paths --name synth_vis
  python plot_results.py synth_vis                 # toàn bộ case
  python plot_results.py synth_vis --only-failures # chỉ case có ít nhất 1 planner hỏng
  -> logs/testsuite/synth_vis/viewer/index.html
"""
import argparse
import html
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchnav.testsuite import PLANNER_NAMES, build_hmap, load_map_data  # noqa: E402
from run_testsuite import DEFAULT_LOG_ROOT, path_file_name  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Màu theo palette tham chiếu (dataviz): xanh = series 1, đỏ/vàng = trạng thái critical/warning
C_OK = "#2a78d6"
C_FAIL = "#d03b3b"
C_WARN = "#fab219"
INK = "#0b0b0b"
INK_2 = "#52514e"
OBST = "#2b2b2a"


def load_run(run_dir):
    with open(os.path.join(run_dir, "results.json"), encoding="utf-8") as f:
        res = json.load(f)
    cases_file = res["meta"]["cases_file"]
    cases_path = cases_file if os.path.isabs(cases_file) else os.path.join(HERE, cases_file)
    with open(cases_path, encoding="utf-8") as f:
        cases = {c["id"]: c for c in json.load(f)["cases"]}
    return res["results"], cases


def draw_background(ax, hm):
    z = hm.Z.astype(float)
    lo, hi = np.percentile(z, [2, 98])
    ext = [-hm.resolution / 2, (hm.nx - 0.5) * hm.resolution, -hm.resolution / 2, (hm.ny - 0.5) * hm.resolution]
    # Độ cao chỉ dùng dải xám nhạt -> vừa; màu đen dành riêng cho vật cản
    grey = matplotlib.colors.LinearSegmentedColormap.from_list("elev", ["#f4f3ef", "#8d8c86"])
    ax.imshow(z, origin="lower", extent=ext, cmap=grey, vmin=lo, vmax=hi, interpolation="nearest")
    occ = np.ma.masked_where(~hm.occupancy_2d, np.ones_like(z))
    ax.imshow(occ, origin="lower", extent=ext, cmap=matplotlib.colors.ListedColormap([OBST]), alpha=0.9,
              interpolation="nearest")
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7, colors=INK_2)
    for s in ax.spines.values():
        s.set_color("#c3c2b7")


def draw_panel(ax, hm, case, doc, goal_tol):
    draw_background(ax, hm)
    sx, sy = case["start"][:2]
    gx, gy = case["goal"][:2]
    ax.add_patch(plt.Circle((gx, gy), goal_tol, fill=False, ls="--", lw=1, color=INK_2))
    status, color = "KHÔNG CÓ DỮ LIỆU", INK_2
    if doc is not None:
        ok = doc.get("correct", doc["valid"])
        if doc.get("path"):
            p = np.array(doc["path"])
            ax.plot(p[:, 0], p[:, 1], color=C_OK if doc["valid"] else C_FAIL, lw=2,
                    ls="-" if doc["valid"] else (0, (4, 2)), solid_capstyle="round", zorder=4)
            idx = doc.get("roll_violation_idx") or []
            if idx:
                ax.scatter(p[idx, 0], p[idx, 1], s=14, color=C_WARN, edgecolors=INK, linewidths=0.4, zorder=5)
        cp = doc.get("collision_points") or []
        if cp:
            cp = np.array(cp)
            ax.scatter(cp[:, 0], cp[:, 1], marker="x", s=30, color=C_FAIL, linewidths=1.6, zorder=6)
        if not case.get("feasible", True):
            status = "✓ TỪ CHỐI ĐÚNG" if ok else "✗ BÁO SAI CÓ ĐƯỜNG"
        elif doc["valid"]:
            status = f"✓ ĐẠT · {doc['path_length_3d']:.1f} m"
        elif doc.get("reported_success"):
            status = "✗ BÁO THÀNH CÔNG SAI"
        else:
            status = "✗ KHÔNG TÌM ĐƯỢC ĐƯỜNG"
        status += f" · {doc['planning_time_sec']:.2f} s"
        color = C_OK if ok else C_FAIL
    ax.scatter([sx], [sy], s=60, color="white", edgecolors=INK, linewidths=1.5, zorder=7)
    ax.scatter([gx], [gy], s=110, marker="*", color="white", edgecolors=INK, linewidths=1.2, zorder=7)
    ax.annotate("S", (sx, sy), xytext=(5, 5), textcoords="offset points", fontsize=8, color=INK, weight="bold")
    ax.annotate("G", (gx, gy), xytext=(5, 5), textcoords="offset points", fontsize=8, color=INK, weight="bold")
    return status, color


def render_case(case, mode, docs, data, goal_tol, out_png):
    hm = build_hmap(data, mode == "sem")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6), dpi=100)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, planner in zip(axes.ravel(), PLANNER_NAMES):
        status, color = draw_panel(ax, hm, case, docs.get(planner), goal_tol)
        ax.set_title(planner, fontsize=10, color=INK, loc="left", pad=14)
        ax.text(0.0, 1.015, status, transform=ax.transAxes, fontsize=8.5, color=color, va="bottom")
    ref = case.get("ref_len_3d_anyangle_m") or case.get("ref_len_3d_m")
    feas = "có đường" if case.get("feasible", True) else "KHÔNG có đường (case âm)"
    fig.suptitle(f"{case['id']}  ·  {case.get('category', '')}  ·  chế độ {mode}  ·  {feas}"
                 + (f"  ·  tham chiếu {ref:.1f} m" if ref else ""), fontsize=11, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Nền xám: độ cao (tối = cao) · ô đen: vật cản · xanh: ĐẠT · đỏ đứt: HỎNG · "
             "× đỏ: cắt vật cản · chấm vàng: vượt lật · vòng tròn: dung sai goal",
             fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96), h_pad=3.0)
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_index(viewer_dir, entries, run_name):
    data = json.dumps(entries, ensure_ascii=False)
    cats = sorted({e["category"] for e in entries})
    opts = "".join(f'<option value="{html.escape(c)}">{html.escape(c)}</option>' for c in cats)
    popts = "".join(f'<option value="{html.escape(p)}">{html.escape(p)} hỏng</option>' for p in PLANNER_NAMES)
    page = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kết quả {html.escape(run_name)}</title>
<style>
:root {{ --surface:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3dd; --ok:#2a78d6; --bad:#d03b3b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --surface:#1a1a19; --card:#232322; --ink:#f0efec; --ink2:#c3c2b7; --line:#3a3a38; --ok:#3987e5; --bad:#e66767; }} }}
body {{ margin:0; background:var(--surface); color:var(--ink); font:14px/1.45 system-ui, sans-serif; }}
header {{ padding:16px; border-bottom:1px solid var(--line); position:sticky; top:0; background:var(--surface); z-index:2; }}
h1 {{ font-size:18px; margin:0 0 10px; }}
.filters {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
select, input {{ font:inherit; padding:5px 8px; border:1px solid var(--line); border-radius:6px; background:var(--card); color:var(--ink); }}
#count {{ color:var(--ink2); margin-left:auto; }}
main {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(min(100%, 360px), 1fr)); gap:14px; padding:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
.card img {{ width:100%; display:block; cursor:zoom-in; }}
.meta {{ padding:8px 10px; }}
.meta b {{ display:block; font-size:13px; word-break:break-all; }}
.chips {{ display:flex; flex-wrap:wrap; gap:4px; margin-top:6px; }}
.chip {{ font-size:12px; padding:2px 6px; border-radius:4px; border:1px solid var(--line); }}
.chip.ok {{ color:var(--ok); }} .chip.bad {{ color:var(--bad); }}
dialog {{ border:none; padding:0; max-width:96vw; background:var(--card); }}
dialog img {{ max-width:96vw; max-height:92vh; display:block; }}
</style></head><body>
<header><h1>Kết quả: {html.escape(run_name)}</h1>
<div class="filters">
<select id="mode"><option value="">Mọi chế độ</option><option value="sem">sem</option><option value="geo">geo</option></select>
<select id="cat"><option value="">Mọi nhóm</option>{opts}</select>
<select id="st"><option value="">Mọi trạng thái</option><option value="anyfail">Có planner hỏng</option>
<option value="allfail">Mọi planner hỏng</option><option value="allok">Mọi planner đạt</option>{popts}</select>
<input id="q" placeholder="Tìm theo tên case">
<span id="count"></span></div></header>
<main id="grid"></main>
<dialog id="dlg"><img id="big" alt=""></dialog>
<script>
const DATA = {data};
const $ = id => document.getElementById(id);
function render() {{
  const m = $("mode").value, c = $("cat").value, s = $("st").value, q = $("q").value.toLowerCase();
  const rows = DATA.filter(e => (!m || e.mode === m) && (!c || e.category === c) && (!q || e.case_id.toLowerCase().includes(q))
    && (!s || (s === "anyfail" && e.planners.some(p => !p.ok)) || (s === "allfail" && e.planners.every(p => !p.ok))
        || (s === "allok" && e.planners.every(p => p.ok)) || e.planners.some(p => p.name === s && !p.ok)));
  $("count").textContent = rows.length + " / " + DATA.length + " ảnh";
  $("grid").innerHTML = rows.map(e => `<div class="card"><img loading="lazy" src="${{e.img}}" alt="${{e.case_id}} ${{e.mode}}">
    <div class="meta"><b>${{e.case_id}} · ${{e.mode}}</b><span style="color:var(--ink2)">${{e.category}}${{e.feasible ? "" : " · case âm"}}</span>
    <div class="chips">${{e.planners.map(p => `<span class="chip ${{p.ok ? "ok" : "bad"}}">${{p.ok ? "✓" : "✗"}} ${{p.name}}</span>`).join("")}}</div></div></div>`).join("");
  document.querySelectorAll(".card img").forEach(img => img.onclick = () => {{ $("big").src = img.src; $("dlg").showModal(); }});
}}
["mode", "cat", "st"].forEach(id => $(id).onchange = render); $("q").oninput = render;
$("dlg").onclick = () => $("dlg").close();
render();
</script></body></html>"""
    with open(os.path.join(viewer_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)


def main():
    ap = argparse.ArgumentParser(description="Vẽ đường đi của một lần chạy test suite + trang HTML duyệt")
    ap.add_argument("run", help="Tên lần chạy trong logs/testsuite/ hoặc đường dẫn thư mục")
    ap.add_argument("--only-failures", action="store_true", help="Chỉ vẽ case có ít nhất 1 planner hỏng")
    ap.add_argument("--limit", type=int, default=None, help="Vẽ tối đa N ảnh")
    ap.add_argument("--goal-tol", type=float, default=None, help="Mặc định lấy từ meta của lần chạy")
    args = ap.parse_args()

    run_dir = args.run if os.path.isdir(args.run) else os.path.join(DEFAULT_LOG_ROOT, args.run)
    paths_dir = os.path.join(run_dir, "paths")
    if not os.path.isdir(paths_dir):
        sys.exit(f"Không có {paths_dir}. Chạy lại run_testsuite.py với --save-paths.")
    results, cases = load_run(run_dir)
    with open(os.path.join(run_dir, "results.json"), encoding="utf-8") as f:
        goal_tol = args.goal_tol or json.load(f)["meta"].get("goal_tol", 1.0)

    groups = {}
    for r in results:
        if r["run"] != 1:
            continue
        groups.setdefault((r["case_id"], r["mode"]), []).append(r)
    keys = [k for k in groups if not args.only_failures or any(not x.get("correct", x["valid"]) for x in groups[k])]
    keys.sort(key=lambda k: (cases[k[0]]["map"], k[0], k[1]))
    if args.limit:
        keys = keys[: args.limit]

    viewer_dir = os.path.join(run_dir, "viewer")
    os.makedirs(os.path.join(viewer_dir, "img"), exist_ok=True)
    map_cache, entries = {}, []
    for i, (cid, mode) in enumerate(keys, 1):
        case = cases[cid]
        mp = case["map_path"] if os.path.isabs(case["map_path"]) else os.path.join(HERE, case["map_path"])
        if mp not in map_cache:
            map_cache.clear()
            map_cache[mp] = load_map_data(mp)
        docs = {}
        for r in groups[(cid, mode)]:
            fp = os.path.join(paths_dir, path_file_name(cid, mode, r["planner"], r["run"]))
            if os.path.isfile(fp):
                with open(fp, encoding="utf-8") as f:
                    docs[r["planner"]] = json.load(f)
        img = f"img/{cid}__{mode}.png"
        render_case(case, mode, docs, map_cache[mp], goal_tol, os.path.join(viewer_dir, img))
        entries.append({"case_id": cid, "mode": mode, "category": case.get("category", ""),
                        "feasible": case.get("feasible", True), "img": img,
                        "planners": [{"name": r["planner"], "ok": bool(r.get("correct", r["valid"]))}
                                     for r in sorted(groups[(cid, mode)], key=lambda x: PLANNER_NAMES.index(x["planner"]))]})
        if i % 20 == 0 or i == len(keys):
            print(f"  đã vẽ {i}/{len(keys)}", flush=True)
    write_index(viewer_dir, entries, os.path.basename(os.path.normpath(run_dir)))
    print(f"[OK] {len(entries)} ảnh -> {os.path.join(viewer_dir, 'index.html')}")


if __name__ == "__main__":
    main()
