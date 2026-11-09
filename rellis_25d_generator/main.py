#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chương trình sinh bộ dữ liệu bản đồ 2.5D (.npz) từ dữ liệu thô RELLIS-3D LiDAR (.bin & .label).

Ví dụ sử dụng:
  1) Tải bộ mẫu RELLIS-3D chính thức (8 frames) và sinh toàn bộ .npz + ảnh dashboard:
     python3 main.py --download-sample --all --preview-every 1

  2) Sinh frame cụ thể hoặc toàn bộ sequence (khi đã có bộ RELLIS-3D trong raw_data/):
     python3 main.py --seq 00000 --frame 000000 --num-frames 5 --preview-every 1
     python3 main.py --seq 00000 --all --workers 6 --preview-every 50

  3) Chạy theo chuẩn báo cáo Nhóm 1 (reproduce schema 150x200 + valid_mask):
     python3 main.py --mode group1 --all --preview-every 1 --out output_group1

  4) Kiểm tra 1 file .npz hoặc đánh giá định lượng cả thư mục output:
     python3 main.py --inspect output_25d/00000_000001_25d_dataset.npz
     python3 main.py --evaluate --out output_25d
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import time

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import RellisConfig
from data_loader import RellisDataLoader, download_sample
from evaluator import evaluate_dataset, inspect_npz
from pipeline import process_single_frame
from visualizer import create_gif


def build_parser():
    parser = argparse.ArgumentParser(
        description="RELLIS-3D Point Cloud (.bin/.label) to 2.5D .npz Map Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["jikay", "group1"],
        default="jikay",
        help="Chọn thuật toán tạo lưới 2.5D: 'jikay' (200x150 BEV, EDT + Inflation, mặc định) hoặc 'group1' (150x200 Z25%% + valid_mask)",
    )
    parser.add_argument("--seq", type=str, default="00000", help="Sequence ID (mặc định: 00000)")
    parser.add_argument("--all", action="store_true", help="Xử lý toàn bộ frame tìm thấy trong sequence")
    parser.add_argument("--frame", type=str, default="000000", help="Frame bắt đầu (mặc định: 000000)")
    parser.add_argument(
        "--num-frames", "--num_frames", dest="num_frames", type=int, default=1,
        help="Số lượng frame liên tiếp cần xử lý nếu không bật --all (mặc định: 1)",
    )
    parser.add_argument("--workers", type=int, default=6, help="Số luồng xử lý song song (mặc định: 6)")
    parser.add_argument(
        "--preview-every", "--preview_every", dest="preview_every", type=int, default=1,
        help="Chu kỳ xuất ảnh Dashboard PNG (1: mọi frame; 50: mỗi 50 frame; 0: tắt)",
    )
    parser.add_argument(
        "--skip-existing", "--skip_existing", dest="skip_existing", action="store_true",
        help="Bỏ qua frame đã tồn tại file .npz",
    )
    parser.add_argument(
        "--out", type=str, default=str(BASE_DIR / "output_25d"),
        help="Thư mục xuất file .npz & ảnh (mặc định: output_25d)",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Thư mục/zip chứa dữ liệu thô RELLIS-3D")
    parser.add_argument("--bin-dir", type=str, default=None, help="Chỉ định trực tiếp thư mục hoặc .zip chứa .bin")
    parser.add_argument("--label-dir", type=str, default=None, help="Chỉ định trực tiếp thư mục hoặc .zip chứa .label")

    # Tiện ích đi kèm
    parser.add_argument(
        "--download-sample", action="store_true",
        help="Tải và giải nén bộ mẫu RELLIS-3D Ouster (8 frames) từ Google Drive vào raw_data/",
    )
    parser.add_argument("--inspect", type=str, default=None, help="Kiểm tra nội dung một file .npz")
    parser.add_argument("--evaluate", action="store_true", help="Đánh giá định lượng thư mục kết quả --out")
    parser.add_argument("--gif", action="store_true", help="Ghép các ảnh Dashboard trong --out thành file GIF")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    out_path = Path(args.out).resolve()

    if args.inspect:
        inspect_npz(args.inspect)
        return

    if args.evaluate:
        evaluate_dataset(out_path)
        return

    if args.gif:
        create_gif(png_dir=str(out_path), output_gif=str(out_path / f"demo_{args.seq}.gif"))
        return

    if args.download_sample:
        raw_dir = Path(args.data_dir).resolve() if args.data_dir else (BASE_DIR / "raw_data")
        download_sample(raw_dir=raw_dir, extract=True)
        # Nếu user chỉ muốn tải dữ liệu mẫu mà không chỉ định --all / --num-frames / --frame riêng
        if len(sys.argv) == 2 and "--download-sample" in sys.argv:
            print("[Info] Đã tải xong bộ mẫu. Chạy `python3 main.py --all` để sinh bản đồ 2.5D.")
            return

    cfg = RellisConfig()
    loader = RellisDataLoader(
        workspace_dir=BASE_DIR,
        data_dir=args.data_dir,
        bin_dir=args.bin_dir,
        label_dir=args.label_dir,
    )

    all_available = loader.get_all_frame_ids(args.seq)
    if not all_available:
        print(f"[LỖI] Không tìm thấy frame nào cho sequence {args.seq}!")
        sys.exit(1)

    if args.all:
        frame_list = all_available
    else:
        if args.frame in all_available and args.num_frames == 1:
            frame_list = [args.frame]
        else:
            start_int = int(args.frame)
            frame_list = [
                f"{start_int + i:06d}"
                for i in range(args.num_frames)
                if f"{start_int + i:06d}" in all_available
            ]
            if not frame_list:
                # Dự phòng theo vị trí trong danh sách hiện có
                frame_list = all_available[: args.num_frames]

    total = len(frame_list)
    print(f"\n============================================================")
    print(f" XỬ LÝ {total:,} FRAME | SEQ: [{args.seq}] | MODE: [{args.mode}]")
    print(f"============================================================")
    print(f" - Số frame khả dụng trong nguồn : {len(all_available):,}")
    print(f" - Chu kỳ xuất Dashboard (--preview-every): {args.preview_every}")
    print(f" - Thư mục xuất (.npz & PNG)     : {out_path}\n")

    if total == 1:
        process_single_frame(
            loader=loader,
            sequence_id=args.seq,
            frame_id=frame_list[0],
            out_dir=out_path,
            cfg=cfg,
            mode=args.mode,
            save_preview=(args.preview_every > 0),
            skip_existing=args.skip_existing,
            verbose=True,
        )
        return

    t0 = time.time()
    ok_cnt = 0
    fail_cnt = 0

    def task_fn(fid: str):
        f_int = int(fid) if fid.isdigit() else 0
        save_prev = (args.preview_every > 0) and (f_int % args.preview_every == 0)
        res = process_single_frame(
            loader=loader,
            sequence_id=args.seq,
            frame_id=fid,
            out_dir=out_path,
            cfg=cfg,
            mode=args.mode,
            save_preview=save_prev,
            skip_existing=args.skip_existing,
            verbose=False,
        )
        return res, fid

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(task_fn, fid): fid for fid in frame_list}
        for fut in as_completed(futures):
            fid = futures[fut]
            try:
                fut.result()
                ok_cnt += 1
            except Exception as exc:
                fail_cnt += 1
                print(f"[LỖI] Frame {fid}: {exc}")

            done = ok_cnt + fail_cnt
            elapsed = time.time() - t0
            fps = done / elapsed if elapsed > 0 else 0.0
            if done <= 5 or done % 25 == 0 or done == total:
                print(f"[{done:4d}/{total:4d}] Frame {fid} xong ({fps:.1f} FPS)")

    total_time = time.time() - t0
    print(f"\n============================================================")
    print(f" HOÀN TẤT {ok_cnt:,}/{total:,} FRAME trong {total_time:.2f}s ({total / max(total_time, 1e-6):.1f} FPS)")
    if fail_cnt:
        print(f" - Số frame lỗi: {fail_cnt}")
    print(f" - Kết quả lưu tại: {out_path}")
    print(f"============================================================\n")


if __name__ == "__main__":
    main()

