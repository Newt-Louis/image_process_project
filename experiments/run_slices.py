"""KỊCH BẢN B - Phân tích theo lát cắt dữ liệu (Mục 3.3).

Không chạy lại inference: đọc lại predictions_*.json do run_eval.py sinh ra,
rồi chạy COCOeval trên từng nhóm ảnh:
  - theo độ khó `level` (easy / medium / hard)
  - theo mật độ sản phẩm/ảnh (<=8 / 9-15 / >=16)
  - recall theo kích thước bbox

    python -m experiments.run_slices --tag main
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments import common as C
import config as cfg


def slice_eval(ts: C.TestSet, dets: list[dict], groups: dict[str, list[int]],
               eval_ids: set[int]) -> dict:
    out = {}
    for gname, ids in groups.items():
        ids = [i for i in ids if i in eval_ids]
        if not ids:
            continue
        stats = C.coco_evaluate(ts.coco, dets, ids)
        n_gt = sum(len(ts.coco.getAnnIds(imgIds=[i], iscrowd=False)) for i in ids)
        out[gname] = {"n_images": len(ids), "n_gt_boxes": n_gt, **stats}
    return out


def main():
    ap = argparse.ArgumentParser(description="Kịch bản B - mAP theo độ khó & mật độ")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--level-source", default=None)
    args = ap.parse_args()

    C.set_seed()
    out_dir = cfg.RESULTS_DIR / args.tag
    ts = C.TestSet(level_source=Path(args.level_source) if args.level_source else None)
    models = cfg.MODEL_ORDER if args.models == ["all"] else args.models

    if ts.level_is_proxy:
        print("[lưu ý] instances_test.json không có field `level` của RPC gốc.\n"
              "        -> Đang suy ra level từ số instance/ảnh theo quy ước RPC "
              f"({', '.join(f'{n}={lo}-{hi}' for n, lo, hi in cfg.LEVEL_BINS)}).\n"
              "        Ghi rõ điều này trong báo cáo, hoặc truyền --level-source "
              "<instances_test2019.json> để dùng nhãn gốc.")

    lv_groups = ts.group("level")
    de_groups = ts.group("density")
    print("\nSố ảnh theo level:  ", {k: len(v) for k, v in sorted(lv_groups.items())})
    print("Số ảnh theo mật độ: ", {k: len(v) for k, v in de_groups.items()})

    ids_file = out_dir / "eval_ids.json"
    if not ids_file.exists():
        raise SystemExit(f"Thiếu {ids_file} — chạy `python -m experiments.run_eval` trước.")
    eval_ids = set(C.load_json(ids_file))

    results = {"level_is_proxy": ts.level_is_proxy, "n_images": len(eval_ids), "models": {}}
    for name in models:
        pred_file = out_dir / f"predictions_{name}.json"
        if not pred_file.exists():
            print(f"[bỏ qua] chưa có {pred_file} — chạy run_eval.py trước.")
            continue
        dets = C.load_json(pred_file)

        print(f"\n--- {cfg.DISPLAY_NAMES[name]} ({len(eval_ids)} ảnh) ---")
        results["models"][name] = {
            "by_level": slice_eval(ts, dets, lv_groups, eval_ids),
            "by_density": slice_eval(ts, dets, de_groups, eval_ids),
        }
        for k, v in results["models"][name]["by_level"].items():
            print(f"  level={k:7s} n={v['n_images']:5d}  mAP50={v['mAP50']:.4f}  "
                  f"mAP50-95={v['mAP50_95']:.4f}")
        for k, v in results["models"][name]["by_density"].items():
            print(f"  density={k:6s} n={v['n_images']:5d}  mAP50={v['mAP50']:.4f}  "
                  f"mAP50-95={v['mAP50_95']:.4f}")

    C.save_json(results, out_dir / "slices.json")

    # Bảng 3 in nhanh
    lv_names = [n for n, _, _ in cfg.LEVEL_BINS]
    print("\n" + "=" * 78 + "\nBẢNG 3 - mAP50-95 theo độ khó\n" + "=" * 78)
    print(C.md_table(["Model", *lv_names],
                     [[cfg.DISPLAY_NAMES[m],
                       *[C.fmt(results["models"][m]["by_level"].get(l, {}).get("mAP50_95"))
                         for l in lv_names]]
                      for m in results["models"]]))

    de_names = [n for n, _, _ in cfg.DENSITY_BINS]
    print("\nBẢNG 3b - mAP50-95 theo mật độ sản phẩm/ảnh")
    print(C.md_table(["Model", *de_names],
                     [[cfg.DISPLAY_NAMES[m],
                       *[C.fmt(results["models"][m]["by_density"].get(d, {}).get("mAP50_95"))
                         for d in de_names]]
                      for m in results["models"]]))
    print(f"\nXong -> {out_dir / 'slices.json'}")


if __name__ == "__main__":
    main()
