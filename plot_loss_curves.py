"""
plot_loss_curves.py
--------------------
Read a DETR-style log.txt file (one JSON object per line) and draw:
  1. Total train / test loss
  2. Classification loss (loss_ce)
  3. Bounding-box L1 loss (loss_bbox)
  4. GIoU loss (loss_giou)
  5. Class error (%)
  6. COCO AP curves (AP@0.50:0.95, AP@0.50, AP@0.75)

Usage
-----
    python plot_loss_curves.py --log path/to/log.txt --out curves.png

The script is robust to:
  * missing fields (e.g. early logs without coco_eval_bbox)
  * extra fields (e.g. loss_ciou if you added CIoU)
  * comments or empty lines
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_log(log_path: Path):
    """Parse the log.txt file into a list of dicts, one per epoch."""
    records = []
    with log_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] skipping malformed line {line_num}: {e}")
    records.sort(key=lambda r: r.get("epoch", 0))
    return records


def safe_get(records, key, default=None):
    """Return a list of values for `key`; fill missing with `default`."""
    return [r.get(key, default) for r in records]


def plot_curves(records, out_path: Path):
    epochs = [r["epoch"] for r in records]

    has_ciou = any("train_loss_ciou" in r or "test_loss_ciou" in r for r in records)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Curves", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, safe_get(records, "train_loss"), label="train_loss", linewidth=2)
    ax.plot(epochs, safe_get(records, "test_loss"), label="test_loss", linewidth=2)
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, safe_get(records, "train_loss_ce"), label="train_loss_ce", linewidth=2)
    ax.plot(epochs, safe_get(records, "test_loss_ce"), label="test_loss_ce", linewidth=2)
    ax.set_title("Classification Loss (loss_ce)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 2]
    ax.plot(epochs, safe_get(records, "train_loss_bbox"), label="train_loss_bbox", linewidth=2)
    ax.plot(epochs, safe_get(records, "test_loss_bbox"), label="test_loss_bbox", linewidth=2)
    ax.set_title("BBox L1 Loss (loss_bbox)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(epochs, safe_get(records, "train_loss_giou"), label="train_loss_giou", linewidth=2)
    ax.plot(epochs, safe_get(records, "test_loss_giou"), label="test_loss_giou", linewidth=2)
    if has_ciou:
        ax.plot(epochs, safe_get(records, "train_loss_ciou"),
                label="train_loss_ciou", linewidth=2, linestyle="--")
        ax.plot(epochs, safe_get(records, "test_loss_ciou"),
                label="test_loss_ciou", linewidth=2, linestyle="--")
        ax.set_title("GIoU / CIoU Loss")
    else:
        ax.set_title("GIoU Loss (loss_giou)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(epochs, safe_get(records, "train_class_error"),
            label="train_class_error", linewidth=2)
    ax.plot(epochs, safe_get(records, "test_class_error"),
            label="test_class_error", linewidth=2)
    ax.set_title("Class Error (%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 2]
    ap_5095, ap_50, ap_75 = [], [], []
    for r in records:
        stats = r.get("test_coco_eval_bbox")
        if stats and len(stats) >= 3:
            ap_5095.append(stats[0])
            ap_50.append(stats[1])
            ap_75.append(stats[2])
        else:
            ap_5095.append(None)
            ap_50.append(None)
            ap_75.append(None)
    ax.plot(epochs, ap_5095, label="AP @ 0.50:0.95", linewidth=2, marker="o", markersize=3)
    ax.plot(epochs, ap_50, label="AP @ 0.50", linewidth=2, marker="s", markersize=3)
    ax.plot(epochs, ap_75, label="AP @ 0.75", linewidth=2, marker="^", markersize=3)
    ax.set_title("COCO AP Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    lrs = safe_get(records, "train_lr")
    if all(v is not None for v in lrs) and len(lrs) > 1:
        for i in range(1, len(lrs)):
            if lrs[i] < lrs[i - 1] * 0.5:  # >=2x drop
                for row in axes:
                    for a in row:
                        a.axvline(epochs[i], color="red", linestyle=":",
                                  alpha=0.6, label="_lr_drop")
                break

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[ok] saved figure -> {out_path}")

    valid = [(r["epoch"], r["test_coco_eval_bbox"][0])
             for r in records
             if r.get("test_coco_eval_bbox") and len(r["test_coco_eval_bbox"]) > 0]
    if valid:
        best_epoch, best_ap = max(valid, key=lambda x: x[1])
        print(f"[info] best AP@0.50:0.95 = {best_ap:.4f} at epoch {best_epoch}")


def main():
    parser = argparse.ArgumentParser(description="Plot DETR training loss curves.")
    parser.add_argument("--log", type=str, default="log.txt",
                        help="Path to log.txt (default: ./log.txt)")
    parser.add_argument("--out", type=str, default="loss_curves.png",
                        help="Output figure path (default: ./loss_curves.png)")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)

    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")

    records = load_log(log_path)
    if not records:
        raise ValueError(f"no valid records parsed from {log_path}")
    print(f"[info] loaded {len(records)} epochs from {log_path}")

    plot_curves(records, out_path)


if __name__ == "__main__":
    main()