import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import util.misc as utils
from datasets import build_dataset
from models import build_model
from util.box_ops import box_cxcywh_to_xyxy, box_iou

def greedy_match(pred_boxes_xyxy, pred_labels, pred_scores,
                 gt_boxes_xyxy, gt_labels,
                 iou_thresh=0.5):
    """
    Greedy 1-to-1 matching between preds and GTs within a single image.

    Returns
    -------
    pairs : list of (gt_idx, pred_idx, pred_label, gt_label)
        Matched pred/GT pairs. A pred_idx of -1 means the GT was not
        matched (FN). A gt_idx of -1 means the pred was not matched (FP).
    """
    n_pred = pred_boxes_xyxy.shape[0]
    n_gt = gt_boxes_xyxy.shape[0]

    if n_gt == 0 and n_pred == 0:
        return []
    if n_gt == 0:
        return [(-1, i, int(pred_labels[i].item()), None) for i in range(n_pred)]
    if n_pred == 0:
        return [(j, -1, None, int(gt_labels[j].item())) for j in range(n_gt)]

    iou, _ = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
    iou_np = iou.detach().cpu().numpy()
    scores_np = pred_scores.detach().cpu().numpy()

    matched_gt = set()
    matched_pred = set()
    pairs = []

    pred_order = np.argsort(-scores_np)
    for p in pred_order:
        best_gt = -1
        best_iou = iou_thresh 
        for g in range(n_gt):
            if g in matched_gt:
                continue
            if iou_np[p, g] > best_iou:
                best_iou = iou_np[p, g]
                best_gt = g
        if best_gt >= 0:
            matched_gt.add(best_gt)
            matched_pred.add(p)
            pairs.append((best_gt, p,
                          int(pred_labels[p].item()),
                          int(gt_labels[best_gt].item())))

    for p in range(n_pred):
        if p not in matched_pred:
            pairs.append((-1, p, int(pred_labels[p].item()), None))

    for g in range(n_gt):
        if g not in matched_gt:
            pairs.append((g, -1, None, int(gt_labels[g].item())))

    return pairs

@torch.no_grad()
def build_confusion_matrix(model, postprocessors, data_loader, device,
                           num_classes, score_thresh, iou_thresh):
    """
    Run inference over data_loader and accumulate a
    (num_classes + 1) x (num_classes + 1) confusion matrix.
    The extra row/col represents "background" (i.e. FN / FP).
    """
    bg_idx = num_classes 
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    model.eval()
    for samples, targets in tqdm(data_loader, desc="Building CM"):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        for target, result in zip(targets, results):
            gt_boxes_cxcywh = target['boxes']  # normalized cxcywh
            gt_labels = target['labels']
            orig_h, orig_w = target['orig_size'].tolist()

            gt_xyxy = box_cxcywh_to_xyxy(gt_boxes_cxcywh)
            if gt_xyxy.numel() > 0:
                scale = torch.tensor([orig_w, orig_h, orig_w, orig_h],
                                     device=gt_xyxy.device, dtype=gt_xyxy.dtype)
                gt_xyxy = gt_xyxy * scale

            pred_scores = result['scores']
            pred_labels = result['labels']
            pred_xyxy = result['boxes']

            keep = pred_scores >= score_thresh
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]
            pred_xyxy = pred_xyxy[keep]

            pairs = greedy_match(pred_xyxy, pred_labels, pred_scores,
                                 gt_xyxy, gt_labels,
                                 iou_thresh=iou_thresh)
            for gt_idx, pred_idx, p_lab, g_lab in pairs:
                if gt_idx == -1:
                    if 0 <= p_lab < num_classes:
                        cm[bg_idx, p_lab] += 1
                elif pred_idx == -1:
                    if 0 <= g_lab < num_classes:
                        cm[g_lab, bg_idx] += 1
                else:
                    if 0 <= g_lab < num_classes and 0 <= p_lab < num_classes:
                        cm[g_lab, p_lab] += 1

    return cm

def plot_cm(cm, class_names, out_path, normalize=True, title_suffix=""):
    """Plot two side-by-side heatmaps: raw counts and row-normalised."""
    num_rows = cm.shape[0]
    labels = class_names + ["background"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    ax = axes[0]
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_title(f"Confusion Matrix (raw counts){title_suffix}", fontsize=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_xticks(range(num_rows))
    ax.set_yticks(range(num_rows))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    max_val = cm.max() if cm.max() > 0 else 1
    for i in range(num_rows):
        for j in range(num_rows):
            v = cm[i, j]
            if v == 0:
                continue
            color = "white" if v > max_val * 0.5 else "black"
            ax.text(j, i, str(v), ha='center', va='center',
                    color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1  # avoid div-by-zero
    cm_norm = cm.astype(np.float64) / row_sum
    im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_title(f"Confusion Matrix (row-normalised){title_suffix}", fontsize=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_xticks(range(num_rows))
    ax.set_yticks(range(num_rows))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    for i in range(num_rows):
        for j in range(num_rows):
            v = cm_norm[i, j]
            if v < 0.01:
                continue
            color = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                    color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[ok] saved -> {out_path}")


def print_summary(cm, class_names):
    """Print per-class precision / recall for a quick sanity check."""
    num_classes = len(class_names)
    bg = num_classes
    print("\nPer-class metrics (IoU >= threshold, score >= threshold):")
    print(f"{'class':<12}{'TP':>6}{'FP':>6}{'FN':>6}{'prec':>8}{'rec':>8}")
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"{class_names[c]:<12}{tp:>6}{fp:>6}{fn:>6}{prec:>8.3f}{rec:>8.3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Path to checkpoint.pth')
    parser.add_argument('--coco_path', required=True,
                        help='Dataset root (contains valid/ and valid.json)')
    parser.add_argument('--out', default='confusion_matrix.png')
    parser.add_argument('--score_thresh', type=float, default=0.3,
                        help='Keep predictions with score >= this')
    parser.add_argument('--iou_thresh', type=float, default=0.5,
                        help='Min IoU to count as matched')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of foreground classes (digits 0-9 => 10)')
    parser.add_argument('--drop_classes', type=int, nargs='*', default=[],
                        help='Class indices to remove from the plotted matrix '
                             '(e.g. --drop_classes 0 will remove the 0-th row and col). '
                             'Useful when a class is unused (e.g. a placeholder id 0).')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[info] loading checkpoint {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model_args = ckpt['args']

    model_args.device = str(device)
    model_args.coco_path = args.coco_path
    model_args.masks = getattr(model_args, 'masks', False)
    model_args.distributed = False

    model, _, postprocessors = build_model(model_args)
    model.load_state_dict(ckpt['model'])
    model.to(device)

    dataset_val = build_dataset(image_set='val', args=model_args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader = DataLoader(
        dataset_val, args.batch_size, sampler=sampler_val,
        drop_last=False, collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    class_names = [str(i) for i in range(args.num_classes)]

    cm = build_confusion_matrix(
        model, postprocessors, data_loader, device,
        num_classes=args.num_classes,
        score_thresh=args.score_thresh,
        iou_thresh=args.iou_thresh,
    )

    if args.drop_classes:
        drop = sorted(set(args.drop_classes))
        for c in drop:
            if c < 0 or c >= args.num_classes:
                raise ValueError(f"--drop_classes {c} is out of range [0, {args.num_classes - 1}]")
        keep_cls = [c for c in range(args.num_classes) if c not in drop]
        bg_idx = args.num_classes
        keep_idx = keep_cls + [bg_idx]  # keep background at the end
        cm = cm[np.ix_(keep_idx, keep_idx)]
        class_names = [class_names[c] for c in keep_cls]
        print(f"[info] dropped classes {drop}; remaining: {class_names}")

    print_summary(cm, class_names)
    suffix = f"  [score>={args.score_thresh}, IoU>={args.iou_thresh}]"
    plot_cm(cm, class_names, Path(args.out), title_suffix=suffix)

    json_path = Path(args.out).with_suffix('.json')
    with json_path.open('w') as f:
        json.dump({
            'labels': class_names + ['background'],
            'matrix': cm.tolist(),
            'score_thresh': args.score_thresh,
            'iou_thresh': args.iou_thresh,
            'dropped_classes': args.drop_classes,
            'checkpoint': args.checkpoint,
        }, f, indent=2)
    print(f"[ok] raw matrix -> {json_path}")


if __name__ == '__main__':
    main()