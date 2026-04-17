"""Final Inference Script for CodaBench Submission."""

import json
import os
from typing import Any, Dict, List

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from models import build_model


def get_transform() -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def resize_image(image: Image.Image, short_size: int = 384, max_size: int = 600) -> Image.Image:
    """【救命關鍵 1】確保測試集大圖片縮放到模型認識的大小"""
    w, h = image.size
    if w < h:
        ow = short_size
        oh = int(short_size * h / w)
    else:
        oh = short_size
        ow = int(short_size * w / h)
        
    if max(ow, oh) > max_size:
        scale = max_size / max(ow, oh)
        ow = int(ow * scale)
        oh = int(oh * scale)
        
    return image.resize((ow, oh), Image.Resampling.BILINEAR)

def box_cxcywh_to_xywh(boxes: torch.Tensor, orig_w: int, orig_h: int) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x_min = (cx - 0.5 * w) * orig_w
    y_min = (cy - 0.5 * h) * orig_h
    w_abs = w * orig_w
    h_abs = h * orig_h
    return torch.stack([x_min, y_min, w_abs, h_abs], dim=-1)

def main() -> None:
    # 記得確認你的 checkpoint 路徑
    checkpoint_path = 'output/cond_detr_digit_v3/checkpoint_best.pth'
    image_dir = 'data/test'
    output_json = 'pred.json' 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    args = checkpoint['args']
    args.device = str(device)

    model, _, _ = build_model(args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    transform = get_transform()
    predictions: List[Dict[str, Any]] = []

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg'))]
    # 確保依照數字排序
    image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))

    print(f"Starting inference on {len(image_files)} images...")
    with torch.no_grad():
        for filename in tqdm(image_files):
            img_id = int(os.path.splitext(filename)[0])
            img_path = os.path.join(image_dir, filename)

            image = Image.open(img_path).convert('RGB')
            orig_w, orig_h = image.size

            # 進行 Resize
            resized_image = resize_image(image)

            img_tensor = transform(resized_image).unsqueeze(0).to(device)
            outputs = model(img_tensor)

            # 1. 完美對齊訓練期的 Sigmoid (不要用 Softmax)
            out_logits = outputs['pred_logits'][0]  # shape: [num_queries, num_classes]
            out_bbox = outputs['pred_boxes'][0]     # shape: [num_queries, 4]
            prob = out_logits.sigmoid()

            
            # 2. 完美對齊 Validation 的 Top-100 邏輯 (取代原本的 score > threshold)
            # 取出單張圖片分數最高的前 100 個預測
            topk_values, topk_indexes = torch.topk(prob.view(-1), min(100, prob.numel()))
            scores = topk_values

            # 【新增】再用一個保守的門檻過濾掉太誇張的垃圾框
            keep = topk_values > 0.05 
            scores = topk_values[keep]
            topk_indexes = topk_indexes[keep]
            
            # 反推回對應的 query index 與 class label
            num_classes = out_logits.shape[1]
            topk_boxes_idx = topk_indexes // num_classes
            labels = topk_indexes % num_classes
            
            boxes = out_bbox[topk_boxes_idx]

            # 3. 座標還原 (注意轉為 [x_min, y_min, w, h] 格式給 JSON)
            abs_boxes = box_cxcywh_to_xywh(boxes, orig_w, orig_h)

            for score, label, box in zip(scores, labels, abs_boxes):
                box_list = box.tolist()
                predictions.append({
                    "image_id": img_id,
                    "bbox": [max(0.0, box_list[0]), max(0.0, box_list[1]), box_list[2], box_list[3]],
                    "score": score.item(),
                    "category_id": int(label.item()) 
                })

    # 【救命關鍵 3】將 JSON 輸出依照 image_id 以及 score (由高到低) 排序
    # 這樣你打開 JSON 檢查時，第一眼看到的才會是模型「最有把握的真正預測」！
    predictions.sort(key=lambda x: (x["image_id"], -x["score"]))

    # 統計每張圖的預測數量分布
    pred_counts = {}
    for p in predictions:
        img_id = p['image_id']
        pred_counts[img_id] = pred_counts.get(img_id, 0) + 1
    
    counts = list(pred_counts.values())
    print(f"平均每張預測數: {sum(counts)/len(counts):.1f}")
    print(f"最多: {max(counts)}, 最少: {min(counts)}")
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to {output_json}. Ready for submission!")

    

if __name__ == '__main__':
    main()