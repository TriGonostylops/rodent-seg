import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURATION =================
DATASET_ROOT = "dataset/rodent-data-final"
OUTPUT_ROOT  = "dataset/magnified-rodent-data"
# Kaggle overrides:
# DATASET_ROOT = "/kaggle/input/rodent-data-final/rodent-data-final"
# OUTPUT_ROOT  = "/kaggle/working/magnified-rodent-data"
SPLITS       = ["train", "val", "test"]

TARGET_SIZE   = 512
PADDING_RATIO = 0.25
# =================================================


def get_zoomed_crop(image, mask, padding=0.25):
    h, w = mask.shape
    coords = np.column_stack(np.where(mask > 0))

    if coords.size == 0:
        return None, None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    bw, bh   = x_max - x_min, y_max - y_min
    pad_w    = int(bw * padding)
    pad_h    = int(bh * padding)
    x_min    = max(0, x_min - pad_w)
    y_min    = max(0, y_min - pad_h)
    x_max    = min(w, x_max + pad_w)
    y_max    = min(h, y_max + pad_h)

    side = max(x_max - x_min, y_max - y_min)
    cx   = (x_min + x_max) // 2
    cy   = (y_min + y_max) // 2
    x_min = max(0, cx - side // 2)
    y_min = max(0, cy - side // 2)
    x_max = min(w, x_min + side)
    y_max = min(h, y_min + side)

    img_crop  = image[y_min:y_max, x_min:x_max]
    mask_crop = mask[y_min:y_max, x_min:x_max]

    if img_crop.size == 0 or mask_crop.size == 0:
        return None, None

    img_res  = cv2.resize(img_crop,  (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)
    mask_res = cv2.resize(mask_crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
    return img_res, mask_res


for split in SPLITS:
    img_dir  = Path(DATASET_ROOT) / split / "images"
    mask_dir = Path(DATASET_ROOT) / split / "masks"

    if not img_dir.exists():
        print(f"[{split}] Skipping — not found at {img_dir}")
        continue

    out_img_dir  = Path(OUTPUT_ROOT) / split / "images"
    out_mask_dir = Path(OUTPUT_ROOT) / split / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    img_map  = {Path(f).stem: f for f in os.listdir(img_dir)  if f.endswith(('.jpg', '.png'))}
    mask_map = {Path(f).stem: f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png'))}
    common   = sorted(set(img_map) & set(mask_map))

    kept = skipped = 0
    for cid in tqdm(common, desc=split):
        img  = cv2.imread(str(img_dir  / img_map[cid]))
        mask = cv2.imread(str(mask_dir / mask_map[cid]), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        zoom_img, zoom_mask = get_zoomed_crop(img, mask, PADDING_RATIO)
        if zoom_img is not None:
            cv2.imwrite(str(out_img_dir  / f"{cid}.png"), zoom_img)
            cv2.imwrite(str(out_mask_dir / f"{cid}.png"), zoom_mask)
            kept += 1
        else:
            skipped += 1

    print(f"  [{split}] kept={kept}  skipped_empty={skipped}")

print(f"\nMagnified dataset ready at {OUTPUT_ROOT}")
