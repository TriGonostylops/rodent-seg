import cv2
import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path

# Relative imports
from .config import INTERIM_DIR, FILTERED_DIR, IOU_THRESHOLD
from .extract_masks import setup_directories

def calculate_iou(mask1, mask2):
    if mask1 is None or mask2 is None: return 0.0

    m1 = mask1 > 0
    m2 = mask2 > 0

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    return intersection / union if union > 0 else 0.0


def run_filtering():
    print(f"--- STEP 2: FILTERING (IoU < {IOU_THRESHOLD}) ---")

    in_img_dir = INTERIM_DIR / "images"
    in_mask_dir = INTERIM_DIR / "masks"

    if not in_img_dir.exists():
        raise FileNotFoundError(f"Raw data not found at {INTERIM_DIR}. Run Step 1 first.")

    out_img_dir, out_mask_dir = setup_directories(FILTERED_DIR)

    img_files = sorted(list(in_img_dir.glob("*.jpg")))
    print(f"Scanning {len(img_files)} raw frames...")

    last_saved_mask = None
    kept_count = 0
    dropped_count = 0

    for img_path in tqdm(img_files, desc="Filtering"):
        mask_path = in_mask_dir / f"{img_path.stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None: continue

        should_keep = True

        if last_saved_mask is not None:
            iou = calculate_iou(last_saved_mask, mask)
            if iou > IOU_THRESHOLD:
                should_keep = False
                dropped_count += 1

        if should_keep:
            shutil.copy(img_path, out_img_dir / img_path.name)
            shutil.copy(mask_path, out_mask_dir / mask_path.name)

            last_saved_mask = mask
            kept_count += 1

    print(f"Step 2 Complete.")
    print(f"Dropped (Static): {dropped_count}")
    print(f"Kept (Diverse): {kept_count}")
    print(f"Clean Data ready in: {FILTERED_DIR}")