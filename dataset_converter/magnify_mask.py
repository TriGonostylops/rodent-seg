import os
import cv2
import numpy as np
from tqdm import tqdm

# ================= CONFIGURATION =================
# Paths to your original dataset
IMG_DIR = "/kaggle/input/datasets/gonoszgonosz/rodent-data-2/processed/images"
MASK_DIR = "/kaggle/input/datasets/gonoszgonosz/rodent-data-2/processed/masks"

# Where to save the zoomed "Expert" dataset
SAVE_IMG_DIR = "/kaggle/working/stage2_dataset/images"
SAVE_MASK_DIR = "/kaggle/working/stage2_dataset/masks"

# Parameters
TARGET_SIZE = 512  # Final resolution of the zoomed crop
PADDING_RATIO = 0.25  # 25% extra space around the rat
# =================================================

os.makedirs(SAVE_IMG_DIR, exist_ok=True)
os.makedirs(SAVE_MASK_DIR, exist_ok=True)


def get_zoomed_crop(image, mask, padding=0.25):
    h, w = mask.shape
    coords = np.column_stack(np.where(mask > 0))

    if coords.size == 0:
        return None, None  # Skip frames with no rat

    # 1. Find the Bounding Box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # 2. Add Padding
    bw, bh = x_max - x_min, y_max - y_min
    pad_w, pad_h = int(bw * padding), int(bh * padding)

    x_min = max(0, x_min - pad_w)
    y_min = max(0, y_min - pad_h)
    x_max = min(w, x_max + pad_w)
    y_max = min(h, y_max + pad_h)

    # 3. Square-ify the crop
    crop_w, crop_h = x_max - x_min, y_max - y_min
    side = max(crop_w, crop_h)

    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2

    x_min = max(0, cx - side // 2)
    y_min = max(0, cy - side // 2)
    x_max = min(w, x_min + side)
    y_max = min(h, y_min + side)

    # Slice
    img_crop = image[y_min:y_max, x_min:x_max]
    mask_crop = mask[y_min:y_max, x_min:x_max]

    # Check if crop is valid (prevents edge-case crashes on image borders)
    if img_crop.size == 0 or mask_crop.size == 0:
        return None, None

    # 4. Upscale with Lanczos (High quality for edges)
    img_res = cv2.resize(img_crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)
    mask_res = cv2.resize(mask_crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)

    return img_res, mask_res


# --- SMART FILE MATCHING ---
all_images = [f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png'))]
all_masks = [f for f in os.listdir(MASK_DIR) if f.endswith(('.jpg', '.png'))]

# Map by filename without extension
img_map = {os.path.splitext(f)[0]: f for f in all_images}
mask_map = {os.path.splitext(f)[0]: f for f in all_masks}

# Find only pairs that exist in BOTH folders
common_ids = sorted(list(set(img_map.keys()) & set(mask_map.keys())))
print(f"Found {len(common_ids)} matching image/mask pairs.")

# Run the processing
for cid in tqdm(common_ids):
    img_path = os.path.join(IMG_DIR, img_map[cid])
    mask_path = os.path.join(MASK_DIR, mask_map[cid])

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Safety check in case the file is corrupted
    if img is None or mask is None:
        print(f"Skipping {cid}: Could not read file.")
        continue

    zoom_img, zoom_mask = get_zoomed_crop(img, mask, PADDING_RATIO)

    if zoom_img is not None:
        # Save as PNG to avoid compression artifacts on our nice clean crops
        save_name = f"{cid}.png"
        cv2.imwrite(os.path.join(SAVE_IMG_DIR, save_name), zoom_img)
        cv2.imwrite(os.path.join(SAVE_MASK_DIR, save_name), zoom_mask)

print(f"Stage 2 Dataset Ready in {SAVE_IMG_DIR}")