import shutil

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

from .config import VIDEO_PATH, JSON_PATH, INTERIM_DIR


def setup_directories(base_dir):
    """Cleans and recreates the output directories."""
    img_dir = base_dir / "images"
    mask_dir = base_dir / "masks"

    if base_dir.exists():
        shutil.rmtree(base_dir)

    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    return img_dir, mask_dir


def load_coco_data(json_path):

    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found at: {json_path}")

    print(f"Loading JSON: {json_path.name}")
    return COCO(str(json_path))


def build_frame_map(coco):

    img_ids = coco.getImgIds()
    frame_map = {}

    print("Parsing frame indices...")
    for i_id in img_ids:
        img_info = coco.loadImgs(i_id)[0]
        fname = img_info['file_name']
        try:
            # Logic: "frame_000013.jpg" -> "000013" -> 13
            frame_str = fname.split('_')[-1].split('.')[0]
            frame_num = int(frame_str)
            frame_map[frame_num] = img_info
        except ValueError:
            print(f"Warning: Could not parse frame number from filename '{fname}'")

    if not frame_map:
        raise ValueError("No frames could be mapped! Check your JSON filenames.")

    return frame_map


def initialize_video(video_path):

    print(f"Opening Video: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise IOError(f"CRITICAL: Could not open video at {video_path}. \n"
                      f"Check codec installation or path.")
    return cap


def generate_binary_mask(coco, anns, height, width):

    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        rle = coco.annToMask(ann)
        mask = np.maximum(mask, rle)
    return mask


def process_extraction_loop(cap, frame_map, coco, img_dir, mask_dir):

    saved_count = 0
    sorted_frames = sorted(frame_map.keys())

    for frame_idx in tqdm(sorted_frames, desc="Extracting"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"Error reading frame {frame_idx}. Stopping extraction.")
            break

        img_info = frame_map[frame_idx]
        ann_ids = coco.getAnnIds(imgIds=img_info['id'])
        anns = coco.loadAnns(ann_ids)

        if not anns:
            continue

        h, w = frame.shape[:2]
        mask = generate_binary_mask(coco, anns, h, w)

        name = f"frame_{frame_idx:06d}"
        cv2.imwrite(str(img_dir / f"{name}.jpg"), frame)
        cv2.imwrite(str(mask_dir / f"{name}.png"), mask * 255)

        saved_count += 1

    return saved_count


def run_extraction():
    print(f"--- STEP 1: EXTRACTING FRAMES & MASKS ---")

    img_dir, mask_dir = setup_directories(INTERIM_DIR)

    coco = load_coco_data(JSON_PATH)
    frame_map = build_frame_map(coco)

    cap = initialize_video(VIDEO_PATH)

    try:
        count = process_extraction_loop(cap, frame_map, coco, img_dir, mask_dir)
    finally:
        cap.release()

    print(f"Step 1 Complete. Extracted {count} pairs to: {INTERIM_DIR}")
