import shutil
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path
from typing import Any

# Cleaned up imports: Removed VIDEO_PATH and XML_PATH
from src.config import RESOURCES_DIR, INTERIM_DIR, DATA_SAMPLES


def setup_directories(base_dir, wipe=True):
    img_dir = base_dir / "images"
    mask_dir = base_dir / "masks"

    if base_dir.exists():
        if not wipe:
            has_images = any(img_dir.iterdir()) if img_dir.exists() else False
            has_masks = any(mask_dir.iterdir()) if mask_dir.exists() else False
            if has_images and has_masks:
                print(f"[{base_dir.name}] Data found. Skipping generation (Cache Hit).")
                return img_dir, mask_dir, True
        shutil.rmtree(base_dir)

    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, mask_dir, False


def load_xml_annotations(xml_path):
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found at: {xml_path}")

    print(f"Loading XML: {xml_path.name}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    frame_map = {}

    for track in root.findall('track'):
        label = track.get('label')
        for polygon in track.findall('polygon'):
            if polygon.get('keyframe') == '1' and polygon.get('outside') == '0':
                frame_idx = int(polygon.get('frame'))
                points_str = polygon.get('points')
                points = []
                for p in points_str.split(';'):
                    if ',' in p:
                        points.append([float(coord) for coord in p.split(',')])

                if frame_idx not in frame_map:
                    frame_map[frame_idx] = []
                frame_map[frame_idx].append({
                    'label': label,
                    'points': np.array(points, dtype=np.int32)
                })
    if not frame_map:
        raise ValueError("No keyframes found in XML!")
    return frame_map


def initialize_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"CRITICAL: Could not open video at {video_path}.")
    return cap


def generate_binary_mask(annotations, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        cv2.fillPoly(mask, [ann['points']], 1)
    return mask


def process_extraction_loop(cap, frame_map, img_dir, mask_dir, video_prefix):
    saved_count = 0
    for frame_idx in tqdm(sorted(frame_map.keys()), desc=f"Extracting {video_prefix}"):
        annotations = frame_map[frame_idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: break

        h, w = frame.shape[:2]
        mask = generate_binary_mask(annotations, h, w)

        # Unique naming to prevent collisions across multiple videos
        name = f"{video_prefix}_frame_{frame_idx:06d}"
        cv2.imwrite(str(img_dir / f"{name}.jpg"), frame)
        cv2.imwrite(str(mask_dir / f"{name}.png"), mask * 255)
        saved_count += 1
    return saved_count


def run_extraction():
    print(f"--- STEP 1: MULTI-VIDEO EXTRACTION ---")

    # Initialize the target directories first
    img_dir, mask_dir, skipped = setup_directories(INTERIM_DIR, wipe=True)

    if skipped:
        print("Step 1 skipped (Cache Hit).")
        return

    total_extracted = 0
    for video_name, xml_name in DATA_SAMPLES.items():
        v_path = RESOURCES_DIR / video_name
        x_path = RESOURCES_DIR / xml_name

        if not v_path.exists() or not x_path.exists():
            print(f"Skipping {video_name}: File not found.")
            continue

        cap = initialize_video(v_path)
        frame_map = load_xml_annotations(x_path)
        video_prefix = v_path.stem

        count = process_extraction_loop(cap, frame_map, img_dir, mask_dir, video_prefix)
        total_extracted += count
        cap.release()

    print(f"Step 1 Complete. Total unique pairs: {total_extracted}")


def prepare_stage(in_dir: Path, out_dir: Path) -> tuple[list[Path], Path, Any, Any]:
    in_img_dir = in_dir / "images"
    in_mask_dir = in_dir / "masks"
    if not in_img_dir.exists():
        raise FileNotFoundError(f"Source data not found at {in_dir}.")
    out_img_dir, out_mask_dir, _ = setup_directories(out_dir, wipe=False)
    img_files = sorted(list(in_img_dir.glob("*.jpg")))
    return img_files, in_mask_dir, out_img_dir, out_mask_dir