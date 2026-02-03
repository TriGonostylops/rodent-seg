import shutil

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

from pathlib import Path
from typing import Any
from src.config import VIDEO_PATH, XML_PATH, INTERIM_DIR


def setup_directories(base_dir, wipe=True):

    img_dir = base_dir / "images"
    mask_dir = base_dir / "masks"

    if base_dir.exists():
        if not wipe:
            has_images = any(img_dir.iterdir()) if img_dir.exists() else False
            has_masks = any(mask_dir.iterdir()) if mask_dir.exists() else False

            if has_images and has_masks:
                print(f"[{base_dir.name}] Data found. Skipping generation (Cache Hit).")
                return img_dir, mask_dir, True  # True = We skipped

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
            # Only process keyframes that are not "outside"
            if polygon.get('keyframe') == '1' and polygon.get('outside') == '0':
                frame_idx = int(polygon.get('frame'))
                points_str = polygon.get('points')
                
                # Parse points: "x1,y1;x2,y2;..." -> [[x1,y1], [x2,y2], ...]
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

    print(f"Opening Video: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise IOError(f"CRITICAL: Could not open video at {video_path}. \n"
                      f"Check codec installation or path.")
    return cap


def generate_binary_mask(annotations, height, width):

    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        points = ann['points']
        cv2.fillPoly(mask, [points], 1)
    return mask


def process_extraction_loop(cap, frame_map, img_dir, mask_dir):
    saved_count = 0

    sorted_frames = sorted(frame_map.keys())

    for frame_idx in tqdm(sorted_frames, desc="Extracting All Annotated"):
        annotations = frame_map[frame_idx]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"Error reading frame {frame_idx}. Stopping extraction.")
            break

        h, w = frame.shape[:2]
        mask = generate_binary_mask(annotations, h, w)

        name = f"frame_{frame_idx:06d}"
        cv2.imwrite(str(img_dir / f"{name}.jpg"), frame)
        cv2.imwrite(str(mask_dir / f"{name}.png"), mask * 255)

        saved_count += 1

    return saved_count

def prepare_stage(in_dir: Path, out_dir: Path) -> tuple[list[Path], Path, Any, Any]:
    in_img_dir = in_dir / "images"
    in_mask_dir = in_dir / "masks"

    if not in_img_dir.exists():
        raise FileNotFoundError(f"Source data not found at {in_dir}.")

    out_img_dir, out_mask_dir, _ = setup_directories(out_dir, wipe=False)

    img_files = sorted(list(in_img_dir.glob("*.jpg")))
    print(f"Scanning {len(img_files)} frames from {in_dir.name}...")
    return img_files, in_mask_dir, out_img_dir, out_mask_dir


def run_extraction():
    print(f"--- STEP 1: EXTRACTING FRAMES & MASKS ---")

    img_dir, mask_dir, skipped = setup_directories(INTERIM_DIR, wipe=False)

    if skipped:
        print("Step 1 skipped (Data already ready). Moving to Step 2.")
        return

    frame_map = load_xml_annotations(XML_PATH)
    cap = initialize_video(VIDEO_PATH)

    try:
        count = process_extraction_loop(cap, frame_map, img_dir, mask_dir)
    finally:
        cap.release()

    print(f"Step 1 Complete. Extracted {count} pairs to: {INTERIM_DIR}")
