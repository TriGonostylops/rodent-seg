import cv2
import numpy as np
import shutil
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from src.config import INTERIM_DIR, FILTERED_DIR, STATS_PATH, IOU_THRESHOLD, DATA_SAMPLES, EMPTY_CAGE_VIDEOS
from src.extract_masks import prepare_stage
from src.stats import compute_camera_stats, aggregate_stats, save_stats, print_all_reports

_STEM_TO_SPLIT = {Path(e["video"]).stem: e["split"] for e in DATA_SAMPLES}
_STEM_TO_SPLIT.update({Path(v).stem: "train" for v in EMPTY_CAGE_VIDEOS})


def _get_camera_stem(mask_path: Path) -> str:
    for video_stem in _STEM_TO_SPLIT:
        if mask_path.stem.startswith(video_stem):
            return video_stem
    return mask_path.stem.rsplit("_frame_", 1)[0]


def calculate_iou(mask1, mask2):
    if mask1 is None or mask2 is None: return 0.0
    m1 = mask1 > 0
    m2 = mask2 > 0
    intersection = np.logical_and(m1, m2).sum()
    union        = np.logical_or(m1, m2).sum()
    return intersection / union if union > 0 else 0.0


def run_filtering():
    print(f"--- STEP 2: FILTERING (IoU < {IOU_THRESHOLD}) ---")

    img_files, in_mask_dir, out_img_dir, out_mask_dir = prepare_stage(INTERIM_DIR, FILTERED_DIR, wipe=True)

    last_saved_mask = None
    current_prefix  = None
    kept_count      = 0
    dropped_count   = 0
    kept_by_camera: dict[str, list[Path]] = defaultdict(list)

    for img_path in tqdm(img_files, desc="Filtering"):
        prefix = img_path.stem.rsplit("_frame_", 1)[0]
        if prefix != current_prefix:
            last_saved_mask = None
            current_prefix  = prefix

        mask_path = in_mask_dir / f"{img_path.stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        should_keep = True
        if last_saved_mask is not None:
            if calculate_iou(last_saved_mask, mask) > IOU_THRESHOLD:
                should_keep = False
                dropped_count += 1

        if should_keep:
            shutil.copy(img_path, out_img_dir / img_path.name)
            out_mask_path = out_mask_dir / mask_path.name
            shutil.copy(mask_path, out_mask_path)
            kept_by_camera[_get_camera_stem(out_mask_path)].append(out_mask_path)
            last_saved_mask = mask
            kept_count += 1

    # compute, save, and report stats
    all_stats = {
        stem: compute_camera_stats(stem, paths, _STEM_TO_SPLIT.get(stem, "unknown"))
        for stem, paths in kept_by_camera.items()
    }
    save_stats(all_stats)

    kept_by_split: dict[str, int] = defaultdict(int)
    for s in all_stats.values():
        kept_by_split[s["split"]] += s["frames"]

    print(f"Step 2 Complete.")
    print(f"Dropped (too similar): {dropped_count}")
    print(f"Kept (diverse):        {kept_count}")
    print(f"  " + "  ".join(f"{split}: {n}" for split, n in sorted(kept_by_split.items())))
    print(f"Camera stats saved to: {STATS_PATH}")

    split_buckets: dict[str, list[dict]] = defaultdict(list)
    for s in all_stats.values():
        split_buckets[s["split"]].append(s)

    for split in ("train", "val", "test"):
        bucket = split_buckets.get(split, [])
        if not bucket:
            continue
        print_all_reports(aggregate_stats(bucket), label=split)
