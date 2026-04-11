import cv2
import numpy as np
import shutil
from tqdm import tqdm

from src.config import INTERIM_DIR, FILTERED_DIR, IOU_THRESHOLD
from src.extract_masks import prepare_stage


def calculate_iou(mask1, mask2):
    if mask1 is None or mask2 is None: return 0.0

    m1 = mask1 > 0
    m2 = mask2 > 0

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    return intersection / union if union > 0 else 0.0


def report_spatial_distribution(kept_mask_paths, grid_rows=3, grid_cols=3):
    """Print an ASCII heatmap of mask centroid positions across the frame."""
    grid = np.zeros((grid_rows, grid_cols), dtype=int)
    empty_count = 0

    for mask_path in kept_mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        h, w = mask.shape
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            empty_count += 1
            continue
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        row = min(int(cy / h * grid_rows), grid_rows - 1)
        col = min(int(cx / w * grid_cols), grid_cols - 1)
        grid[row, col] += 1

    total = grid.sum()
    if total == 0:
        print("  No masks with foreground pixels found.")
        return

    cell_w = 9
    border = "  +" + (("-" * cell_w + "+") * grid_cols)

    print(f"\n  Spatial distribution — mask centroids ({grid_rows}×{grid_cols} grid, {total} frames):")
    print(f"  TOP-LEFT = top-left of frame\n")
    print(border)
    for row in range(grid_rows):
        counts = "|".join(f"{grid[row, col]:^{cell_w}}" for col in range(grid_cols))
        pcts   = "|".join(f"{'(' + str(round(grid[row,col]/total*100)) + '%)':^{cell_w}}" for col in range(grid_cols))
        print(f"  |{counts}|")
        print(f"  |{pcts}|")
        print(border)

    if empty_count:
        print(f"  ({empty_count} empty-cage masks excluded from grid)")

    # Warn if any single cell exceeds 40% of frames
    hot = np.unravel_index(np.argmax(grid), grid.shape)
    hot_pct = grid[hot] / total * 100
    if hot_pct > 40:
        print(f"\n  WARNING: cell ({hot[0]},{hot[1]}) holds {hot_pct:.0f}% of frames — possible spatial bias.")


def report_compactness(kept_mask_paths):
    """
    Print an ASCII histogram of mask compactness (mask_area / bounding_box_area).

    Low compactness  (~0.0–0.4) → elongated/extended rat: running, jumping, stretching
    High compactness (~0.6–1.0) → compact blob: sitting, grooming, sleeping
    """
    BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    BIN_LABELS = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    BIN_HINTS  = ["very extended", "extended", "mixed", "compact", "very compact"]
    counts = [0] * (len(BINS) - 1)
    empty_count = 0

    for mask_path in kept_mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        binary = mask > 0
        ys, xs = np.where(binary)
        if len(xs) == 0:
            empty_count += 1
            continue

        mask_area = binary.sum()
        bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        compactness = mask_area / bbox_area

        for i in range(len(BINS) - 1):
            if BINS[i] <= compactness < BINS[i + 1]:
                counts[i] += 1
                break

    total = sum(counts)
    if total == 0:
        print("  No foreground masks found.")
        return

    bar_max = 20
    max_count = max(counts) if max(counts) > 0 else 1

    print(f"\n  Compactness distribution — mask_area / bounding_box_area ({total} frames):")
    print(f"  Low = extended/active   High = compact/stationary\n")
    for i, (label, hint) in enumerate(zip(BIN_LABELS, BIN_HINTS)):
        pct = counts[i] / total * 100
        bar = "█" * int(counts[i] / max_count * bar_max)
        print(f"  {label}  {bar:<{bar_max}}  {counts[i]:>3} ({pct:>4.1f}%)  {hint}")

    if empty_count:
        print(f"  ({empty_count} empty-cage masks excluded)")

    # Warn if over 60% of frames are compact (sitting/grooming heavy)
    compact_pct = sum(counts[3:]) / total * 100
    if compact_pct > 60:
        print(f"\n  WARNING: {compact_pct:.0f}% of frames are compact — dataset may be stationary-heavy.")
    active_pct = sum(counts[:2]) / total * 100
    if active_pct < 15:
        print(f"\n  WARNING: only {active_pct:.0f}% of frames are extended — consider annotating more active poses.")


def run_filtering():
    print(f"--- STEP 2: FILTERING (IoU < {IOU_THRESHOLD}) ---")

    img_files, in_mask_dir, out_img_dir, out_mask_dir = prepare_stage(INTERIM_DIR, FILTERED_DIR, wipe=True)

    last_saved_mask = None
    current_prefix = None
    kept_count = 0
    dropped_count = 0
    kept_mask_paths = []

    for img_path in tqdm(img_files, desc="Filtering"):
        prefix = img_path.stem.rsplit('_frame_', 1)[0]
        if prefix != current_prefix:
            last_saved_mask = None
            current_prefix = prefix
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
            out_mask_path = out_mask_dir / mask_path.name
            shutil.copy(mask_path, out_mask_path)
            kept_mask_paths.append(out_mask_path)
            last_saved_mask = mask
            kept_count += 1

    print(f"Step 2 Complete.")
    print(f"Dropped (too similar): {dropped_count}")
    print(f"Kept (diverse):        {kept_count}")
    print(f"Clean data ready in:   {FILTERED_DIR}")

    report_spatial_distribution(kept_mask_paths)
    report_compactness(kept_mask_paths)
