import cv2
import json
import numpy as np
from pathlib import Path

from src.config import STATS_PATH, TARGET_SIZE, parse_video_stem

_SPATIAL_ROWS  = 3
_SPATIAL_COLS  = 3
_COMPACT_BINS  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
_AREA_BINS     = [0.0, 2.0, 5.0, 10.0, 20.0, 100.1]  # % of frame area


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_camera_stats(camera_stem: str, mask_paths: list[Path], split: str) -> dict:
    """Compute all statistics for one camera's filtered masks."""
    spatial_grid     = np.zeros((_SPATIAL_ROWS, _SPATIAL_COLS), dtype=int)
    compactness_hist = [0] * (len(_COMPACT_BINS) - 1)
    area_hist        = [0] * (len(_AREA_BINS) - 1)
    area_values      = []
    empty_count      = 0
    padding_bg_pct   = None  # % of padded frame that is black padding (computed once per camera)

    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        h, w   = mask.shape
        binary = mask > 0
        ys, xs = np.where(binary)

        # Compute padding ratio once (same for all frames of this camera)
        if padding_bg_pct is None:
            scale    = TARGET_SIZE / max(h, w)
            scaled_h = round(h * scale)
            scaled_w = round(w * scale)
            visible_area  = scaled_h * scaled_w
            padded_area   = TARGET_SIZE * TARGET_SIZE
            padding_bg_pct = (1.0 - visible_area / padded_area) * 100

        if len(xs) == 0:
            empty_count += 1
            continue

        # spatial centroid
        cy  = int(np.mean(ys))
        cx  = int(np.mean(xs))
        row = min(int(cy / h * _SPATIAL_ROWS), _SPATIAL_ROWS - 1)
        col = min(int(cx / w * _SPATIAL_COLS), _SPATIAL_COLS - 1)
        spatial_grid[row, col] += 1

        # compactness
        mask_area   = int(binary.sum())
        bbox_area   = int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        compactness = mask_area / bbox_area
        for i in range(len(_COMPACT_BINS) - 1):
            if _COMPACT_BINS[i] <= compactness < _COMPACT_BINS[i + 1]:
                compactness_hist[i] += 1
                break

        # mask area as % of padded (TARGET_SIZE × TARGET_SIZE) frame.
        # Raw mask is at original resolution; after LongestMaxSize → PadIfNeeded the
        # rat shrinks relative to the full canvas, so we correct for that here.
        scale        = TARGET_SIZE / max(h, w)
        area_pct     = mask_area * (scale ** 2) / (TARGET_SIZE * TARGET_SIZE) * 100
        area_values.append(area_pct)
        for i in range(len(_AREA_BINS) - 1):
            if _AREA_BINS[i] <= area_pct < _AREA_BINS[i + 1]:
                area_hist[i] += 1
                break

    try:
        meta = parse_video_stem(camera_stem)
    except ValueError:
        meta = {"camera": None, "rat_type": "unknown", "time": "unknown"}

    return {
        "camera":           meta["camera"],
        "rat_type":         meta["rat_type"],
        "time":             meta["time"],
        "split":            split,
        "frames":           len(mask_paths),
        "empty_frames":     empty_count,
        "spatial_grid":     spatial_grid.tolist(),
        "compactness_hist": compactness_hist,
        "area_pct": {
            "hist":        area_hist,
            "mean":        round(float(np.mean(area_values)), 2) if area_values else 0.0,
            "std":         round(float(np.std(area_values)),  2) if area_values else 0.0,
            "padding_bg":  round(padding_bg_pct, 1) if padding_bg_pct is not None else 0.0,
        },
    }


def aggregate_stats(camera_stats_list: list[dict]) -> dict:
    """Merge multiple per-camera stat dicts into one for split-level reporting."""
    merged_spatial     = np.zeros((_SPATIAL_ROWS, _SPATIAL_COLS), dtype=int)
    merged_compactness = [0] * 5
    merged_area_hist   = [0] * 5
    area_means, area_stds, area_frames = [], [], []
    padding_bg_values  = []
    empty = 0

    for s in camera_stats_list:
        merged_spatial     += np.array(s["spatial_grid"])
        merged_compactness  = [a + b for a, b in zip(merged_compactness, s["compactness_hist"])]
        merged_area_hist    = [a + b for a, b in zip(merged_area_hist,   s["area_pct"]["hist"])]
        n = s["frames"] - s["empty_frames"]
        if n > 0:
            area_means.append(s["area_pct"]["mean"])
            area_stds.append(s["area_pct"]["std"])
            area_frames.append(n)
        padding_bg_values.append(s["area_pct"].get("padding_bg", 0.0))
        empty += s["empty_frames"]

    if area_frames:
        w_mean = sum(m * n for m, n in zip(area_means, area_frames)) / sum(area_frames)
        w_std  = float(np.sqrt(sum(sd**2 * n for sd, n in zip(area_stds, area_frames)) / sum(area_frames)))
    else:
        w_mean, w_std = 0.0, 0.0

    # Use the max padding across cameras so the warning is conservative
    max_padding_bg = max(padding_bg_values) if padding_bg_values else 0.0

    return {
        "frames":           sum(s["frames"] for s in camera_stats_list),
        "empty_frames":     empty,
        "spatial_grid":     merged_spatial.tolist(),
        "compactness_hist": merged_compactness,
        "area_pct":         {
            "hist":       merged_area_hist,
            "mean":       round(w_mean, 2),
            "std":        round(w_std, 2),
            "padding_bg": round(max_padding_bg, 1),
        },
    }


def save_stats(all_stats: dict[str, dict]):
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump(all_stats, f, indent=2)


def load_stats() -> dict[str, dict]:
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"Camera stats not found at {STATS_PATH}. Run the pipeline first.")
    with open(STATS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def print_spatial_report(stats: dict, label: str):
    grid  = np.array(stats["spatial_grid"])
    total = int(grid.sum())
    if total == 0:
        print(f"  [{label}] No foreground masks.")
        return

    cell_w = 9
    border = "  +" + (("-" * cell_w + "+") * _SPATIAL_COLS)
    print(f"\n  Spatial distribution [{label}] — {total} frames  (TOP-LEFT = top-left of frame)\n")
    print(border)
    for row in range(_SPATIAL_ROWS):
        counts = "|".join(f"{grid[row, col]:^{cell_w}}" for col in range(_SPATIAL_COLS))
        pcts   = "|".join(f"{'(' + str(round(grid[row,col]/total*100)) + '%)':^{cell_w}}" for col in range(_SPATIAL_COLS))
        print(f"  |{counts}|")
        print(f"  |{pcts}|")
        print(border)

    if stats["empty_frames"]:
        print(f"  ({stats['empty_frames']} empty-cage masks excluded from grid)")

    hot     = np.unravel_index(np.argmax(grid), grid.shape)
    hot_pct = grid[hot] / total * 100
    if hot_pct > 40:
        print(f"  WARNING: cell ({hot[0]},{hot[1]}) holds {hot_pct:.0f}% of frames — possible spatial bias.")


def print_compactness_report(stats: dict, label: str):
    BIN_LABELS = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    BIN_HINTS  = ["very extended", "extended", "mixed", "compact", "very compact"]
    counts = stats["compactness_hist"]
    total  = sum(counts)
    if total == 0:
        print(f"  [{label}] No foreground masks.")
        return

    bar_max   = 20
    max_count = max(counts) if max(counts) > 0 else 1
    print(f"\n  Compactness [{label}] — {total} frames  (low = active/extended, high = stationary/compact)\n")
    for i, (lbl, hint) in enumerate(zip(BIN_LABELS, BIN_HINTS)):
        pct = counts[i] / total * 100
        bar = "█" * int(counts[i] / max_count * bar_max)
        print(f"  {lbl}  {bar:<{bar_max}}  {counts[i]:>3} ({pct:>4.1f}%)  {hint}")

    compact_pct = sum(counts[3:]) / total * 100
    active_pct  = sum(counts[:2]) / total * 100
    if compact_pct > 60:
        print(f"  WARNING: {compact_pct:.0f}% of frames are compact — dataset may be stationary-heavy.")
    if active_pct < 15:
        print(f"  WARNING: only {active_pct:.0f}% of frames are extended — consider annotating more active poses.")


def print_area_report(stats: dict, label: str):
    BIN_LABELS = ["0–2%  ", "2–5%  ", "5–10% ", "10–20%", "20%+  "]
    BIN_HINTS  = ["very small", "small", "medium", "large", "very large"]
    counts     = stats["area_pct"]["hist"]
    total      = sum(counts)
    if total == 0:
        return

    bar_max    = 20
    max_count  = max(counts) if max(counts) > 0 else 1
    mean, std  = stats["area_pct"]["mean"], stats["area_pct"]["std"]
    padding_bg = stats["area_pct"].get("padding_bg", 0.0)

    note = ""
    if padding_bg > 0:
        note = f"  (NOTE: {padding_bg:.1f}% of each padded frame is black background from aspect-ratio padding)"

    print(f"\n  Mask area [{label}] — {total} frames  "
          f"(rat as % of {TARGET_SIZE}×{TARGET_SIZE} padded frame, mean={mean:.1f}% ±{std:.1f}%)")
    if note:
        print(f"  {note}")
    print()
    for i, (lbl, hint) in enumerate(zip(BIN_LABELS, BIN_HINTS)):
        pct = counts[i] / total * 100
        bar = "█" * int(counts[i] / max_count * bar_max)
        print(f"  {lbl}  {bar:<{bar_max}}  {counts[i]:>3} ({pct:>4.1f}%)  {hint}")


def print_all_reports(stats: dict, label: str):
    print_spatial_report(stats,     label=label)
    print_compactness_report(stats, label=label)
    print_area_report(stats,        label=label)
