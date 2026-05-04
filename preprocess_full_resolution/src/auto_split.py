"""
Automatic camera-to-split assignment.

Groups all DATA_SAMPLES entries by camera number, then finds the
camera→split assignment that minimises a weighted cost:

  cost = w_ratio  × frame-count-ratio deviation from SPLIT_RATIOS
       + w_train  × penalty when train doesn't have the most frames
       + w_cov    × coverage penalty (val/test missing rat_types/times seen in train)
       + w_dist   × distribution divergence (spatial/compactness/area)
                    only when camera_stats.json is available

Test split uses a one-sided floor: the optimizer only penalises falling *below*
SPLIT_RATIOS["test"]; exceeding it is free.  This makes test naturally land on
the smallest camera(s) that satisfy the floor — i.e. minimum frames for test.

Brute-forces all assignments — feasible up to ~10 cameras (3^10 = 59 049).
Requires train + val + test each to be represented once ≥3 cameras are present.

Entry point: apply_auto_split()
  Mutates DATA_SAMPLES[i]["split"] in place.
  Must be called BEFORE filter_masks and augment are imported (their module-level
  _STEM_TO_SPLIT dicts are computed at import time from DATA_SAMPLES).
"""

from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

from src.config import DATA_SAMPLES, SPLIT_RATIOS, parse_video_stem
from src.stats import load_stats


# ---------------------------------------------------------------------------
# Camera grouping & feature extraction
# ---------------------------------------------------------------------------

def _group_by_camera(samples: list[dict]) -> dict[int, list[dict]]:
    """Return {camera_number: [entry, ...]} grouped from DATA_SAMPLES."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for entry in samples:
        stem = Path(entry["video"]).stem
        meta = parse_video_stem(stem)
        groups[meta["camera"]].append(entry)
    return dict(groups)


def _camera_features(cam_id: int, entries: list[dict], stats: dict) -> dict:
    """
    Build a feature dict for one camera.

    When camera_stats.json is available the frame count and distribution
    histograms come from measured data; otherwise frame count is estimated
    at 30 frames per video (a rough placeholder that preserves relative scale).
    """
    rat_types: set[str] = set()
    times: set[str]     = set()
    frames = 0

    for entry in entries:
        stem = Path(entry["video"]).stem
        meta = parse_video_stem(stem)
        rat_types.add(meta["rat_type"])
        times.add(meta["time"])
        frames += stats[stem]["frames"] if (stats and stem in stats) else 30

    # Aggregate normalised distribution vectors for this camera
    spatial     = None
    compactness = None
    area_hist   = None

    if stats:
        sp = np.zeros(9,  dtype=float)
        cp = np.zeros(5,  dtype=float)
        ar = np.zeros(5,  dtype=float)
        n_found = 0
        for entry in entries:
            stem = Path(entry["video"]).stem
            if stem not in stats:
                continue
            s = stats[stem]
            sp += np.array(s["spatial_grid"]).flatten()
            cp += np.array(s["compactness_hist"], dtype=float)
            ar += np.array(s["area_pct"]["hist"],  dtype=float)
            n_found += 1

        if n_found:
            def _norm(v: np.ndarray) -> np.ndarray:
                return v / v.sum() if v.sum() > 0 else v
            spatial     = _norm(sp)
            compactness = _norm(cp)
            area_hist   = _norm(ar)

    return {
        "cam_id":      cam_id,
        "rat_types":   rat_types,
        "times":       times,
        "frames":      frames,
        "spatial":     spatial,
        "compactness": compactness,
        "area_hist":   area_hist,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence — 0 = identical distributions, 1 = maximally different."""
    p = p + 1e-9;  p /= p.sum()
    q = q + 1e-9;  q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def _avg_normed(vecs: list[np.ndarray]) -> np.ndarray | None:
    if not vecs:
        return None
    avg = np.array(vecs).mean(axis=0)
    return avg / avg.sum() if avg.sum() > 0 else avg


def _score_assignment(
    features:      dict[int, dict],
    assignment:    dict[int, str],
    target_ratios: dict[str, float],
) -> float:
    """
    Compute the cost of a camera→split assignment.  Lower is better.

    Weights are intentionally asymmetric:
      - Frame ratio deviation is the primary signal (squared error)
      - Train-dominance and coverage penalties are hard bumps
      - Distribution divergence is a soft tie-breaker
    """
    split_frames:    dict[str, int]       = defaultdict(int)
    split_rat_types: dict[str, set[str]]  = defaultdict(set)
    split_times:     dict[str, set[str]]  = defaultdict(set)
    split_sp:        dict[str, list]      = defaultdict(list)
    split_cp:        dict[str, list]      = defaultdict(list)
    split_ar:        dict[str, list]      = defaultdict(list)

    for cam_id, split in assignment.items():
        feat = features[cam_id]
        split_frames[split]    += feat["frames"]
        split_rat_types[split].update(feat["rat_types"])
        split_times[split].update(feat["times"])
        if feat["spatial"]     is not None: split_sp[split].append(feat["spatial"])
        if feat["compactness"] is not None: split_cp[split].append(feat["compactness"])
        if feat["area_hist"]   is not None: split_ar[split].append(feat["area_hist"])

    total = sum(split_frames.values())
    if total == 0:
        return float("inf")

    cost = 0.0

    # 1. Frame-count ratio deviation
    #    train/val: two-sided squared error — stay close to the target.
    #    test: one-sided floor — only penalise falling *below* the minimum;
    #          exceeding it is free, so the optimizer picks the smallest
    #          camera(s) that satisfy the floor (minimum frames for test).
    for split, target in target_ratios.items():
        actual = split_frames.get(split, 0) / total
        if split == "test":
            if actual < target:
                cost += (actual - target) ** 2
        else:
            cost += (actual - target) ** 2

    # 2. Train must have the most frames
    if split_frames.get("train", 0) < max(
        split_frames.get("val", 0), split_frames.get("test", 0)
    ):
        cost += 5.0

    # 3. Coverage: val/test should cover the same rat_types & times as train
    #    (so evaluation isn't blind to a condition the model was trained on)
    train_types = split_rat_types.get("train", set())
    train_times = split_times.get("train", set())
    for split in ("val", "test"):
        if split_frames.get(split, 0) == 0:
            continue
        cost += 0.5 * len(train_types - split_rat_types.get(split, set()))
        cost += 0.3 * len(train_times - split_times.get(split, set()))

    # 4. Distribution similarity: val/test vs train (soft tie-breaker)
    train_sp_v = _avg_normed(split_sp.get("train", []))
    train_cp_v = _avg_normed(split_cp.get("train", []))
    train_ar_v = _avg_normed(split_ar.get("train", []))

    for split in ("val", "test"):
        if split_frames.get(split, 0) == 0:
            continue
        if train_sp_v is not None:
            ev = _avg_normed(split_sp.get(split, []))
            if ev is not None:
                cost += 0.4 * _js_divergence(train_sp_v.copy(), ev.copy())
        if train_cp_v is not None:
            ev = _avg_normed(split_cp.get(split, []))
            if ev is not None:
                cost += 0.3 * _js_divergence(train_cp_v.copy(), ev.copy())
        if train_ar_v is not None:
            ev = _avg_normed(split_ar.get(split, []))
            if ev is not None:
                cost += 0.3 * _js_divergence(train_ar_v.copy(), ev.copy())

    return cost


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _find_best_assignment(
    features:      dict[int, dict],
    target_ratios: dict[str, float],
) -> tuple[dict[int, str], float]:
    """
    Brute-force over all camera→split combos.
    Only considers splits that have a non-zero target ratio.
    """
    cam_ids = sorted(features.keys())
    viable  = [s for s in ("train", "val", "test") if target_ratios.get(s, 0) > 0]

    best_cost       = float("inf")
    best_assignment = {cam_id: "train" for cam_id in cam_ids}

    for combo in product(viable, repeat=len(cam_ids)):
        assignment = dict(zip(cam_ids, combo))
        if "train" not in assignment.values():
            continue
        if len(cam_ids) >= 2 and "val" not in assignment.values():
            continue
        if len(cam_ids) >= 3 and "test" in viable and "test" not in assignment.values():
            continue
        cost = _score_assignment(features, assignment, target_ratios)
        if cost < best_cost:
            best_cost       = cost
            best_assignment = assignment

    return best_assignment, best_cost


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_report(
    assignment:    dict[int, str],
    features:      dict[int, dict],
    target_ratios: dict[str, float],
    stats_loaded:  bool,
):
    split_frames: dict[str, int] = defaultdict(int)
    for cam_id, split in assignment.items():
        split_frames[split] += features[cam_id]["frames"]
    total = sum(split_frames.values())

    print("\n  Camera assignments:")
    for cam_id in sorted(assignment):
        feat  = features[cam_id]
        split = assignment[cam_id]
        types = "+".join(sorted(feat["rat_types"]))
        times = "+".join(sorted(feat["times"]))
        est   = "  (estimated)" if not stats_loaded else ""
        print(f"    camera-{cam_id:<4} → {split:<6}  "
              f"{types}/{times}  {feat['frames']} frames{est}")

    print("\n  Split distribution:")
    for split in ("train", "val", "test"):
        n      = split_frames.get(split, 0)
        actual = n / total * 100 if total else 0
        target = target_ratios.get(split, 0) * 100
        bar    = "█" * int(actual / 5)
        status = "(empty — not enough cameras)" if n == 0 else ""
        print(f"    {split:<6}  {bar:<20}  {n:>4} frames ({actual:>4.1f}%)  "
              f"target {target:.0f}%  {status}")

    # Coverage check
    split_rat_types: dict[str, set] = defaultdict(set)
    split_times:     dict[str, set] = defaultdict(set)
    for cam_id, split in assignment.items():
        split_rat_types[split].update(features[cam_id]["rat_types"])
        split_times[split].update(features[cam_id]["times"])
    train_types = split_rat_types.get("train", set())
    train_times = split_times.get("train", set())

    print("\n  Coverage vs train:")
    for split in ("val", "test"):
        if split_frames.get(split, 0) == 0:
            continue
        missing_types = train_types - split_rat_types.get(split, set())
        missing_times = train_times - split_times.get(split, set())
        type_str = (f"missing rat_type: {', '.join(sorted(missing_types))}"
                    if missing_types else "rat_types OK")
        time_str = (f"missing time: {', '.join(sorted(missing_times))}"
                    if missing_times else "times OK")
        print(f"    {split:<6}  {type_str}  |  {time_str}")

    if not stats_loaded:
        print("\n  NOTE: camera_stats.json not found — frame counts are estimates (30/video).")
        print("        Re-run after a first full pipeline pass for accurate frame counts.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def suggest_and_print() -> dict[str, str]:
    """
    Compute the best assignment, print a report, and return
    {video_stem: split} for every entry in DATA_SAMPLES.
    """
    try:
        stats        = load_stats()
        stats_loaded = True
    except FileNotFoundError:
        stats        = {}
        stats_loaded = False

    groups   = _group_by_camera(DATA_SAMPLES)
    features = {
        cam_id: _camera_features(cam_id, entries, stats)
        for cam_id, entries in groups.items()
    }

    assignment, _ = _find_best_assignment(features, SPLIT_RATIOS)

    print("--- AUTO-SPLIT ---")
    _print_report(assignment, features, SPLIT_RATIOS, stats_loaded)
    print()

    stem_to_split: dict[str, str] = {}
    for cam_id, entries in groups.items():
        split = assignment[cam_id]
        for entry in entries:
            stem_to_split[Path(entry["video"]).stem] = split

    return stem_to_split


def apply_auto_split():
    """
    Compute the best split assignment and mutate DATA_SAMPLES[i]["split"] in place.

    IMPORTANT: call this before importing filter_masks or augment — both modules
    compute _STEM_TO_SPLIT at import time from DATA_SAMPLES.
    """
    stem_to_split = suggest_and_print()
    for entry in DATA_SAMPLES:
        stem = Path(entry["video"]).stem
        if stem in stem_to_split:
            entry["split"] = stem_to_split[stem]
