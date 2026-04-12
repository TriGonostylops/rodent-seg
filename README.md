# rodent-seg — Dataset Preparation Pipeline

Extracts, filters, and augments annotated video frames into a train/val/test dataset for rodent segmentation.

---

## Pipeline overview

```
resources/  (videos + CVAT XMLs)
     │
     ▼
[1] Extract      — pull keyframes and generate binary masks from CVAT polygons
     │               → dataset/interim/
     ▼
[2] Filter       — drop near-duplicate frames (IoU > threshold)
     │               → dataset/interim_filtered/
     │               prints spatial heatmap of mask centroids
     ▼
[3] Augment      — resize+pad all splits; augment training only
                    → dataset/train/   dataset/val/   dataset/test/
```

Run the full pipeline:
```bash
cd preprocess_full_resolution
python main.py
```

---

## Adding a new video

**1. Name the files using the required convention:**
```
camera-{N}_{rat_type}_{time}[_{seq}].mp4
camera-{N}_{rat_type}_{time}[_{seq}].xml
```
- `N` — camera number (e.g. `5`, `10`, `12`)
- `rat_type` — `albino` or `black_white`
- `time` — `day` or `night`
- `seq` — optional integer when multiple recordings share the same camera/type/time

Examples:
```
camera-5_albino_day.mp4
camera-3_black_white_night_2.mp4
```

Place both files in `resources/`.

**2. Add an entry to `DATA_SAMPLES` in `preprocess_full_resolution/src/config.py`:**
```python
{"video": "camera-5_albino_day.mp4", "xml": "camera-5_albino_day.xml", "split": "train"},
```
`split` must be `"train"`, `"val"`, or `"test"`. The pipeline derives `camera`, `rat_type`, and `time` from the filename automatically — a bad filename raises an error immediately on startup.

**3. Re-run `python main.py`.**

---

## Camera → split assignments

| Split | Cameras |
|-------|---------|
| val   | camera-1, camera-3 |
| train | camera-12 (camera-5, 10, 11 pending) |
| test  | TBD |

Splits are enforced by camera to prevent data leakage — no footage from the same camera appears in more than one split.

---

## Config reference (`src/config.py`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `IOU_THRESHOLD` | `0.5` | Frames with mask IoU above this vs. the previous kept frame are dropped. Lower = stricter diversity requirement. |
| `TARGET_SIZE` | `1024` | All output images are resized to `TARGET_SIZE × TARGET_SIZE`. |
| `BASE_AUGMENT_MULTIPLIER` | `6` | Base number of augmented copies per training frame. The majority rat-type class gets exactly this many; minority classes are scaled up by the class imbalance ratio. |
| `MAX_AUGMENT_MULTIPLIER` | `8` | Hard cap on augmented copies per frame. Prevents small minority classes from being over-augmented with low-diversity transforms. |
| `AUGMENTATION_SEED` | `42` | Seed for reproducibility. |

---

## Output structure

```
dataset/
├── interim/                  # Step 1 output — raw extractions (always wiped on re-run)
├── interim_filtered/         # Step 2 output — after IoU filter
├── train/
│   ├── images/               # originals + augmented copies (_aug_0.jpg, _aug_1.jpg …)
│   └── masks/
├── val/
│   ├── images/               # originals only, resized
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Filenames encode their source: `camera-12_black_white_day_1_frame_000123.jpg`

---

## Reading the spatial heatmap

After step 2 the pipeline prints a 3×3 grid per split showing where mask centroids fall across the frame (top-left = top-left of the camera view):

```
  Spatial distribution [train] — 117 frames  (TOP-LEFT = top-left of frame)

  +---------+---------+---------+
  |    0    |   45    |   40    |
  |  (0%)   |  (38%)  |  (34%)  |
  +---------+---------+---------+
  |    1    |    0    |    3    |
  |  (1%)   |  (0%)   |  (3%)   |
  +---------+---------+---------+
  |   10    |   12    |    6    |
  |  (9%)   |  (10%)  |  (5%)   |
  +---------+---------+---------+

  Spatial distribution [val] — 49 frames  (TOP-LEFT = top-left of frame)

  +---------+---------+---------+
  |   19    |   12    |    1    |
  |  (39%)  |  (24%)  |  (2%)   |
  +---------+---------+---------+
  |    2    |   13    |    0    |
  |  (4%)   |  (27%)  |  (0%)   |
  +---------+---------+---------+
  |    0    |    2    |    0    |
  |  (0%)   |  (4%)   |  (0%)   |
  +---------+---------+---------+
```

- Each split is reported separately — **only the `[train]` report should guide annotation decisions**; val is informational
- A `WARNING` prints if any single cell holds more than 40% of frames
- Cells at 0% or very low % in `[train]` are where new annotations have the most value
- Gaps in `[train]` that reflect real cage behavior (rat never goes there) cannot be fixed by annotation alone

---

## Reading the compactness histogram

Immediately after the spatial heatmap, step 2 prints a compactness distribution.
Compactness = `mask_area / bounding_box_area` — how much of the rat's bounding box is actually filled by the rat.

```
  Compactness [train] — 117 frames  (low = active/extended, high = stationary/compact)

  0.0–0.2                          0 ( 0.0%)  very extended
  0.2–0.4  ████████████████       43 (36.8%)  extended
  0.4–0.6  ████████████████████   53 (45.3%)  mixed
  0.6–0.8  ██████                 16 (13.7%)  compact
  0.8–1.0  █                       5 ( 4.3%)  very compact

  Compactness [val] — 49 frames  (low = active/extended, high = stationary/compact)

  0.0–0.2                          0 ( 0.0%)  very extended
  0.2–0.4  ███████████████        19 (38.8%)  extended
  0.4–0.6  ████████████████████   24 (49.0%)  mixed
  0.6–0.8  █████                   6 (12.2%)  compact
  0.8–1.0                          0 ( 0.0%)  very compact
```

- **Low bins (0.0–0.4):** rat is elongated — running, jumping, stretching
- **High bins (0.6–1.0):** rat is a compact blob — sitting, grooming, sleeping
- A `WARNING` prints if >60% of frames are compact (stationary-heavy dataset)
- A `WARNING` prints if <15% of frames are extended (no active poses)

**Note:** the IoU filter (step 2) naturally biases the surviving frames toward active poses — a stationary rat produces near-identical consecutive masks, which get dropped. So a healthy compactness distribution is partly an emergent consequence of the IoU threshold, not just annotation quality.

---

## Augmentation balancing

Training frames are augmented per `rat_type`. The multiplier for each class is:

```
multiplier = min( ceil(max_count / class_count) × BASE, MAX_AUGMENT_MULTIPLIER )
```

Example with 20 albino / 60 black_white frames and `BASE=6`, `MAX=8`:
- `black_white`: ceil(60/60) × 6 = 6
- `albino`: ceil(60/20) × 6 = 18 → capped to **8**

The pipeline prints `(capped from 18)` when the cap activates, so it is always visible in the run log.

Val and test frames are **never augmented** — only resized and padded. Augmenting evaluation data would make metrics unrepresentative of real deployment conditions.
