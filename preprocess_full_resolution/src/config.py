import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESOURCES_DIR = BASE_DIR / "resources"

# ---------------------------------------------------------------------------
# Video naming convention (REQUIRED):
#   camera-{N}_{rat_type}_{time}[_{seq}].mp4
#
#   N        : camera number, e.g. 1, 3, 5, 10, 11, 12
#   rat_type : "albino" or "black_white"
#   time     : "day" or "night"
#   seq      : optional integer suffix when multiple recordings share the
#              same camera/rat_type/time, e.g. camera-3_black_white_night_1.mp4
#
# Examples:
#   camera-1_black_white_day.mp4
#   camera-3_black_white_night_1.mp4
#   camera-5_albino_day.mp4
# ---------------------------------------------------------------------------

_VIDEO_STEM_RE = re.compile(
    r'^camera-(?P<camera>\d+)_(?P<rat_type>albino|black_white)_(?P<time>day|night)(?:_(?P<seq>\d+))?$'
)


def parse_video_stem(stem: str) -> dict:
    """
    Extract camera, rat_type, and time from a video filename stem.
    Raises ValueError immediately if the stem does not match the convention.
    """
    m = _VIDEO_STEM_RE.match(stem)
    if not m:
        raise ValueError(
            f"Video stem '{stem}' does not match the required naming convention:\n"
            f"  camera-{{N}}_{{albino|black_white}}_{{day|night}}[_{{seq}}]\n"
            f"Example: camera-5_albino_night or camera-3_black_white_night_2"
        )
    return {
        "camera":   int(m.group("camera")),
        "rat_type": m.group("rat_type"),
        "time":     m.group("time"),
    }


# Each entry needs only the filenames and which split it belongs to.
# Metadata (camera, rat_type, time) is parsed from the video name at runtime.
# "split": "train" | "val" | "test"
DATA_SAMPLES = [
    # --- Validation (camera-1, camera-3) ---
    {"video": "camera-1_black_white_day.mp4",     "xml": "camera-1_black_white_day.xml",     "split": "val"},
    {"video": "camera-3_black_white_day_1.mp4",   "xml": "camera-3_black_white_day_1.xml",   "split": "val"},
    {"video": "camera-3_black_white_day_2.mp4",   "xml": "camera-3_black_white_day_2.xml",   "split": "val"},
    {"video": "camera-3_black_white_night_1.mp4", "xml": "camera-3_black_white_night_1.xml", "split": "val"},
    {"video": "camera-3_black_white_night_2.mp4", "xml": "camera-3_black_white_night_2.xml", "split": "val"},
    {"video": "camera-3_black_white_night_3.mp4", "xml": "camera-3_black_white_night_3.xml", "split": "val"},

    # --- Train (camera-12 confirmed; camera-5, 10, 11 — add when videos + XMLs arrive) ---
    {"video": "camera-12_black_white_day_1.mp4",  "xml": "camera-12_black_white_day_1.xml",  "split": "train"},
    {"video": "camera-12_black_white_day_2.mp4",  "xml": "camera-12_black_white_day_2.xml",  "split": "train"},
    {"video": "camera-11_black_white_day_2.mp4",  "xml": "camera-11_black_white_day_2.xml", "split": "train"},
    # {"video": "camera-5_TODO_TODO.mp4",  "xml": "camera-5_TODO_TODO.xml",  "split": "train"},
    # {"video": "camera-10_TODO_TODO.mp4", "xml": "camera-10_TODO_TODO.xml", "split": "train"},

    # --- Test — add when videos arrive ---
]

# Validate all entries at import time so a bad filename fails immediately.
for _entry in DATA_SAMPLES:
    parse_video_stem(Path(_entry["video"]).stem)


OUTPUT_DIR   = BASE_DIR / "dataset"
INTERIM_DIR  = OUTPUT_DIR / "interim"
FILTERED_DIR = OUTPUT_DIR / "interim_filtered"

TRAIN_DIR  = OUTPUT_DIR / "train"
VAL_DIR    = OUTPUT_DIR / "val"
TEST_DIR   = OUTPUT_DIR / "test"
STATS_PATH = OUTPUT_DIR / "camera_stats.json"

IOU_THRESHOLD = 0.5
TARGET_SIZE   = 1024

# Set to True to let the pipeline compute the optimal camera→split assignment
# automatically from camera_stats.json (or from rough estimates on first run).
# The "split" fields in DATA_SAMPLES are then ignored and overridden at runtime.
AUTO_SPLIT = False

# Target fraction of frames per split when AUTO_SPLIT is enabled.
SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}

# Base augmented copies per training frame before class-balance scaling.
# Majority class → BASE copies. Minority class → ceil(majority/minority) × BASE copies.
# Example: 20 albino vs 60 black_white, BASE=6 → albino gets 18, black_white gets 6.
BASE_AUGMENT_MULTIPLIER = 6
# Hard cap on augmented copies per frame regardless of class imbalance.
# Prevents low-diversity minority classes (e.g. albino with ~20 unique poses) from being
# augmented so heavily that the model learns transform artifacts rather than real variation.
MAX_AUGMENT_MULTIPLIER = 8
AUGMENTATION_SEED = 42

AUG_PROBS = {
    "horizontal_flip":        0.5,
    "vertical_flip":          0.1,
    "shift_scale_rotate":     0.8,
    "random_brightness_contrast": 0.5,
    "hue_saturation":         0.3,
    "gauss_noise":            0.2,
}
