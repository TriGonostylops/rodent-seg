from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESOURCES_DIR = BASE_DIR / "resources"

DATA_SAMPLES = {
    "01.59.10-02.03.13[M][0@0][0].mp4": "annotations.xml",
    "03.23.31-03.24.39[M][0@0][0].mp4": "03.23.31-03.24.39[M][0@0][0].xml",
    "01.15.05-01.15.36[M][0@0][0].mp4": "01.15.05-01.15.36[M][0@0][0].xml"
}

OUTPUT_DIR = BASE_DIR / "dataset"
INTERIM_DIR = OUTPUT_DIR / "interim"
FILTERED_DIR = OUTPUT_DIR / "interim_filtered"
PROCESSED_DIR = OUTPUT_DIR / "processed"

IOU_THRESHOLD = 0.8
TARGET_SIZE = 1024

AUGMENT_MULTIPLIER = 1
AUGMENTATION_SEED = 42

AUG_PROBS = {
    "horizontal_flip": 0.5,
    "vertical_flip": 0.1,
    "shift_scale_rotate": 0.8,
    "random_brightness_contrast": 0.5,
    "hue_saturation": 0.3,
    "gauss_noise": 0.2
}