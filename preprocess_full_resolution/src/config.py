from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RESOURCES_DIR = BASE_DIR / "resources"
VIDEO_PATH = RESOURCES_DIR / "01.59.10-02.03.13[M][0@0][0].mp4"
XML_PATH = RESOURCES_DIR / "annotations.xml"

OUTPUT_DIR = BASE_DIR / "dataset"
INTERIM_DIR = OUTPUT_DIR / "interim"
FILTERED_DIR = OUTPUT_DIR / "interim_filtered"
PROCESSED_DIR = OUTPUT_DIR / "processed"

IOU_THRESHOLD = 0.8
TARGET_SIZE = 1024

# Augmentation Settings
AUGMENT_MULTIPLIER = 1  # Number of augmented copies per original image
AUGMENTATION_SEED = 42

AUG_PROBS = {
    "horizontal_flip": 0.5,
    "vertical_flip": 0.1,
    "shift_scale_rotate": 0.8,
    "random_brightness_contrast": 0.2
}