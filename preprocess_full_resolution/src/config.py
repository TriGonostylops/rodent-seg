from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RESOURCES_DIR = BASE_DIR / "resources"
VIDEO_PATH = RESOURCES_DIR / "01.59.10-02.03.13[M][0@0][0].mp4"
XML_PATH = RESOURCES_DIR / "annotations.xml"

OUTPUT_DIR = BASE_DIR / "b5-dataset"
INTERIM_DIR = OUTPUT_DIR / "b5-interim"
FILTERED_DIR = OUTPUT_DIR / "b5-interim_filtered"
PROCESSED_DIR = OUTPUT_DIR / "b5-processed"

IOU_THRESHOLD = 0.8
TARGET_SIZE = 640

AUGMENT_MULTIPLIER = 1  # Number of augmented copies per original image
AUGMENTATION_SEED = 42

AUG_PROBS = {
    "horizontal_flip": 0.5,
    "vertical_flip": 0.1,
    "shift_scale_rotate": 0.8,
    "random_brightness_contrast": 0.5,
    "hue_saturation": 0.3,
    "gauss_noise": 0.2
}