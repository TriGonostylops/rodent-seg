from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RESOURCES_DIR = BASE_DIR / "resources"
VIDEO_PATH = RESOURCES_DIR / "01.59.10-02.03.13[M][0@0][0].dav"
JSON_PATH = RESOURCES_DIR / "instances_default.json"

OUTPUT_DIR = BASE_DIR / "dataset"
INTERIM_DIR = OUTPUT_DIR / "interim"
PROCESSED_DIR = OUTPUT_DIR / "processed"

IOU_THRESHOLD = 0.8
TARGET_SIZE = 1024