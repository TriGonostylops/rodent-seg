import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# AUTO_SPLIT must run before filter_masks and augment are imported:
# both modules compute _STEM_TO_SPLIT at module-level from DATA_SAMPLES,
# so the split assignments must already be mutated in DATA_SAMPLES by then.
from src.config import AUTO_SPLIT
if AUTO_SPLIT:
    from src.auto_split import apply_auto_split
    apply_auto_split()

from src import extract_masks
from src import filter_masks
from src import augment


def main():
    print("=== RODENT SEGMENTATION PIPELINE STARTED ===")

    try:
        extract_masks.run_extraction()
    except Exception as e:
        print(f"CRITICAL ERROR in Step 1: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        filter_masks.run_filtering()
    except Exception as e:
        print(f"CRITICAL ERROR in Step 2 (Filtering): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        augment.run_augmentation()
    except Exception as e:
        print(f"CRITICAL ERROR in Step 3 (Augmentation): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        augment.run_generalist_export()
    except Exception as e:
        print(f"CRITICAL ERROR in Step 4 (Generalist Export): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n=== PIPELINE FINISHED ===")
    print("Data is ready in: dataset/train/  dataset/val/  dataset/test/  dataset/generalist/")


if __name__ == "__main__":
    main()
