import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

    print("\n=== PIPELINE FINISHED ===")
    print("Data is ready in '../dataset/processed'")


if __name__ == "__main__":
    main()
