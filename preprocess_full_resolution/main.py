import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import step1_extract

def main():
    print("=== RODENT SEGMENTATION PIPELINE STARTED ===")

    try:
        step1_extract.run_extraction()
    except Exception as e:
        print(f"CRITICAL ERROR in Step 1: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # try:
    #     step2_augment.run_augmentation()
    # except Exception as e:
    #     print(f"CRITICAL ERROR in Step 2: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     sys.exit(1)

    print("\n=== PIPELINE FINISHED ===")
    print("Data is ready in '../dataset/processed'")


if __name__ == "__main__":
    main()
