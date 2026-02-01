import cv2
import shutil
import random
import numpy as np
import albumentations as A
from tqdm import tqdm
from src.config import FILTERED_DIR, PROCESSED_DIR, AUGMENT_MULTIPLIER, AUGMENTATION_SEED, AUG_PROBS, TARGET_SIZE

from src.extract_masks import setup_directories, prepare_stage


def get_augmentor():
    return A.Compose([
        A.HorizontalFlip(p=AUG_PROBS["horizontal_flip"]),
        A.VerticalFlip(p=AUG_PROBS["vertical_flip"]),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.8, 1.2),
            rotate=(-15, 15),
            p=AUG_PROBS["shift_scale_rotate"]
        ),
        A.RandomBrightnessContrast(p=AUG_PROBS["random_brightness_contrast"]),
        A.LongestMaxSize(max_size=TARGET_SIZE),
        A.PadIfNeeded(min_height=TARGET_SIZE, min_width=TARGET_SIZE, border_mode=cv2.BORDER_CONSTANT),
    ], is_check_shapes=False)


def run_augmentation():
    print(f"--- STEP 3: AUGMENTATION & RESIZING ---")

    if AUGMENTATION_SEED is not None:
        random.seed(AUGMENTATION_SEED)
        np.random.seed(AUGMENTATION_SEED)

    img_files, in_mask_dir, out_img_dir, out_mask_dir = prepare_stage(FILTERED_DIR, PROCESSED_DIR)

    print(f"Processing {len(img_files)} frames (Multiplier: {AUGMENT_MULTIPLIER})...")

    augmentor = get_augmentor()
    # Simple resizer for original images if they don't meet target size
    resizer = A.Compose([
        A.LongestMaxSize(max_size=TARGET_SIZE),
        A.PadIfNeeded(min_height=TARGET_SIZE, min_width=TARGET_SIZE, border_mode=cv2.BORDER_CONSTANT),
    ], is_check_shapes=False)

    count = 0

    for img_path in tqdm(img_files, desc="Augmenting"):
        mask_path = in_mask_dir / f"{img_path.stem}.png"
        image = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            continue

        base_name = img_path.stem

        # Save resized original
        resized = resizer(image=image, mask=mask)
        cv2.imwrite(str(out_img_dir / f"{base_name}.jpg"), resized['image'])
        cv2.imwrite(str(out_mask_dir / f"{base_name}.png"), resized['mask'])

        # Generate multiple augmentations
        for i in range(AUGMENT_MULTIPLIER):
            suffix = f"_aug_{i}" if AUGMENT_MULTIPLIER > 1 else "_aug"
            aug = augmentor(image=image, mask=mask)
            cv2.imwrite(str(out_img_dir / f"{base_name}{suffix}.jpg"), aug['image'])
            cv2.imwrite(str(out_mask_dir / f"{base_name}{suffix}.png"), aug['mask'])

        count += 1

    print(f"Step 3 Complete. Final Dataset ready in: {PROCESSED_DIR}")