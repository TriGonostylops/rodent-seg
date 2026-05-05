import cv2
import math
import random
import numpy as np
import albumentations as A
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from src.config import (
    DATA_SAMPLES, FILTERED_DIR, EMPTY_CAGE_VIDEOS,
    TRAIN_DIR, VAL_DIR, TEST_DIR, GENERALIST_DIR,
    BASE_AUGMENT_MULTIPLIER, MAX_AUGMENT_MULTIPLIER, AUGMENTATION_SEED, AUG_PROBS, TARGET_SIZE,
    parse_video_stem,
)
from src.extract_masks import setup_directories

_STEM_TO_SPLIT = {Path(e["video"]).stem: e["split"] for e in DATA_SAMPLES}
_STEM_TO_SPLIT.update({Path(v).stem: "train" for v in EMPTY_CAGE_VIDEOS})
_EMPTY_STEMS   = {Path(v).stem for v in EMPTY_CAGE_VIDEOS}

SPLIT_DIRS = {
    "train": TRAIN_DIR,
    "val":   VAL_DIR,
    "test":  TEST_DIR,
}


def get_meta_for_file(img_path: Path) -> dict | None:
    """
    Resolve a frame filename back to its full metadata.
    Split comes from config; camera/rat_type/time are parsed from the video stem.
    Returns None if the file cannot be matched to any config entry.
    """
    for video_stem, split in _STEM_TO_SPLIT.items():
        if img_path.stem.startswith(video_stem):
            if video_stem in _EMPTY_STEMS:
                return {"split": "train", "rat_type": None, "camera": None, "time": None}
            parsed = parse_video_stem(video_stem)  # always valid — checked at import
            return {**parsed, "split": split}
    return None


def compute_multipliers(train_files: list[Path], base: int) -> dict[str, int]:
    """
    Count training frames per rat_type, then assign augmentation multipliers
    so that after augmentation every class has roughly the same number of frames.

    majority class  → base copies
    minority class  → ceil(majority_count / minority_count) * base copies
    """
    counts: dict[str, int] = defaultdict(int)
    for f in train_files:
        meta = get_meta_for_file(f)
        if meta and meta["rat_type"] is not None:
            counts[meta["rat_type"]] += 1

    if not counts:
        return {}

    max_count = max(counts.values())
    multipliers = {
        rat_type: min(math.ceil(max_count / cnt) * base, MAX_AUGMENT_MULTIPLIER)
        for rat_type, cnt in counts.items()
    }

    print("  Class distribution (training):")
    for rat_type, cnt in counts.items():
        raw = math.ceil(max_count / cnt) * base
        capped = multipliers[rat_type]
        cap_note = f" (capped from {raw})" if capped < raw else ""
        print(f"    {rat_type}: {cnt} frames → {capped} augmented copies each{cap_note}")

    return multipliers


def get_augmentor() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=AUG_PROBS["horizontal_flip"]),
        A.VerticalFlip(p=AUG_PROBS["vertical_flip"]),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.7, 1.4),
            rotate=(-90, 90),
            p=AUG_PROBS["shift_scale_rotate"],
        ),
        A.RandomBrightnessContrast(p=AUG_PROBS["random_brightness_contrast"]),
        A.RandomGamma(gamma_limit=(70, 130), p=AUG_PROBS["random_gamma"]),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,
            p=AUG_PROBS.get("hue_saturation", 0.3),
        ),
        A.GaussNoise(std_range=(0.1, 0.3), p=AUG_PROBS.get("gauss_noise", 0.2)),
        A.LongestMaxSize(max_size=TARGET_SIZE),
        A.PadIfNeeded(min_height=TARGET_SIZE, min_width=TARGET_SIZE, border_mode=cv2.BORDER_CONSTANT),
    ], is_check_shapes=False)


def get_resizer() -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=TARGET_SIZE),
        A.PadIfNeeded(min_height=TARGET_SIZE, min_width=TARGET_SIZE, border_mode=cv2.BORDER_CONSTANT),
    ], is_check_shapes=False)


def run_augmentation():
    print("--- STEP 3: AUGMENTATION & RESIZING ---")

    if AUGMENTATION_SEED is not None:
        random.seed(AUGMENTATION_SEED)
        np.random.seed(AUGMENTATION_SEED)

    in_img_dir  = FILTERED_DIR / "images"
    in_mask_dir = FILTERED_DIR / "masks"
    if not in_img_dir.exists():
        raise FileNotFoundError(f"Filtered data not found at {FILTERED_DIR}. Run step 2 first.")
    img_files = sorted(in_img_dir.glob("*.jpg"))

    # Group files by their split
    split_files: dict[str, list[Path]] = defaultdict(list)
    unknown = []
    for f in img_files:
        meta = get_meta_for_file(f)
        if meta:
            split_files[meta["split"]].append(f)
        else:
            unknown.append(f)

    if unknown:
        print(f"  WARNING: {len(unknown)} file(s) could not be matched to a DATA_SAMPLES entry — skipped.")

    # Compute per-rat_type augmentation multipliers from training set
    multipliers = compute_multipliers(split_files.get("train", []), BASE_AUGMENT_MULTIPLIER)

    augmentor = get_augmentor()
    resizer   = get_resizer()

    total_written = 0

    for split, files in split_files.items():
        if not files:
            continue

        out_dir = SPLIT_DIRS.get(split)
        if out_dir is None:
            print(f"  WARNING: unknown split '{split}', skipping {len(files)} file(s).")
            continue

        out_img_dir, out_mask_dir, _ = setup_directories(out_dir, wipe=True)
        print(f"  Processing [{split}]: {len(files)} source frames → {out_dir}")

        for img_path in tqdm(files, desc=f"[{split}]"):
            mask_path = in_mask_dir / f"{img_path.stem}.png"
            image = cv2.imread(str(img_path))
            mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                continue

            # Always save the resized original
            resized = resizer(image=image, mask=mask)
            cv2.imwrite(str(out_img_dir / f"{img_path.stem}.jpg"), resized["image"])
            cv2.imwrite(str(out_mask_dir / f"{img_path.stem}.png"), resized["mask"])
            total_written += 1

            # Augmentation only for training frames
            if split == "train":
                meta  = get_meta_for_file(img_path)
                n_aug = multipliers.get(meta["rat_type"], BASE_AUGMENT_MULTIPLIER) if meta else BASE_AUGMENT_MULTIPLIER
                for i in range(n_aug):
                    aug = augmentor(image=image, mask=mask)
                    cv2.imwrite(str(out_img_dir / f"{img_path.stem}_aug_{i}.jpg"), aug["image"])
                    cv2.imwrite(str(out_mask_dir / f"{img_path.stem}_aug_{i}.png"), aug["mask"])
                    total_written += 1

    print(f"Step 3 Complete. Total frames written: {total_written}")
    for split, out_dir in SPLIT_DIRS.items():
        img_dir = out_dir / "images"
        if img_dir.exists():
            n = len(list(img_dir.glob("*.jpg")))
            print(f"  dataset/{split}/images: {n} files")


def run_generalist_export():
    """
    Step 4: Export every filtered frame (all splits, all cameras) into
    dataset/generalist/ with resize-only — no augmentation.
    Used to evaluate zero-shot generalist models on the full annotated set.
    """
    print("--- STEP 4: GENERALIST DATASET EXPORT ---")

    in_img_dir  = FILTERED_DIR / "images"
    in_mask_dir = FILTERED_DIR / "masks"
    if not in_img_dir.exists():
        raise FileNotFoundError(f"Filtered data not found at {FILTERED_DIR}. Run steps 1-2 first.")

    out_img_dir  = GENERALIST_DIR / "images"
    out_mask_dir = GENERALIST_DIR / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    resizer   = get_resizer()
    img_files = sorted(in_img_dir.glob("*.jpg"))
    total     = 0

    for img_path in tqdm(img_files, desc="[generalist]"):
        mask_path = in_mask_dir / f"{img_path.stem}.png"
        image = cv2.imread(str(img_path))
        mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        resized = resizer(image=image, mask=mask)
        cv2.imwrite(str(out_img_dir  / f"{img_path.stem}.jpg"), resized["image"])
        cv2.imwrite(str(out_mask_dir / f"{img_path.stem}.png"), resized["mask"])
        total += 1

    print(f"Step 4 Complete. Generalist frames written: {total}")
    print(f"  dataset/generalist/images: {total} files")
