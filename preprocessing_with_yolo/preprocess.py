import cv2
import numpy as np
import os
import glob
from pathlib import Path
import albumentations as A

def get_rat_center(yolo_line, img_width, img_height):
    parts = list(map(float, yolo_line.strip().split()))
    if not parts:
        return None
    # YOLO segmentation format: class x1 y1 x2 y2 ...
    coords = parts[1:]
    xs = coords[0::2]
    ys = coords[1::2]
    
    if not xs or not ys:
        return None
    
    # Simple average of all polygon points to find center
    center_x = sum(xs) / len(xs) * img_width
    center_y = sum(ys) / len(ys) * img_height
    return int(center_x), int(center_y)

def create_mask(yolo_line, img_width, img_height):
    parts = list(map(float, yolo_line.strip().split()))
    if not parts:
        return None
    coords = parts[1:]
    points = []
    for i in range(0, len(coords), 2):
        points.append([coords[i] * img_width, coords[i+1] * img_height])
    
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    if points:
        poly = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)
    return mask

def crop_centered(image, mask, center, crop_size=512):
    cx, cy = center
    h, w = image.shape[:2]
    
    half_size = crop_size // 2
    
    x1 = max(0, cx - half_size)
    y1 = max(0, cy - half_size)
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    
    if x2 > w:
        x2 = w
        x1 = max(0, x2 - crop_size)
    if y2 > h:
        y2 = h
        y1 = max(0, y2 - crop_size)
        
    # Final check if it's still smaller than crop_size (in case image is smaller than 512)
    # But usually videos are larger.
    
    crop_img = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    
    # If smaller than 512, pad it
    if crop_img.shape[0] < crop_size or crop_img.shape[1] < crop_size:
        pad_h = max(0, crop_size - crop_img.shape[0])
        pad_w = max(0, crop_size - crop_img.shape[1])
        crop_img = cv2.copyMakeBorder(crop_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        crop_mask = cv2.copyMakeBorder(crop_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        
    return crop_img, crop_mask

def get_augmentation():
    return A.Compose([
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5), # Jitter
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
    ])

def preprocess_and_augment():
    video_path = r"preprocessing_with_yolo\dav\01.59.10-02.03.13[M][0@0][0].dav"
    labels_dir = r"preprocessing_with_yolo\dataset\task_1_ultralytics yolo segmentation 1.0\labels\train"
    output_dir = r"dataset\train"
    
    images_out = os.path.join(output_dir, "images")
    masks_out = os.path.join(output_dir, "masks")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(masks_out, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {frame_count} frames.")

    augmentor = get_augmentation()
    
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    label_files.sort()

    # Filter to only annotated frames (up to frame 3826)
    label_files = [f for f in label_files if int(Path(f).stem.split('_')[1]) <= 3826]
    print(f"Total annotated frames found: {len(label_files)}")

    # Limit to first 1 frame for testing the flow
    # label_files = label_files[:1]
    
    total_to_process = len(label_files)
    processed_count = 0
    
    for label_file in label_files:
        processed_count += 1
        frame_idx = int(Path(label_file).stem.split('_')[1])
        base_name = f"frame_{frame_idx:06d}"
        
        # Check if BOTH original and augmented exist
        if os.path.exists(os.path.join(images_out, f"{base_name}.jpg")) and \
           os.path.exists(os.path.join(images_out, f"{base_name}_aug.jpg")):
            if processed_count % 100 == 0:
                print(f"Progress: {processed_count}/{total_to_process} (Skipped {processed_count} frames)", end="\r")
            continue
            
        print(f"Progress: {processed_count}/{total_to_process} - Processing frame {frame_idx}...", end="\r")
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame {frame_idx}")
            continue
            
        h, w = frame.shape[:2]
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            continue
            
        # Assuming one rat per frame for simplicity, or taking the first one
        yolo_line = lines[0]
        
        center = get_rat_center(yolo_line, w, h)
        mask = create_mask(yolo_line, w, h)
        
        if center and mask is not None:
            crop_img, crop_mask = crop_centered(frame, mask, center)
            
            # Save original crop
            base_name = f"frame_{frame_idx:06d}"
            img_out_path = os.path.join(images_out, f"{base_name}.jpg")
            mask_out_path = os.path.join(masks_out, f"{base_name}.png")
            aug_img_out_path = os.path.join(images_out, f"{base_name}_aug.jpg")
            aug_mask_out_path = os.path.join(masks_out, f"{base_name}_aug.png")

            if not os.path.exists(img_out_path) or not os.path.exists(mask_out_path):
                cv2.imwrite(img_out_path, crop_img)
                cv2.imwrite(mask_out_path, crop_mask)
            
            if not os.path.exists(aug_img_out_path) or not os.path.exists(aug_mask_out_path):
                # Apply augmentation
                augmented = augmentor(image=crop_img, mask=crop_mask)
                aug_img = augmented['image']
                aug_mask = augmented['mask']
                
                cv2.imwrite(aug_img_out_path, aug_img)
                cv2.imwrite(aug_mask_out_path, aug_mask)
            
            if frame_idx % 100 == 0:
                print(f"Processed frame {frame_idx}")

    cap.release()
    print("Preprocessing and augmentation finished.")

if __name__ == "__main__":
    preprocess_and_augment()
