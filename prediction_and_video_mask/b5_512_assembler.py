!pip
install - q
transformers

import cv2
import torch
import numpy as np
import os
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

MODEL_PATH = "/kaggle/working/final_rat_model_b5_512"
INPUT_VIDEO = "/kaggle/input/YOUR_VIDEO_DATASET/test.mp4"  # <--- UPDATE THIS
OUTPUT_VIDEO = "/kaggle/working/b5_512_rat_tracking.mp4"

TARGET_SIZE = 512
CONFIDENCE = 0.5
SMOOTHING_ALPHA = 0.7

def main():
    print(f"--- STARTING RAT TRACKER (b5 @ {TARGET_SIZE}px) ---")

    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Model path not found: {MODEL_PATH}")
        return
    if not os.path.exists(INPUT_VIDEO):
        print(f"CRITICAL ERROR: Video path not found: {INPUT_VIDEO}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Acceleration: {device.upper()}")

    try:
        print("Loading Model...")
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).to(device)
        processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)
        print("Model Loaded Successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(INPUT_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print(f"Processing Video: {width}x{height} | {total_frames} Frames")

    frame_count = 0
    previous_prob_map = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        old_h, old_w = frame.shape[:2]
        desired_size = max(old_h, old_w)

        square_frame = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)
        y_offset = (desired_size - old_h) // 2
        x_offset = (desired_size - old_w) // 2
        square_frame[y_offset:y_offset + old_h, x_offset:x_offset + old_w] = frame

        image_rgb = cv2.cvtColor(square_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = processor(images=pil_image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

            logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=(desired_size, desired_size),
                mode="bilinear",
                align_corners=False
            )
            probs = torch.nn.functional.softmax(logits, dim=1)
            rat_prob = probs[0, 1, :, :]

            if previous_prob_map is None:
                smoothed_prob = rat_prob
            else:
                smoothed_prob = (SMOOTHING_ALPHA * rat_prob) + \
                                ((1 - SMOOTHING_ALPHA) * previous_prob_map)
            previous_prob_map = smoothed_prob

            mask_square = (smoothed_prob > CONFIDENCE).cpu().numpy().astype(np.uint8)

        mask_original = mask_square[y_offset:y_offset + old_h, x_offset:x_offset + old_w]

        cnts, _ = cv2.findContours(mask_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            clean_mask = np.zeros_like(mask_original)
            cv2.drawContours(clean_mask, [c], -1, 1, thickness=cv2.FILLED)
            mask_original = clean_mask
        else:
            mask_original = np.zeros_like(mask_original)

        green_layer = np.zeros_like(frame, dtype=np.uint8)
        green_layer[:, :] = [0, 255, 0]
        rat_pixels = (mask_original == 1)
        frame[rat_pixels] = cv2.addWeighted(frame[rat_pixels], 0.6, green_layer[rat_pixels], 0.4, 0)

        out.write(frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"   Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"\nDONE! Video saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()