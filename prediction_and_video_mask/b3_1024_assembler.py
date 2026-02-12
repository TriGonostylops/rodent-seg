!pip
install - q
transformers

import cv2
import torch
import numpy as np
import os
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image


MODEL_PATH = "/kaggle/input/datasets/gonoszgonosz/b2-1024-weights/final_rat_model_b3_1024"

INPUT_VIDEO = "/kaggle/input/rat-test-video/test.mp4"

OUTPUT_VIDEO = "/kaggle/working/final_rat_tracking.mp4"

CONFIDENCE = 0.5  # Ignore weak predictions
SMOOTHING_ALPHA = 0.7  # Higher = Smoother mask (less flickering)

def main():
    print(f"--- STARTING RAT TRACKER ---")

    if not os.path.exists(MODEL_PATH):
        print(f" CRITICAL ERROR: Model path not found: {MODEL_PATH}")
        return
    if not os.path.exists(INPUT_VIDEO):
        print(f" CRITICAL ERROR: Video path not found: {INPUT_VIDEO}")
        return

    # 2. Setup Device (GPU Check)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Acceleration: {device.upper()}")
    if device == "cpu":
        print("️ WARNING: Running on CPU. This will be slow! Enable GPU in Settings.")

    # 3. Load Model
    try:
        print(" Loading Model...")
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).to(device)
        processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)
        print(" Model Loaded Successfully")
    except Exception as e:
        print(f" Error loading model: {e}")
        return

    # 4. Open Video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print(f"Processing Video: {width}x{height} | {total_frames} Frames")

    # 5. Processing Loop
    frame_count = 0
    previous_prob_map = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- STEP 1: LETTERBOX (PAD) TO SQUARE ---
        # Ensures the rat isn't squashed, matching training data
        old_h, old_w = frame.shape[:2]
        desired_size = max(old_h, old_w)

        # Create black square canvas
        square_frame = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)

        # Calculate centering offsets
        y_offset = (desired_size - old_h) // 2
        x_offset = (desired_size - old_w) // 2

        # Paste original frame into center
        square_frame[y_offset:y_offset + old_h, x_offset:x_offset + old_w] = frame

        # --- STEP 2: INFERENCE ---
        image_rgb = cv2.cvtColor(square_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

            # Resize logits to match the SQUARE size
            logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=(desired_size, desired_size),
                mode="bilinear",
                align_corners=False
            )
            probs = torch.nn.functional.softmax(logits, dim=1)
            rat_prob = probs[0, 1, :, :]  # Class 1 = Rat

            # --- STEP 3: SMOOTHING ---
            if previous_prob_map is None:
                smoothed_prob = rat_prob
            else:
                smoothed_prob = (SMOOTHING_ALPHA * rat_prob) + \
                                ((1 - SMOOTHING_ALPHA) * previous_prob_map)
            previous_prob_map = smoothed_prob

            # Create Mask
            mask_square = (smoothed_prob > CONFIDENCE).cpu().numpy().astype(np.uint8)

        # --- STEP 4: CROP & CLEAN ---
        # Crop back to original video size (remove black bars)
        mask_original = mask_square[y_offset:y_offset + old_h, x_offset:x_offset + old_w]

        # HIGHLANDER RULE: Keep only the largest blob
        cnts, _ = cv2.findContours(mask_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            clean_mask = np.zeros_like(mask_original)
            cv2.drawContours(clean_mask, [c], -1, 1, thickness=cv2.FILLED)
            mask_original = clean_mask
        else:
            # If no rat found, mask is empty
            mask_original = np.zeros_like(mask_original)

        # --- STEP 5: OVERLAY ---
        green_layer = np.zeros_like(frame, dtype=np.uint8)
        green_layer[:, :] = [0, 255, 0]

        rat_pixels = (mask_original == 1)
        # Blend: 60% Original + 40% Green
        frame[rat_pixels] = cv2.addWeighted(frame[rat_pixels], 0.6, green_layer[rat_pixels], 0.4, 0)

        out.write(frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"   Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"\n DONE! Video saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()