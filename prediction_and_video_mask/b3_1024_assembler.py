import cv2
import torch
import numpy as np
import os
import sys
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

# --- CONFIGURATION ---
# We use '..' to go up one level from 'prediction_and_video_mask' to the root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "final_rat_model_b3_1024")
INPUT_VIDEO = os.path.join(BASE_DIR, "resources", "test.mp4")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "resources", "test_masked_smooth.mp4")
MAX_FRAMES = 300
# Confidence: Only show mask if model is >50% sure
CONFIDENCE = 0.5

# Smoothing Factor (0.0 to 1.0)
# 0.0 = No smoothing (Jittery)
# 0.5 = Average current frame with previous frame
# 0.7 = Trust new frame mostly, but keep 30% history (Good balance)
SMOOTHING_ALPHA = 0.7


# ---------------------

def main():
    # 1. Validation
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    if not os.path.exists(INPUT_VIDEO):
        print(f"Error: Video not found at {INPUT_VIDEO}")
        return

    # 2. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}...")

    # 3. Load Model
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).to(device)
        processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 4. Open Video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 5. Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print(f"Processing {total_frames} frames...")

    # 6. Processing Loop
    frame_count = 0
    previous_prob_map = None  # Stores the history for smoothing

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_count >= MAX_FRAMES:
            print(f"Reached limit of {MAX_FRAMES} frames. Saving and stopping...")
            break
        # --- STEP 1: LETTERBOX (PAD) TO SQUARE ---
        # We paste the rectangular frame into the center of a black square
        old_h, old_w = frame.shape[:2]
        desired_size = max(old_h, old_w)

        # Create black square canvas
        square_frame = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)

        # Calculate centering offsets
        y_offset = (desired_size - old_h) // 2
        x_offset = (desired_size - old_w) // 2

        # Paste original frame into center
        square_frame[y_offset:y_offset + old_h, x_offset:x_offset + old_w] = frame
        # ------------------------------------------

        # --- STEP 2: INFERENCE ON SQUARE ---
        image_rgb = cv2.cvtColor(square_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

            # Upscale logits to match the SQUARE size (not the video size yet)
            logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=(desired_size, desired_size),
                mode="bilinear",
                align_corners=False
            )
            probs = torch.nn.functional.softmax(logits, dim=1)
            rat_prob = probs[0, 1, :, :]  # Class 1 = Rat

            # Smoothing Logic (Optional)
            if previous_prob_map is None:
                smoothed_prob = rat_prob
            else:
                smoothed_prob = (SMOOTHING_ALPHA * rat_prob) + \
                                ((1 - SMOOTHING_ALPHA) * previous_prob_map)
            previous_prob_map = smoothed_prob

            # Create Mask on the SQUARE canvas
            mask_square = (smoothed_prob > CONFIDENCE).cpu().numpy().astype(np.uint8)

        # --- STEP 3: CROP BACK TO ORIGINAL SIZE ---
        # We only want the mask where the original video was (remove black bars)
        mask_original = mask_square[y_offset:y_offset + old_h, x_offset:x_offset + old_w]

        # --- STEP 4: OVERLAY ---
        green_layer = np.zeros_like(frame, dtype=np.uint8)
        green_layer[:, :] = [0, 255, 0]

        rat_pixels = (mask_original == 1)
        frame[rat_pixels] = cv2.addWeighted(frame[rat_pixels], 0.6, green_layer[rat_pixels], 0.4, 0)

        out.write(frame)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    out.release()
    print(f"Done. Video saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()