import cv2
import torch
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

MODEL_PATH = "/kaggle/input/datasets/gonoszgonosz/b2-1024-weights/final_rat_model_b3_1024"
INPUT_VIDEO = "/kaggle/input/datasets/gonoszgonosz/rat-test-video/test.mp4"
OUTPUT_VIDEO = "coarse_to_fine_b3_test.mp4"

CONFIDENCE = 0.5
MARGIN = 40  # Breathing room for the tight crop


def get_mask_from_tensor(outputs, target_size, confidence):
    logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=target_size,
        mode="bilinear",
        align_corners=False
    )
    probs = torch.nn.functional.softmax(logits, dim=1)
    mask = (probs[0, 1, :, :] > confidence).cpu().numpy().astype(np.uint8)
    return mask


def main():
    print("--- INITIALIZING B3 2-PASS TRACKER ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).to(device)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        final_full_mask = np.zeros((height, width), dtype=np.uint8)

        pil_full = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs_coarse = processor(images=pil_full, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs_coarse = model(**inputs_coarse)
            coarse_mask = get_mask_from_tensor(outputs_coarse, (height, width), CONFIDENCE)

        cnts, _ = cv2.findContours(coarse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(c)

            x1 = max(0, x - MARGIN)
            y1 = max(0, y - MARGIN)
            x2 = min(width, x + w_box + MARGIN)
            y2 = min(height, y + h_box + MARGIN)

            crop_img = frame[y1:y2, x1:x2]
            pil_crop = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

            inputs_fine = processor(images=pil_crop, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs_fine = model(**inputs_fine)
                crop_h, crop_w = crop_img.shape[:2]
                fine_mask = get_mask_from_tensor(outputs_fine, (crop_h, crop_w), CONFIDENCE)

            fine_cnts, _ = cv2.findContours(fine_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(fine_cnts) > 0:
                c_fine = max(fine_cnts, key=cv2.contourArea)
                clean_mask = np.zeros_like(fine_mask)
                cv2.drawContours(clean_mask, [c_fine], -1, 1, thickness=cv2.FILLED)

                final_full_mask[y1:y2, x1:x2] = clean_mask

        green_layer = np.zeros_like(frame, dtype=np.uint8)
        green_layer[:, :] = [0, 255, 0]

        rat_pixels = (final_full_mask == 1)

        blended = cv2.addWeighted(frame, 0.6, green_layer, 0.4, 0)

        frame[rat_pixels] = blended[rat_pixels]

        out.write(frame)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("DONE! Video saved.")


if __name__ == "__main__":
    main()