import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
from scipy.spatial.distance import directed_hausdorff

SCOUT_MODEL_PATH = "/kaggle/input/datasets/gonoszgonosz/b2-1024-weights/final_rat_model_b3_1024"
FINE_MODEL_PATH  = "/kaggle/working/final_rat_model_magnified"   # <--- UPDATE to magnified model path

INPUT_VIDEOS = [
    "/kaggle/input/datasets/gonoszgonosz/rat-test-video/test.mp4",
    "/kaggle/input/datasets/gonoszgonosz/rat-test-video/test2.mp4",
    "/kaggle/input/datasets/gonoszgonosz/rat-test-video/test3.mp4",
]
OUTPUT_VIDEOS = [
    "/kaggle/working/coarse_to_fine_b3_test1.mp4",
    "/kaggle/working/coarse_to_fine_b3_test2.mp4",
    "/kaggle/working/coarse_to_fine_b3_test3.mp4",
]

TEST_IMG_DIR  = "/kaggle/input/YOUR_TEST_DATASET/test/images"   # <--- UPDATE
TEST_MASK_DIR = "/kaggle/input/YOUR_TEST_DATASET/test/masks"    # <--- UPDATE
OUTPUT_CSV    = "/kaggle/working/b3_1024_metrics.csv"

CONFIDENCE_SCOUT  = 0.3
CONFIDENCE_FINE   = 0.6
MARGIN            = 60
BOUNDARY_DILATION = 7

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_mask_from_tensor(outputs, target_size, confidence):
    logits = torch.nn.functional.interpolate(
        outputs.logits, size=target_size, mode="bilinear", align_corners=False
    )
    probs = torch.nn.functional.softmax(logits, dim=1)
    return (probs[0, 1, :, :] > confidence).cpu().numpy().astype(np.uint8)


def predict_frame(scout_model, scout_proc, fine_model, fine_proc, frame):
    """2-pass coarse-to-fine inference. Returns full-resolution binary mask."""
    height, width = frame.shape[:2]
    final_mask = np.zeros((height, width), dtype=np.uint8)

    pil_full = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        coarse_mask = get_mask_from_tensor(
            scout_model(**scout_proc(images=pil_full, return_tensors="pt").to(device)),
            (height, width), CONFIDENCE_SCOUT
        )

    cnts, _ = cv2.findContours(coarse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return final_mask

    c = max(cnts, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(c)
    x1 = max(0, x - MARGIN);          y1 = max(0, y - MARGIN)
    x2 = min(width, x + w_box + MARGIN); y2 = min(height, y + h_box + MARGIN)

    crop = frame[y1:y2, x1:x2]
    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        fine_mask = get_mask_from_tensor(
            fine_model(**fine_proc(images=pil_crop, return_tensors="pt").to(device)),
            (y2 - y1, x2 - x1), CONFIDENCE_FINE
        )

    fine_cnts, _ = cv2.findContours(fine_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(fine_cnts) > 0:
        c_fine = max(fine_cnts, key=cv2.contourArea)
        clean = np.zeros_like(fine_mask)
        cv2.drawContours(clean, [c_fine], -1, 1, thickness=cv2.FILLED)
        final_mask[y1:y2, x1:x2] = clean

    return final_mask


def process_video(scout_model, scout_proc, fine_model, fine_proc, input_video, output_video):
    if not os.path.exists(input_video):
        print(f"  SKIPPED (not found): {input_video}")
        return

    cap = cv2.VideoCapture(input_video)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {width}x{height} | {fps:.1f} FPS | {total} frames")

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        mask = predict_frame(scout_model, scout_proc, fine_model, fine_proc, frame)

        green = np.zeros_like(frame)
        green[mask == 1] = [0, 255, 0]
        frame = cv2.addWeighted(frame, 0.7, green, 0.3, 0)
        out.write(frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"    {frame_count}/{total} frames...")

    cap.release()
    out.release()
    print(f"  Saved: {output_video}")


# --- Metric helpers ---
def calculate_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0: return 1.0 if inter == 0 else 0.0
    return inter / union

def calculate_dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0: return 1.0
    return (2. * inter) / total

def calculate_boundary_iou(pred, gt, dilation=BOUNDARY_DILATION):
    kernel = np.ones((dilation, dilation), dtype=np.uint8)
    gt_b   = cv2.morphologyEx(gt.astype(np.uint8),   cv2.MORPH_GRADIENT, kernel) > 0
    pred_b = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    inter  = np.logical_and(pred_b, gt_b).sum()
    union  = np.logical_or(pred_b, gt_b).sum()
    if union == 0: return 1.0 if inter == 0 else 0.0
    return inter / union

def calculate_hausdorff(pred, gt):
    pred_pts = np.argwhere(cv2.Canny((pred.astype(np.uint8) * 255), 0, 1) > 0)
    gt_pts   = np.argwhere(cv2.Canny((gt.astype(np.uint8)   * 255), 0, 1) > 0)
    if len(pred_pts) == 0 or len(gt_pts) == 0: return np.nan
    return max(directed_hausdorff(pred_pts, gt_pts)[0],
               directed_hausdorff(gt_pts, pred_pts)[0])


def evaluate_test_set(scout_model, scout_proc, fine_model, fine_proc):
    img_map  = {os.path.splitext(f)[0]: f for f in os.listdir(TEST_IMG_DIR)  if f.endswith(('.jpg', '.png'))}
    mask_map = {os.path.splitext(f)[0]: f for f in os.listdir(TEST_MASK_DIR) if f.endswith(('.jpg', '.png'))}
    common_ids = sorted(set(img_map.keys()) & set(mask_map.keys()))
    print(f"  Evaluating on {len(common_ids)} test pairs...")

    rows = []
    for cid in tqdm(common_ids, desc="B3-1024 + magnified eval"):
        image   = cv2.imread(os.path.join(TEST_IMG_DIR,  img_map[cid]))
        gt_gray = cv2.imread(os.path.join(TEST_MASK_DIR, mask_map[cid]), cv2.IMREAD_GRAYSCALE)
        gt      = np.where(gt_gray > 0, 1, 0).astype(bool)
        pred    = predict_frame(scout_model, scout_proc, fine_model, fine_proc, image).astype(bool)
        rows.append({
            "Frame_ID"          : cid,
            "mIoU"              : calculate_iou(pred, gt),
            "Dice"              : calculate_dice(pred, gt),
            "Boundary_IoU"      : calculate_boundary_iou(pred, gt),
            "Hausdorff_Distance": calculate_hausdorff(pred, gt),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 50)
    print(" B3-1024 TEST SET EVALUATION")
    print("=" * 50)
    print(f" Frames          : {len(df)}")
    print(f" Mean mIoU       : {df['mIoU'].mean():.4f}")
    print(f" Mean Dice       : {df['Dice'].mean():.4f}")
    print(f" Mean B-IoU      : {df['Boundary_IoU'].mean():.4f}")
    print(f" Mean Hausdorff  : {df['Hausdorff_Distance'].mean():.2f} px")
    print(f" Saved to        : {OUTPUT_CSV}")
    print("=" * 50)


def main():
    print(f"--- B3-1024 + MAGNIFIED 2-PASS TRACKER | Device: {device.upper()} ---")

    print("Loading scout model (B3-1024)...")
    scout_proc  = SegformerImageProcessor.from_pretrained(SCOUT_MODEL_PATH)
    scout_model = SegformerForSemanticSegmentation.from_pretrained(SCOUT_MODEL_PATH).to(device).eval()

    print("Loading fine model (magnified)...")
    fine_proc  = SegformerImageProcessor.from_pretrained(FINE_MODEL_PATH)
    fine_model = SegformerForSemanticSegmentation.from_pretrained(FINE_MODEL_PATH).to(device).eval()

    for input_video, output_video in zip(INPUT_VIDEOS, OUTPUT_VIDEOS):
        print(f"\n--- VIDEO: {input_video} ---")
        process_video(scout_model, scout_proc, fine_model, fine_proc, input_video, output_video)

    print("\n--- TEST SET EVALUATION ---")
    evaluate_test_set(scout_model, scout_proc, fine_model, fine_proc)


if __name__ == "__main__":
    main()
