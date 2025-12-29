#%%
import cv2
from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pyplot as plt

run = r"C:\Users\gimes\Src\yoloruns\yolo11s-seg"

model_path = rf"{run}\weights\best.pt"

model = YOLO(model_path, task="segment")

test_dir = r"C:\Users\gimes\Src\preproc\gonca\yolo-patkany\test"
test_ims = [f"{test_dir}\\{f}" for f in os.listdir(test_dir) if f.endswith(".png")]

# %%
# results = model(test_ims[0:100])
# results[0].show()
# %%
# --- 3. Play images as video ---
for idx, img_path in enumerate(test_ims):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # Run segmentation
    results = model(img)
    result = results[0]
    
    # result.show()
    
    print(f"There is a result mask: {result.masks is not None}")
    
    # Overlay segmentation masks
    if result.masks:
        
        for mask in result.masks.data:
            mask = cv2.resize(mask.cpu().numpy().astype(np.uint8), (w, h)) * 255
            # mask = mask.astype(np.uint8) * 255
            color_mask = np.zeros_like(img)
            color_mask[:, :, 1] = mask  # green mask overlay
            img = cv2.addWeighted(img, 1, color_mask, 0.5, 0)
    
    # Add frame index on top corner
    cv2.putText(img, f"Frame: {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display frame
    cv2.imshow("YOLO Segmentation Video", img)
    
    # Wait a bit (adjust FPS)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# %%
