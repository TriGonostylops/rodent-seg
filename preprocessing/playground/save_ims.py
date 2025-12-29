# %%
import cv2
import xml.etree.ElementTree as ET

import numpy as np
import random
import shutil
import os

# CHANGE PATHS HERE
root_dir = f"C:\\Users\\gimes\\Src\\preproc\\gonca\\"


cap = cv2.VideoCapture(f"{root_dir}01.59.10-02.03.13[M][0@0][0].dav")
annotations_xml = f"{root_dir}annotations.xml"
print("Video capture status:", cap.isOpened())
if cap.isOpened():
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:")
    print(f"  Total frames: {frame_count}")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {frame_count/fps:.2f} seconds" if fps > 0 else "  Duration: Unknown")



tree = ET.parse(annotations_xml)
root = tree.getroot()

# Extract frame annotations
annotations = {}
polygons = root.findall("track")[0].findall("polygon")
polygon_idx = [int(polygon.get("frame")) for polygon in polygons]
polygon_keyframe = [int(polygon.get("keyframe")) for polygon in polygons]
polygon_points = [polygon.get("points") for polygon in polygons]

# Find the highest frame value for polygons with keyframe 1
highest_keyframe_1_frame = max([polygon_idx[i] for i in range(len(polygon_keyframe)) if polygon_keyframe[i] == 1])
effective_frames = [frame for frame in annotations.keys() if frame <= highest_keyframe_1_frame]
frames = np.unique([polygon.get("frame") for polygon in polygons])

print(f"Highest frame value for keyframe 1 polygons: {highest_keyframe_1_frame}")
print(f"Total effective frames with annotations: {len(effective_frames)}")
print(f"Total frames in annotations: {len(frames)}")

DATASPLIT = 0.85

# Create a random 80-20 split of the polygon indices
random.seed(42)  # For reproducible results
shuffled_indices = polygon_idx[:highest_keyframe_1_frame].copy()
random.shuffle(shuffled_indices)

split_point = int(DATASPLIT * len(shuffled_indices))
train_indices = shuffled_indices[:split_point]
val_indices = shuffled_indices[split_point:]

print(f"Train set: {len(train_indices)} frames")
print(f"Validation set: {len(val_indices)} frames")

print(f"Total annotations found: {len(polygon_idx)}")
for idx, keyframe, points in zip(polygon_idx, polygon_keyframe, polygon_points):
    annotations[idx] = {
        "keyframe": keyframe,
        "points": points
    }


#%%
# Save all frames into an array
frames = []


train_img_dir = f"{root_dir}yolo-patkany\\images\\train\\"
train_labels_dir = f"{root_dir}yolo-patkany\\labels\\train\\"
val_img_dir = f"{root_dir}yolo-patkany\\images\\val\\"
val_labels_dir = f"{root_dir}yolo-patkany\\labels\\val\\"
test_img_dir = f"{root_dir}yolo-patkany\\test\\"
    

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
for i in range(polygon_idx.__len__()):


    ret, frame = cap.read()
    print(f"Processing frame {i+1}/{polygon_idx.__len__()}", end="\r")
    
    if i < highest_keyframe_1_frame:
        continue
    
    # Clear directories on first iteration
    annotation_file = f"{root_dir}yolo-patkany\\labels\\train\\frame_{i:06d}.txt"

    # if i in train_indices:
    #     img_dir = "C:\\Users\\gimes\\Src\\preproc\\gonca\\yolo-patkany\\images\\train\\"
    #     labels_dir = "C:\\Users\\gimes\\Src\\preproc\\gonca\\yolo-patkany\\labels\\train\\"
        
    #     # Select corresponding annotation file
    # else:
    #     img_dir = "C:\\Users\\gimes\\Src\\preproc\\gonca\\yolo-patkany\\images\\val\\"
    #     labels_dir = "C:\\Users\\gimes\\Src\\preproc\\gonca\\yolo-patkany\\labels\\val\\"
    
    img_dir = test_img_dir
    
    cv2.imwrite(fr"{img_dir}frame_{i:06d}.png", frame)
    # Copy annotation file if it exists
    
    # if os.path.exists(annotation_file) and i in val_indices:
    #     shutil.copy(annotation_file, labels_dir)
    
    if ret:
        frames.append(frame)
    else:
        break

print(f"Saved {len(frames)} frames to array")


#%%

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning for display loop

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press Q on keyboard to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
# %%
