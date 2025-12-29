#%%
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# %%
cap = cv2.VideoCapture("C:\\Users\\gimes\\Src\\preproc\\gonca\\01.59.10-02.03.13[M][0@0][0].dav")
annotations_xml = "C:\\Users\\gimes\\Src\\preproc\\gonca\\annotations.xml"

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
else:
    print("Failed to open video file!")
    print("Please check if the file exists and is accessible.")

tree = ET.parse(annotations_xml)
root = tree.getroot()

# Extract frame annotations
annotations = {}
polygons = root.findall("track")[0].findall("polygon")
polygon_idx = [int(polygon.get("frame")) for polygon in polygons]
polygon_keyframe = [int(polygon.get("keyframe")) for polygon in polygons]
polygon_points = [[str_coords for str_coords in (polygon.get("points").split(";"))] for polygon in polygons]

polygon_points = [[np.array([float(coord) for coord in point.split(",")]) for point in polygon] for polygon in polygon_points]
# Create annotations dictionary with polygon points
for i, frame_idx in enumerate(polygon_idx):
    if frame_idx not in annotations:
        annotations[frame_idx] = []
    
    annotations[frame_idx].append({
        'polygon_points': np.array(polygon_points[i]),
        'keyframe': polygon_keyframe[i]
    })
    
print(f"Total annotations found: {len(polygon_idx)}")
print(f"Total unique frames with annotations: {len(annotations)}")

    
# Find the highest frame value for polygons with keyframe 1
highest_keyframe_1_frame = max([polygon_idx[i] for i in range(len(polygon_keyframe)) if polygon_keyframe[i] == 1])
print(f"Highest frame value for keyframe 1 polygons: {highest_keyframe_1_frame}")

effective_frames = [frame for frame in annotations.keys() if frame <= highest_keyframe_1_frame]
print(f"Total effective frames with annotations: {len(effective_frames)}")

frames = np.unique([polygon.get("frame") for polygon in polygons])
print(f"Total frames in annotations: {len(frames)}")

# %%
class BackgroundSubtractionCV:
    def __init__(self, video_capture, annotations, highest_keyframe_frame):
        self.cap = video_capture
        self.annotations = annotations
        self.highest_keyframe_1_frame = highest_keyframe_frame
        self.current_frame = 0
        self.is_playing = False
        
        # Initialize background subtractors
        self.mog = cv2.createBackgroundSubtractorMOG2()
        self.knn = cv2.createBackgroundSubtractorKNN()
        self.current_subtractor = self.mog
        self.algorithm = 0  # 0 for MOG2, 1 for KNN
        
        # Parameters
        self.history = 500
        self.var_threshold = 16
        self.dist_threshold = 400
        
        self.create_windows()
        
    def create_windows(self):
        # Create windows
        cv2.namedWindow('Background Subtraction', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Controls', cv2.WINDOW_AUTOSIZE)
        
        # Create trackbars
        cv2.createTrackbar('Algorithm (0:MOG2 1:KNN)', 'Controls', 0, 1, self.change_algorithm)
        cv2.createTrackbar('History', 'Controls', 500, 1000, self.update_history)
        cv2.createTrackbar('Variance Threshold', 'Controls', 16, 100, self.update_var_threshold)
        cv2.createTrackbar('Distance Threshold', 'Controls', 400, 1000, self.update_dist_threshold)
        cv2.createTrackbar('Play/Pause (0:Pause 1:Play)', 'Controls', 0, 1, self.toggle_playback)
        cv2.createTrackbar('Reset', 'Controls', 0, 1, self.reset_subtractor)
        
        # Create a dummy image for controls window
        control_img = np.zeros((200, 500, 3), dtype=np.uint8)
        cv2.putText(control_img, 'Background Subtraction Controls', (50, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(control_img, 'Algorithm: 0=MOG2, 1=KNN', (50, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(control_img, 'Press ESC to exit', (50, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(control_img, 'Press SPACE to play/pause', (50, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(control_img, 'Press R to reset', (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('Controls', control_img)
        
    def change_algorithm(self, val):
        self.algorithm = val
        if val == 0:
            self.current_subtractor = self.mog
        else:
            self.current_subtractor = self.knn
        self.reset_subtractor(0)
        
    def update_history(self, val):
        self.history = val
        self.reset_subtractor(0)
        
    def update_var_threshold(self, val):
        self.var_threshold = val
        self.reset_subtractor(0)
        
    def update_dist_threshold(self, val):
        self.dist_threshold = val
        self.reset_subtractor(0)
        
    def toggle_playback(self, val):
        self.is_playing = val == 1
        
    def reset_subtractor(self, val):
        if self.algorithm == 0:
            self.mog = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=False
            )
            self.current_subtractor = self.mog
        else:
            self.knn = cv2.createBackgroundSubtractorKNN(
                history=self.history,
                dist2Threshold=float(self.dist_threshold),
                detectShadows=False
            )
            self.current_subtractor = self.knn
            
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame = 0
        
    def run(self):
        print("Controls:")
        print("- ESC: Exit")
        print("- SPACE: Play/Pause")
        print("- R: Reset")
        print("- Use trackbars to adjust parameters")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # Handle keyboard shortcuts
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE
                self.is_playing = not self.is_playing
                cv2.setTrackbarPos('Play/Pause (0:Pause 1:Play)', 'Controls', int(self.is_playing))
            elif key == ord('r') or key == ord('R'):  # R
                self.reset_subtractor(0)
                cv2.setTrackbarPos('Reset', 'Controls', 0)
            
            if self.is_playing:
                ret, frame = self.cap.read()
                
                if not ret or self.current_frame > self.highest_keyframe_1_frame:
                    self.is_playing = False
                    cv2.setTrackbarPos('Play/Pause (0:Pause 1:Play)', 'Controls', 0)
                    continue
                    
                # Apply background subtraction
                fg_mask = self.current_subtractor.apply(frame)
                
                SIZE = (640, 640)
                
                # Create display image with original and mask side by side
                frame_resized = cv2.resize(frame, SIZE)
                mask_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                mask_resized = cv2.resize(mask_colored, SIZE)
                
                # Combine images
                display_img = np.hstack((frame_resized, mask_resized))
                
                # Add text info
                algo_name = "MOG2" if self.algorithm == 0 else "KNN"
                cv2.putText(display_img, f'Frame: {self.current_frame}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_img, f'Algorithm: {algo_name}', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_img, 'Original', (10, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_img, 'Foreground Mask', (330, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Background Subtraction', display_img)
                
                self.current_frame += 1
            else:
                # Show current frame even when paused
                ret, frame = self.cap.read()
                if ret:
                    frame_resized = cv2.resize(frame, (320, 240))
                    # Create empty mask for display
                    empty_mask = np.zeros((240, 320, 3), dtype=np.uint8)
                    display_img = np.hstack((frame_resized, empty_mask))
                    
                    cv2.putText(display_img, f'Frame: {self.current_frame} (PAUSED)', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(display_img, 'Original', (10, 220), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(display_img, 'Foreground Mask', (330, 220), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow('Background Subtraction', display_img)
                    # Reset frame position to show the same frame
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        
        cv2.destroyAllWindows()

# Create and run the application
if __name__ == "__main__":
    ui = BackgroundSubtractionCV(cap, annotations, highest_keyframe_1_frame)
    ui.run()
