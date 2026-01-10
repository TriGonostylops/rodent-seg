# rodent-seg
## Rodent segmentation with transformer models.
--- 
## Goal definition
 * Binary instance segmentation
 * Source: video feed
---
## **Supervised** vs **self-supervised
### Self-superviesed(Masked auto encoders): 
 - We could approach this problem as self-supervised learning problem. We have infinite amount of annotated data, so the model is able to create "jigsaw" puzzles with the masked and original images/frames
 - The second step is to use the enhanced encoder, we can later use a pilot (small annotated dataset) to connect the encoder to the head. Then on the first few epoch we freeze the encoder, its weights aren't adjusted.
---
### Supervised learning 
Fine tuning a generalist/specialist model with annotated segment masks of rats.
### Data annotation
#### Brush vs polygons:
Each has its pros and cons, in the end brush masks will be converted to polygons as well. 
**bruhs mask:**
 - precise
 - could be made faster with integrated ai (irl it works like a$$)
 - you basically need to annotate each frame by hand, because the integrated ai is useless.
**polygon mask:**
 - tracking mode: connects the mask and tries to predict the objects trajectory bsaed on two keyframes.
 - slow: you can't and probably shouldn't try to create a perfect mask for each frame. This will undermine the powert of tracking mode, and creates the flickering effect.
Brush masks generally are better for non-rigid objects. However with tracking mode, the machine can connect 2 keyframes that result in a slowly morphing mask instead of a flickering one.
 - Use CVAT for video annotation.
 - Use the polygon tool with tracking enabled.
 - Rats should be segmented with 15-20 points
 - Their tail will be a different entity, and will be tracked with the polyline tool.

Automated vs. Interactive Annotation Workflows
#### Options for creating annotated datasets. 
 * 0-shot: Grounded SAM (https://github.com/IDEA-Research/GroundingDINO): This approach leverages two robust systems. Grounding DINO creates a bounding box around the rodent based on a prompt like 'rodent' **(Text-Driven)**, which is then segmented by the SAM model. The end result is reviewed and corrected with interactive video editing. THis is highly computing power reliant. 
 - 1-shot: SAM + human in the loop approach (f.e.: **Roboflow, Encord, Labelbox**):
   - The annotator clicks on the rodent to be segmented **(Point-Driven)** 
   - SAM creates a mask
   - The mask is propagated through the video
   - The annotator reviews and corrects the annotation
     
  *Goal: Ensure high information density and eliminate bias before training.*

- [ ] **Prune for Diversity (The IoU Rule):** Use a script to filter frames. If the mask overlap (Intersection over Union) between consecutive frames is >95%, discard the redundant ones. 
- [ ] **Target Sample Size:** Aim for ~300–500 high-variety images from this specific 5-minute video rather than using all 6,000.
- [ ] **Balance the "Corner Bias":** Ensure images of the rat in the left corner make up no more than 20–30% of your total set. Delete excess corner images to achieve balance.
- [ ] **Capture Full Pose Variance:** Verify the dataset includes an even mix of:
    - [ ] Stationary/Huddled
    - [ ] Elongated/Walking
    - [ ] Rearing (on hind legs)
    - [ ] Grooming/Scratching
- [ ] **Include Negative Samples (False Positives):** Add roughly 10% "empty cage" images (no rat present) to teach the model not to hallucinate masks in shadows or bedding.
- [ ] **Geometric Augmentation Plan:** Use code (e.g., Albumentations or PyTorch) to apply:
    - [ ] Horizontal/Vertical Flips (to move the rat to different corners).
    - [ ] Random Rotations.
- [ ] **Texture Augmentation:** Include Brightness/Contrast jitter and Gaussian Noise to help the model handle different lighting and sensor grain.

- [ ] **Occlusion Handling (Cage Bars):** - [ ] Create a high-resolution static mask of the foreground cage bars.
    - [ ] Decide on a subtraction strategy: either "striping" the training masks or using a weight-map to ignore bar pixels.
    - [ ] Ensure the model doesn't learn the cage pattern as a "feature" of the rat by using slight translations in augmentation.
---

## 2. Image Processing & Deployment Strategy
*Goal: Determine how the model will handle the Full HD video feed.*

### Option A: Direct Full-Frame Processing
- [ ] **Input Resizing:** Check if resizing $1920 \times 1080$ to the model's native input (usually $640 \times 640$ or $1024 \times 1024$) makes the rat too small/blurry to segment.
- [ ] **GPU Memory:** Ensure your Kaggle instance can handle the VRAM requirements for high-resolution transformer inputs.

### Option B: The Tiling Strategy (Grid Search)
- [ ] **Overlap Management:** Ensure 512x512 tiles overlap (e.g., by 50px) so the rat isn't "cut" at a border.
- [ ] **Stitching Logic:** Develop a method to merge masks if they appear in two adjacent tiles simultaneously.

### Option C: The Hybrid "Crop-on-the-Fly" (Recommended)
- [ ] **Step 1: Detection:** Use a lightweight detector (like YOLO) to find the rat's bounding box in the Full HD frame.
- [ ] **Step 2: Dynamic Cropping:** Create a script to crop that box with 10–20% extra "padding" around the rat.
- [ ] **Step 3: Segmentation:** Pass that high-detail $512 \times 512$ crop to the transformer model.
- [ ] **Step 4: Re-projection:** Map the coordinates of the resulting mask back onto the original $1920 \times 1080$ frame for the final video output.
   - 
---
## Generalist vs Specialist
### Choosing a model 
 - **memory-based models (genearlist) :**  Transformers	Uses self-attention to look at all patches in all frames at once (in parallel).
   - **Global Context:** Natively captures long-range spatial and temporal relationships.
   - The Trade-off: The "memory" problem is drifting.

Naive Per-Frame: Fails by being jittery and inconsistent.

Temporal Propagation: Fails by drifting over long videos and accumulating errors.
 - memory-free models (specialist):
   -  Will suffer from temporal flickering and occlusion problems.
---
### Model fine tuning:
 - Generalist  (SAM2): is a great model with video segmentation, however it is not as fine tuneable. (trains only the lightweight mask decoder and prompt encoder)
 - Specialist (Mask2Former https://huggingface.co/facebook/mask2former-swin-large-cityscapes-semantic ): This is a more "traditional" model, with great tuneability support on hugging face
---
### Analysis of SAM 2 applications and user reports 
Confirms that standard consumer hardware is often insufficient.
 - An 8GB VRAM GPU, such as a 3070ti, "is definitely not enough" for larger models.   
 - For deploying a SAM 2 application, "a 16GB VRAM is the minimum I would need".   
 - Cloud-based equivalents, such as the NVIDIA T4 (16GB), are a common recommendation for this level of workload.
Similar project: https://www.youtube.com/watch?v=cEgF0YknpZw

https://github.com/facebookresearch/sam3/tree/b26a5f330e05d321afb39d01d3d4881f258f65ff?tab=License-1-ov-file
- Mask2Former https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Mask2Former/Inference_with_Mask2Former.ipynb
  
