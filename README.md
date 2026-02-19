# Thesis Roadmap: Semantic Segmentation of Rodents in Video Data

## 1. Theoretical Background & Architectural Evolution

### 1.1. Evolution of Computer Vision Models

* **CNNs (Convolutional Neural Networks):** High inductive bias; assumes local pixel correlation. Effective for spatial feature extraction but lacks global context.

* **RNNs (Recurrent Neural Networks):** Historically used for temporal sequences but suffer from vanishing gradient problems in long sequences.

* **Traditional ViT (Vision Transformers):** Low inductive bias; learns image concepts from scratch.

  * **Bottleneck:** Standard self-attention possesses $O(N^2)$ quadratic computational complexity. Memory requirements explode with increased image resolution, making them inefficient for high-resolution video frames.

### 1.2. The SegFormer Architecture

* **Inductive Bias Hybridization:** While SegFormer is a Transformer, its **Mix-FFN** (using $3 \times 3$ depth-wise convolutions) explicitly restores a degree of the local inductive bias found in CNNs. This "best of both worlds" approach allows for precise boundary localization, which is paramount in medical and biological imaging.

* **Hierarchical Receptive Fields:** The hierarchical encoder structure (stages $S_1$ to $S_4$) allows the model to capture both coarse global context (e.g., the general location of the cage) and fine local details (e.g., the rodent's whiskers or tail) simultaneously.

* **Resolution Robustness & Coarse-to-Fine Synergy:** Standard ViTs use fixed positional encodings, meaning they "expect" certain features at specific coordinates. SegFormer’s **Mix-FFN** makes it "Position-Agnostic." This is the theoretical bridge that allows you to reuse the SegFormer architecture for both the wide-angle (Stage 1) and zoomed-in (Stage 2) models without the model getting "confused" by the massive scale change.

* **Overlapping Patch Merging:** Preserves local continuity better than non-overlapping patches, enhancing edge details.

### 1.3. Strategic Scope: The Spatial vs. Spatiotemporal Decision

* **The Core Dilemma:** Deciding whether to treat the video as a collection of independent images (Spatial) or as a continuous 3D volume (Spatiotemporal).

* **Spatial Awareness (The Chosen Approach):**

  * **Mechanism:** The model (SegFormer) looks at Frame $t$ and segments it based purely on texture, shape, and contrast within that single image.

  * **Strengths:** High precision per frame; robust to camera cuts; easier to train on smaller datasets.

* **Spatiotemporal Awareness (The Alternative):**

  * **Mechanism:** Architectures like **Video Swin Transformers** explicitly process the time dimension.

  * **The Data Hunger Bottleneck:** To learn temporal dynamics without overfitting, the model needs thousands of continuous sequences. Given limited sessions, a 3D model would likely memorize specific paths rather than generalized physics.

* **The Rejection of CVAT Interpolation:**

  * **Why it fits Spatiotemporal:** 3D models can tolerate the "drift" or noise in interpolated masks because they average information across time.

  * **Why it kills Spatial:** 2D models need high-precision boundaries to learn edge detection. Interpolated masks often drift slightly off the animal, poisoning the training data with "bad edges."

  * **Conclusion:** Prioritized **Quality (Keyframes/Spatial)** over **Quantity (Interpolation/Spatiotemporal)**.

## 2. Data Acquisition & Annotation Pipeline

### 2.1. Dataset Construction

* **Source:** Video data chopped into discrete frames.

* **Annotation Tool:** CVAT (Computer Vision Annotation Tool).

* **Strategy:** Usage of Keyframe extraction to ensure high-precision labels.

* **Format:** Conversion of CVAT XML/JSON to binary bitmasks ($1 = \text{Rat}, 0 = \text{Background}$).

### 2.2. Class Imbalance Management

* **The Problem:** Rat occupies ~5% of pixels; model risks 95% accuracy by predicting only background.

* **Augmentation:** Heavy usage of geometric and photometric augmentation to artificially increase the "rat" presence and variance.

### 2.3. Data Splitting & Leakage Prevention (Data Peaking)

* **The Temporal Correlation Risk:** Frame $t$ is visually nearly identical to Frame $t+1$. Standard random shuffling allows the model to "peak" at the validation set via temporal proximity, inflating metrics.

* **Strict Split Protocol:** Data split by **Video Sequence ID** or distinct **Temporal Blocks**. The validation set consists of entirely new sequences (unseen environments/movements) to test true generalization.

## 3. Methodology: Training & Model Selection

### 3.1. Model Configuration

* **Backbone Selection:** SegFormer-B3.

* **Rationale:** B3 offers a balanced trade-off between receptive field depth and generalization capability, whereas B5 poses a higher risk of overfitting on smaller biological datasets.

* **Input Resolution:** Baseline set to $1024 \times 1024$. Padding applied to maintain aspect ratio.

### 3.2. Loss Function Strategy

* **Experimentation:** Comparative runs between Dice Loss and Cross-Entropy.

* **Rationale (Dice Loss vs. Cross-Entropy):**

  * **Cross-Entropy Weakness:** Treats pixels as independent classifications. In 95% background images, it over-prioritizes background accuracy.

  * **Dice Loss Strength:** Optimizes for overlap volume. If the model misses the rat, the score is 0. This forces prioritization of the minority class.

* **Optimization:** Focal Loss (to focus on hard negatives) and Boundary IoU (to penalize contour errors).

### 3.3. Training Stability & Scheduling

* **Learning Rate Scheduler:** Polynomial Decay with a Warmup phase.

* **Academic Rigor:** Warmup prevents the weights from being "violently twisted" by initial random gradients, while decay ensures surgical precision during the final epochs.

## 4. Evaluation Metrics & Analysis

### 4.1. Core Metrics

* **mIoU (Mean Intersection over Union):** Primary academic standard for general area overlap.

* **Boundary IoU (Edge-Centric Evaluation):** Specifically penalizes morphological errors along contours. Crucial for ensuring the model isn't just learning "blobs" but precise anatomy.

* **Hausdorff Distance (Tertiary Metric):** Measures the maximum distance between predicted and ground-truth boundaries. Highlights the worst-case "miss" (e.g., a detached tail).

* **Dice Coefficient (F1-Score):** Double-weights True Positives; useful for interpreting recall capabilities.

### 4.2. Evaluation Knobs (Thresholding)

* **Binarization Threshold:** Analysis performed across 0.1 to 0.9 range to observe the Precision-Recall trade-off.

### 4.3. The Referee Protocol: Selecting the Gold Standard

* **Concept:** The "Referee" is a comprehensive evaluation suite. If mIoU is similar between models, **Boundary IoU** acts as the tie-breaker to prioritize structural integrity.

* **Qualitative Failure Analysis (Gallery of Failures):** Visual evidence proving that standard mIoU is biased toward large body mass, justifying the selection of morphologically accurate models.

| Metric | What it Rewards | Critical for Rodents? | Stage 2 Impact | 
 | ----- | ----- | ----- | ----- | 
| **mIoU** | General area overlap. | High (global tracking). | Confirms the "Zoom" contains the rat. | 
| **Boundary IoU** | Alignment of contours. | Highest (tail/ear detail). | Key metric to prove Stage 2 worth. | 
| **Hausdorff** | Worst-case boundary error. | High (detached parts). | Ensures the tail is attached. | 

## 5. Post-Processing & Temporal Consistency

### 5.1. Inference Optimization

* **Test-Time Augmentation (TTA):** Averaging predictions of original and flipped frames to smooth edge uncertainties.

* **The "Highlander Rule":** Retaining only the largest connected blob.

  * **Refinement:** Morphological Closing/Dilating applied *before* the rule to bridge gaps caused by occlusions (e.g., cage bars).

### 5.2. Video-Specific Handling

* **Alpha Smoothing:** Temporal smoothing of the mask to reduce jitter between frames.

* **Velocity Tracking:** Kalman Filters to predict bounding box trajectory.

## 6. Advanced Baseline Improvements

### 6.1. Coarse-to-Fine Architecture (Cascaded Inference)

* **Limitation:** A single 1024px model lacks the pixel density to capture a 2px-wide tail accurately.

* **The "Magnifying Glass" Solution:** Crop and upscale the rodent to provide higher pixel density for the same morphological structure.

#### 6.1.1. Dynamic Tiling & ROI Strategy

* **Extraction Pipeline:**

  1. **Stage 1 Detection:** Generate a coarse mask from the full $1024 \times 1024$ frame.

  2. **Convex Hull / Dilation:** Buffer the mask to unify fragmented predictions.

  3. **Dynamic Padding (The 25% Rule):** Calculate a bounding box and expand it by 25%. This accounts for non-rigid stretching of the rodent's body.

  4. **Square-ify & Rescale:** Convert to a square aspect ratio and upscale to $512 \times 512$ or $1024 \times 1024$ using **Lanczos** interpolation.

* **Theoretical Advantage:** SegFormer's **Mix-FFN** allows the model to be "Position-Agnostic," making it an expert in **Rat Anatomy** regardless of its original location in the cage.

#### 6.1.2. The Multi-Scale Ablation Study: Loss Function Dynamics

* **The Hypothesis:** The resolution and scale of the subject dictating the behavior of the loss function.

* **Stage 1 (Global) Bottleneck:**

  * **Extreme Class Imbalance & Feature Discrepancy:** The frame is ~95% background. While the body is massive, the tail is merely ~2 pixels wide. Dice Loss evaluates global overlap, so missing a 2-pixel structure barely registers, causing the model to become "lazy" on fine details. Weighted BCE (e.g., a 5.0 penalty) excels here by forcing hyper-fixation on every minority pixel.

* **Stage 2 (Zoomed) Paradigm Shift:**

  * Cropping the frame permanently alters the mathematical battlefield. The extreme background imbalance is resolved (the rodent occupies 40-60% of the crop), and the tail scales up to a structurally significant 20-30 pixels in thickness.

* **Predicted Loss Behaviors:**

  1. **Dice Loss Reactivation:** With sufficient resolution, Dice Loss becomes highly sensitive to anatomical boundaries, fully exploiting its theoretical advantage for localized object segmentations (paralleling standard medical imaging).

  2. **BCE Over-Aggression Risk:** Applying a massive class weight to Stage 2's perfectly balanced frames risks "hallucinations." The model may over-predict (e.g., rendering a thick, "fat" tail or bleeding into shadows) out of an exaggerated penalty for missed pixels.

* **The 2x2 Experimental Matrix:**
  By testing both loss functions on the magnified dataset, we perform a sophisticated Multi-Scale Ablation Study:

| Viewport | BCE Performance | Dice Performance | Scientific Conclusion | 
 | ----- | ----- | ----- | ----- | 
| **Stage 1 (Global)** | Won (Saved the tail) | Failed (Ignored the tail) | BCE is required to fight extreme class imbalance at low pixel density. | 
| **Stage 2 (Zoomed)** | *Pending* | *Pending* | Does anatomical scaling allow Dice to reclaim its theoretical superiority? | 

## 7. Future Directions & Optimization

### 7.1. Data-Centric Upgrades

* **Dataset Expansion:** Reducing reliance on heavy augmentation via diverse keyframe libraries.

* **Synthetic Data Generation:** Utilizing Diffusion models for perfect ground-truth generation.

* **Active Learning:** Identifying low-confidence frames for targeted human correction.

### 7.2. Training Regime Optimization

* **Curriculum Learning:** Presenting "easy" frames first, followed by complex occlusions.

* **Extended Epochs with Early Stopping:** Ensuring convergence to absolute minima.

### 7.3. Complex Pipeline Evolution

* **Video-Native Transformers (Video Swin):** Modeling 3D space-time volumes.

* **Foundation Model Adaptation:** Fine-tuning SAM/SAM-2 using LoRA.

* **Multi-Animal Instance Segmentation:** Moving beyond the "Highlander Rule" to track social interactions.