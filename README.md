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

  * **Mechanism:** The model (SegFormer) looks at Frame $t$ and segments it based purely on texture, shape, and contrast within that single image. It has no memory of Frame $t-1$.

  * **Strengths:** High precision per frame; robust to camera cuts; easier to train on smaller datasets.

* **Spatiotemporal Awareness (The Alternative):**

  * **Mechanism:** Architectures like **Video Swin Transformers** or **3D CNNs** explicitly process the time dimension. They understand that if a blob moves 5 pixels left in $t-1$, it should be here in $t$.

  * **The Data Hunger Bottleneck:** To learn temporal dynamics without overfitting, the model needs thousands of continuous sequences. Given the limited number of unique experimental sessions, a spatiotemporal model would likely memorize specific rat paths rather than generalized movement physics.

* **The Rejection of CVAT Interpolation:**

  * **Why it fits Spatiotemporal:** 3D models need dense (30fps) data and can tolerate the "drift" or noise in interpolated masks because they average information across time.

  * **Why it kills Spatial:** 2D models need high-precision boundaries to learn edge detection. Interpolated masks often drift slightly off the animal. Feeding these noisy labels to a frame-by-frame model poisons the training data with bad edges.

  * **Conclusion:** The decision was made to prioritize **Quality (Keyframes/Spatial)** over **Quantity (Interpolation/Spatiotemporal)**.

## 2. Data Acquisition & Annotation Pipeline

### 2.1. Dataset Construction

* **Source:** Video data chopped into discrete frames.

* **Annotation Tool:** CVAT (Computer Vision Annotation Tool).

* **Strategy (Quality > Quantity):** Usage of Keyframe extraction to ensure high-precision labels rather than noisy, interpolated frames (Direct consequence of Section 1.3).

* **Format:** Conversion of CVAT XML/JSON to binary bitmasks ($1 = \text{Rat}, 0 = \text{Background}$).

### 2.2. Class Imbalance Management

* **The Problem:** Rat occupies \~5% of pixels; model risks 95% accuracy by predicting only background.

* **Augmentation:** Heavy usage of geometric and photometric augmentation to artificially increase the "rat" presence and variance, preferred over simple linear interpolation.

### 2.3. Data Splitting & Leakage Prevention (Data Peaking)

* **The Temporal Correlation Risk:** In video datasets, Frame $t$ is visually nearly identical to Frame $t+1$. A standard random shuffle split allows the model to "peak" at the validation set because it has already seen a virtually identical frame in the training set. This results in artificially inflated validation metrics.

* **Strict Split Protocol:** To prevent leakage, data must be split by **Video Sequence ID** or distinct **Temporal Blocks**, never by individual frames. The validation set must consist of entirely new video sequences (unseen environments/movements) to test true generalization rather than texture memorization.

## 3. Methodology: Training & Model Selection

### 3.1. Model Configuration

* **Backbone Selection:** SegFormer-B3 vs. SegFormer-B5.

  * **Decision:** Prioritize **SegFormer-B3**.

  * **Rationale:** B5 has significantly higher parameter counts and embedding dimensions. Without massive datasets, B5 poses a high risk of overfitting. B3 offers a balanced trade-off between receptive field depth and generalization capability.

* **Input Resolution:** Baseline set to $1024 \times 1024$. Padding is applied to maintain aspect ratio and uniform format.

### 3.2. Loss Function Strategy

* **Experimentation:** Comparative training runs to establish a Gold Standard.

  * **Run A:** Dice Loss (optimizes F1 score directly).

  * **Run B:** Cross-Entropy / IoU Loss.

* **Rationale (Dice Loss vs. Cross-Entropy):**

  * **Cross-Entropy Weakness:** It treats every pixel as an independent classification. Since 95% of the image is background, the model can achieve high "Global Accuracy" simply by ignoring the rodent.

  * **Dice Loss Strength:** It optimizes for the overlap volume. If the model predicts only background, the Dice score is 0, regardless of how accurate the background pixels are. This forces the model to prioritize the rodent class.

* **Optimization:**

  * **Focal Loss:** Applied to address class imbalance by down-weighting easy examples (background) and focusing on hard negatives.

  * **Boundary IoU:** Implemented alongside mIoU to penalize boundary errors more strictly than internal pixel errors.

### 3.3. Training Stability & Scheduling

* **Learning Rate Scheduler:** Transformers are notoriously sensitive to learning rates. To prevent divergence or plateaus, a **Polynomial Decay** or **Cosine Annealing** scheduler is employed.

* **Academic Rigor:** Documenting the scheduler prevents training instability and ensures reproducible convergence.

## 4. Evaluation Metrics & Analysis

### 4.1. Core Metrics

* **mIoU (Mean Intersection over Union):** The primary academic standard for segmentation accuracy.

  * **Limitation:** Standard mIoU is "forgiving" for large objects; it can report high scores (e.g., $0.90$) even if the model consistently misses fine structures like the tail.

* **Boundary IoU (Edge-Centric Evaluation):** Specifically penalizes morphological errors along the contours. Unlike standard mIoU, this metric degrades significantly if fine details (whiskers, tail tip) are inaccurate, ensuring the model isn't just learning "blobs."

* **Hausdorff Distance (Tertiary Metric):** Provides metric complementarity to IoU. While IoU measures area overlap, Hausdorff Distance measures the maximum distance between predicted and ground-truth boundaries. It highlights the single most significant "miss" (e.g., a completely detached tail segment) that area-based metrics might average out.

* **Dice Coefficient (F1-Score):** Double-weights True Positives. Provides a more optimistic view of overlap; useful for interpreting recall capabilities.

### 4.2. Evaluation Knobs (Thresholding)

* **Binarization Threshold:** The probability cutoff ($P > x$) for converting the softmax output (0.0 to 1.0) into a binary mask.

* **Precision-Recall Trade-off:**

  * High Threshold ($>0.8$): Increases Precision (thinner masks, fewer false positives).

  * Low Threshold ($<0.3$): Increases Recall (thicker masks, potential shadow inclusion).

  * **Analysis:** Plotting performance across the 0.1 to 0.9 threshold range rather than reporting a single static number.

### 4.3. The Referee Protocol: Selecting the Gold Standard

* **Concept:** The "Referee" is a comprehensive evaluation suite rather than a single number. In high-stakes biological segmentation, no single metric captures the full picture.

* **The Decision Rules:**

  * **Standard mIoU (The Accuracy Baseline):** Acts as the primary judge for general detection. *Note:* If Cross-Entropy achieves higher mIoU but visually misses thin structures, it may be biased by the dominant background class.

  * **Boundary IoU (The Detail Judge):** Specifically arbitrates morphological precision. *Decision Rule:* If Dice Loss yields a significantly higher Boundary IoU—even with a lower or equal mIoU—it becomes the "Gold Standard" because it ensures structural integrity (e.g., tail tips).

* **Qualitative Failure Analysis (Gallery of Failures):**

  * **Method:** Include a visual comparison where Model A (BCE) might have higher mIoU but completely misses the tail, whereas Model B (Dice) has a lower mIoU but correctly segments the structure.

  * **Justification:** This proves to the reviewer that standard mIoU is biased toward large body mass and justifies the choice of a morphologically accurate model over a purely statistical winner.

* **Comparative Matrix:**

| Metric | What it Rewards | Critical for Rodents? | Stage 2 Impact |
| :--- | :--- | :--- | :--- |
| **mIoU** | General area overlap. | High (general tracking). | Confirms the "Zoom" contains the rat. |
| **Dice (F1)** | Harmonic mean of precision/recall. | High (volume accuracy). | - |
| **Boundary IoU** | Alignment of contours. | Highest (tail/ear detail). | The key metric to prove Stage 2 was worth it. |
| **Hausdorff** | Worst-case boundary error. | High (catching detached parts). | Ensures the tail is attached to the body. |

## 5. Post-Processing & Temporal Consistency

### 5.1. Inference Optimization

* **Test-Time Augmentation (TTA):** Averaging predictions of the original frame and a horizontally flipped version during inference. Smooths edge uncertainties without retraining.

* **The "Highlander Rule" (Connected Components):** Filtering technique to retain only the largest connected blob, assuming only one subject exists.

  * **Refinement:** Implementation of **Morphological Operations (Closing/Dilating)** *before* applying the Highlander Rule. This bridges small gaps caused by occlusions (e.g., cage bars) to prevent the rule from discarding half the animal if the mask is split.

### 5.2. Video-Specific Handling

* **Motion Blur Compensation:** Addressing feature loss in fast-moving frames.

* **Alpha Smoothing:** Temporal smoothing of the prediction mask to reduce jitter between consecutive frames.

* **Velocity Tracking:** Implementation of Kalman Filters to predict the next bounding box location based on trajectory.

## 6. Advanced Baseline Improvements

### 6.1. Coarse-to-Fine Architecture

* **Limitation:** A single model at full resolution often lacks the Receptive Field focus for small subjects.

* **Two-Stage Cascaded Network:**

  1. **Stage 1 (Coarse):** Use SegFormer-B3 on $1024 \times 1024$ full frames to generate a bounding box.

  2. **Stage 2 (Fine):** Crop the Region of Interest (ROI) and feed it into a specialized SegFormer model trained strictly on "zoomed-in" rat data.

### 6.2. Literature Context

* Referencing "Coarse-to-fine semantic segmentation frameworks."

* Referencing "Hierarchical encoders in Vision Transformers."

## 7. Future Directions & Optimization

### 7.1. Data-Centric Upgrades (The "No-Brainer" Improvements)

* **Dataset Expansion:** Increasing the raw number of annotated keyframes. A larger, more diverse dataset reduces the reliance on heavy augmentation and prevents overfitting on specific cage textures.

* **Synthetic Data Generation:** Utilizing GANs or Diffusion models to generate realistic rodent-in-cage images with perfect ground-truth masks. This effectively solves the "data quantity" bottleneck without manual labor.

* **Active Learning:** Implementing a "Human-in-the-Loop" pipeline where the model identifies frames with low confidence scores, and a human annotator manually corrects only those specific frames, maximizing annotation efficiency.

### 7.2. Training Regime Optimization

* **Extended Epochs with Early Stopping:** significantly increasing the training duration while monitoring validation loss. This ensures the model converges to its absolute minimum potential rather than stopping prematurely.

* **Curriculum Learning:** A training strategy that presents "easy" examples (clear background, high contrast) first, then progressively introduces "hard" examples (occlusions, motion blur). This stabilizes the early training phase and often leads to higher final accuracy.

### 7.3. Complex Pipeline Evolution

* **Video-Native Transformers (Video Swin):** Moving beyond frame-by-frame SegFormer to architectures that explicitly model the temporal dimension (3D space-time volumes). This would allow the model to understand that a "disappearing tail" is likely an occlusion, not a disappearance.

* **Foundation Model Adaptation (SAM-Adapter):** Fine-tuning the "Segment Anything Model" (SAM) or SAM-2 using Low-Rank Adaptation (LoRA). Leveraging a model trained on billions of images could provide superior zero-shot generalization compared to training SegFormer from scratch.