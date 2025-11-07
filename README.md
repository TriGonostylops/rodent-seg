# rodent-seg
## Rodent segmentation with transformer models.
--- 
## Goal definition
 * Binary instance segmentation
 * Source: video feed
---
### Data annotation
Automated vs. Interactive Annotation Workflows
#### Options for creating annotated datasets. 
 * 0-shot: Grounded SAM (https://github.com/IDEA-Research/GroundingDINO): This approach leverages two robust systems. Grounding DINO creates a bounding box around the rodent based on a prompt like 'rodent' **(Text-Driven)**, which is then segmented by the SAM model. The end result is reviewed and corrected with interactive video editing. THis is highly computing power reliant. 
 - 1-shot: SAM + human in the loop approach (f.e.: **Roboflow, Encord, Labelbox**):
   - The annotator clicks on the rodent to be segmented **(Point-Driven)** 
   - SAM creates a mask
   - The mask is propagated through the video
   - The annotator reviews and corrects the annotation 
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

- SAM notebook: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-anything-with-sam.ipynb
- Mask2Former https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Mask2Former/Inference_with_Mask2Former.ipynb
