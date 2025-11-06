# rodent-seg
## Rodent segmentation with transformator models.
### Transformers	Uses self-attention to look at all patches in all frames at once (in parallel).
 - Global Context: Natively captures long-range spatial and temporal relationships. It can easily connect frame 1 to frame 30.
 - **Computationally Expensive**: The self-attention mechanism has a computational cost that grows quadratically (O(n^2)) with the sequence length. Doubling the number of frames makes the computation roughly four times harder. This is why models often process short clips.Data-Hungry: Like all large neural networks, transformers require massive amounts of video data to be trained effectively.Complexity: These models are large and complex to train and deploy.25
--- 
## Goal definition
 * Binary instance segmentation 
 * Instance recognition? 
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
### Model fine tuning:
#### Generalist vs Specialist. 
 - SAM2: is a great model with video segmentation, however it is not as fine tuneable. (trains only the lightweight mask decoder and prompt encoder)
 - SegFormer: This is a more "traditional" model, with great tuneability support on hugging face
---
### Analysis of SAM 2 applications and user reports 
Confirms that standard consumer hardware is often insufficient.
 - An 8GB VRAM GPU, such as a 3070ti, "is definitely not enough" for larger models.   
 - For deploying a SAM 2 application, "a 16GB VRAM is the minimum I would need".   
 - Cloud-based equivalents, such as the NVIDIA T4 (16GB), are a common recommendation for this level of workload.
Similar project: https://www.youtube.com/watch?v=cEgF0YknpZw
