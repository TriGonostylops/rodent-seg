# rodent-seg
## Rodent segmentation with transformator models.
---
### Model fine tuning:
#### Generalist vs Specialist. 
 - SAM: is a great model, however it is not as fine tuneable. (trains only the lightweight mask decoder and prompt encoder)
 - SegFormer: This is a more "traditional" model, with great tuneability support on hugging face
---
### Data annotation
#### Options for creating annotated datasets. 
 * 0-shot: Grounded SAM (https://github.com/IDEA-Research/GroundingDINO): This approach leverages two robust systems. Grounding DINO creates a bounding box around the rodent based on a prompt like 'rodent', which is then segmented by the SAM model. The end result is reviewed and corrected with interactive video editing. THis is highly computing power reliant.
 - 1-shot: SAM + human in the loop approach:
   - The annotator clicks on the rodent to be segmented
   - SAM creates a mask
   - The mask is propagated through the video
   - The annotator reviews and corrects the annotation 
