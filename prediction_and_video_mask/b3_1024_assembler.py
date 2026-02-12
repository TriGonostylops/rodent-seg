import os
import torch
import multiprocessing
import subprocess
import time
import numpy as np
import evaluate
from datasets import Dataset, Image as DSImage
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer,
    logging
)
from torch import nn

# Reduce verbosity
logging.set_verbosity_info()

# --- CONFIGURATION ---
MODEL_NAME = "nvidia/mit-b3"
OUTPUT_DIR = "/kaggle/working/checkpoints_b3_1024"
FINAL_MODEL_DIR = "/kaggle/working/final_rat_model_b3_1024"

# --- HARDCODED PATHS ---
IMAGE_DIR = "/kaggle/input/rodent-data-2/processed/images"
MASK_DIR = "/kaggle/input/rodent-data-2/processed/masks"

# Training Hyperparameters
EPOCHS = 30
LEARNING_RATE = 6e-5
BATCH_SIZE = 1  # Must be 1 for 1024px on T4
GRAD_ACCUMULATION = 16  # Effective Batch Size = 16


# --- 1. GPU MONITOR ---
def monitor_gpu(interval=60):
    while True:
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
            ).decode().strip().split('\n')
            stats = [f"GPU {i}: {line.split(',')[0]}% Util | {line.split(',')[1]}/{line.split(',')[2]} MB" for i, line
                     in enumerate(result)]
            print(f"\n[GPU MONITOR] " + " | ".join(stats) + "\n")
        except Exception:
            pass
        time.sleep(interval)


# --- 2. DATA LOAD ---
def load_dataset():
    print(f"--- LOADING DATA FROM: {IMAGE_DIR} ---")
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"CRITICAL: {IMAGE_DIR} does not exist.")
    if not os.path.exists(MASK_DIR):
        raise FileNotFoundError(f"CRITICAL: {MASK_DIR} does not exist.")

    all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
    all_masks = [f for f in os.listdir(MASK_DIR) if f.endswith(('.jpg', '.png'))]

    img_map = {os.path.splitext(f)[0]: f for f in all_images}
    mask_map = {os.path.splitext(f)[0]: f for f in all_masks}

    common_ids = sorted(list(set(img_map.keys()) & set(mask_map.keys())))

    print(f"--- DIAGNOSTICS ---")
    print(f"Valid Pairs:  {len(common_ids)}")

    if len(common_ids) == 0:
        raise ValueError("No matching image/mask pairs found!")

    final_image_paths = [os.path.join(IMAGE_DIR, img_map[i]) for i in common_ids]
    final_mask_paths = [os.path.join(MASK_DIR, mask_map[i]) for i in common_ids]

    ds = Dataset.from_dict({"image": final_image_paths, "label": final_mask_paths})
    ds = ds.cast_column("image", DSImage())
    ds = ds.cast_column("label", DSImage())
    ds = ds.train_test_split(test_size=0.10, seed=42)
    return ds


# --- 3. PROCESSOR & TRANSFORMS (FIXED) ---
processor = SegformerImageProcessor.from_pretrained(
    MODEL_NAME,
    reduce_labels=False,
    do_resize=True,
    size={"height": 1024, "width": 1024}
)


def train_transforms(example_batch):
    images = [x.convert("RGB") for x in example_batch["image"]]

    # --- FIX START: FORCE BINARY MASKS ---
    labels = []
    for x in example_batch["label"]:
        # 1. Convert to grayscale numpy
        mask_np = np.array(x.convert("L"))
        # 2. Threshold: Any value > 0 becomes 1 (Rat). 0 stays 0 (Background).
        # This prevents "Class 255" errors.
        mask_np = np.where(mask_np > 0, 1, 0).astype(np.uint8)
        labels.append(mask_np)
    # --- FIX END ---

    return processor(images, labels, return_tensors="pt")


# --- 4. SANITY CHECK ---
def sanity_check(ds):
    print("--- RUNNING SANITY CHECK ---")
    sample = ds["train"][0]
    # Process one sample manually
    output = train_transforms({"image": [sample["image"]], "label": [sample["label"]]})
    unique_vals = torch.unique(output["labels"]).tolist()
    print(f"Processed Mask Values: {unique_vals}")

    if any(v > 1 for v in unique_vals):
        raise ValueError(f"❌ CRITICAL: Mask contains values {unique_vals}. Must be only [0, 1].")
    print("✅ DATA IS SAFE.")


# --- 5. METRICS & MODEL ---
metric = evaluate.load("mean_iou")
id2label = {0: "background", 1: "rat"}
label2id = {"background": 0, "rat": 1}


def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred

        logits_tensor = torch.from_numpy(logits)

        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False
        ).argmax(dim=1)

        metrics = metric.compute(
            predictions=logits_tensor.numpy(),
            references=labels,  # <--- FIXED: Removed .numpy()
            num_labels=2,
            ignore_index=255,
            reduce_labels=False,
        )

        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()}


# --- 6. TRAINER ---
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        upsampled_logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )

        weights = torch.tensor([1.0, 5.0]).to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)

        loss = loss_fct(upsampled_logits, labels)
        return (loss, outputs) if return_outputs else loss


# --- 7. MAIN ---
def main():
    ds = load_dataset()

    # Run safety check before passing to trainer
    sanity_check(ds)

    ds["train"].set_transform(train_transforms)
    ds["test"].set_transform(train_transforms)

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        fp16=True,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="mean_iou",
        report_to="none",
        remove_unused_columns=False  # Keeps 'image' column for transforms
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=compute_metrics,
    )

    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            last_checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
            print(f"!!! RESUMING FROM: {last_checkpoint} !!!")

    print(f"--- TRAINING START: MIT-B3 @ 1024x1024 ---")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print(f"--- SAVING TO {FINAL_MODEL_DIR} ---")
    trainer.save_model(FINAL_MODEL_DIR)
    processor.save_pretrained(FINAL_MODEL_DIR)
    print("DONE.")


if __name__ == "__main__":
    p = multiprocessing.Process(target=monitor_gpu, daemon=True)
    p.start()
    main()