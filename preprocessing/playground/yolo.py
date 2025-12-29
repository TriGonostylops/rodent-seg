from ultralytics import YOLO, settings
from ultralytics import YOLO

yolo = "yolo11s-seg"
model = YOLO(f"{yolo}.pt", task="segment")
batch_size = 16
patience = 25
epochs = 100
img_size = 640
data_yaml_path = r"C:\Users\gimes\Src\preproc\gonca\yolo-patkany\data.yaml"
optimizer_name = "Adam"
mask_ratio = 1
learning_rate = 1e-4
cosine_annealing = True


if __name__ == "__main__":
    model.train(
        data=data_yaml_path,
        imgsz=img_size,
        epochs=epochs,
        batch=batch_size,
        workers=8,
        device="0",
        project="yoloruns",
        name=yolo,
        overlap_mask=False,
        mask_ratio=mask_ratio,
        optimizer=optimizer_name,
        cos_lr=cosine_annealing,
        patience=patience,
        lr0=learning_rate,
        augment=True,
    )