from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="data/client2/client2.yaml",
    epochs=1,
    imgsz=64,
    project="results",
    name="client2_test",
)