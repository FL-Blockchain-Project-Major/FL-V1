from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="data/client3/client3.yaml",
    epochs=1,
    imgsz=64,
    project="results",
    name="client3_test",
)