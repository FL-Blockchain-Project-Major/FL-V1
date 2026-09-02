from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="data/client1/client1.yaml",
    epochs=1,
    imgsz=64,
    project="results",
    name="client1_test",
)