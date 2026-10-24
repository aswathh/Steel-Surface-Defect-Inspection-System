from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    model.train(
        data= r"C:\Tasks\Steel Surface Defect Inspection System\data\yolo\data.yaml",
        epochs = 30,
        batch = 16,
        imgsz= 320,
        workers =0,
        project=r"C:\Tasks\Steel Surface Defect Inspection System\outputs",
        name = "yolo_run",
    )