from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\Tasks\Steel Surface Defect Inspection System\outputs\yolo_run\weights\best.pt")
    path= model.export(format="onnx",imgsz=320)
    print("exmported:",path)