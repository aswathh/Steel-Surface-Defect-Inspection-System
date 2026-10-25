from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO
from torchvision import models
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
STD = np.array([0.229, 0.224, 0.225], dtype="float32")

@st.cache_resource
def load_classifier():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 6)
    m.load_state_dict(torch.load(ROOT / "models" / "resnet18.pt"))
    return m.eval()

@st.cache_resource
def load_yolo():
    return YOLO(str(ROOT / "outputs" / "yolo_run" / "weights" / "best.pt"))

st.title("Steel Surface Defect Inspection")
file = st.file_uploader("Upload a steel surface image", type=["jpg", "jpeg", "png"])

if file:
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    st.image(img, caption="Input image", width=300)

    x = cv2.resize(img, (224, 224)).astype("float32") / 255.0
    x = torch.tensor(((x - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(load_classifier()(x), dim=1)[0]
    top = int(probs.argmax())
    st.subheader("Classification (ResNet18)")
    st.write(f"**{CLASSES[top]}** with confidence {probs[top].item():.1%}")

    st.subheader("Detection (YOLO11n)")
    result = load_yolo().predict(img, imgsz=320, conf=0.25, verbose=False)[0]
    st.image(result.plot(), caption=f"{len(result.boxes)} defect(s) found", width=300)