from pathlib import Path
import cv2
import numpy as np

ROOT = Path(r"C:\Tasks\Steel Surface Defect Inspection System\data\NEU-DET")
NORMAL = "patches"
SIZE = 128

def load_folder(split, cls):
    folder = ROOT / split / "images" / cls
    imgs = []
    for p in folder.glob("*.jpg"):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (SIZE, SIZE)).astype("float32") / 255.0
        imgs.append(img)
    return np.array(imgs)

if __name__ == "__main__":
    train = load_folder("train", NORMAL)
    print("normal train:", train.shape)
    for c in ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]:
        print(c, load_folder("validation", c).shape)