import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

class SteelDetDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root) / split
        self.items = []
        for xml_path in (self.root / "annotations").glob("*.xml"):
            stem = xml_path.stem
            img = self.root / "images" / stem.rsplit("_", 1)[0] / f"{stem}.jpg"
            if img.exists():
                self.items.append((img, xml_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_path, xml_path = self.items[i]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        boxes, labels = [], []
        for obj in ET.parse(xml_path).getroot().findall("object"):
            b = obj.find("bndbox")
            boxes.append([float(b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")])
            labels.append(CLASSES.index(obj.find("name").text) + 1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img, target
    

if __name__ == "__main__":
    ds = SteelDetDataset(r"C:\Tasks\Steel Surface Defect Inspection System\data\NEU-DET", "train")
    img, target = ds[0]
    print(len(ds), img.shape, target)