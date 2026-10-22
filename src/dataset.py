from pathlib import Path 
import torch 
from torch.utils.data import Dataset
import cv2

class SteelDefectDataset(Dataset):
    def __init__(self, root, split, classes, transform=None):
        self.transform = transform
        self.classes = classes
        self.samples = []

        for idx, c in enumerate(classes):
            folder = Path(root)/split/"images"/c
            for img_path in folder.glob("*.jpg"):
                self.samples.append((img_path, idx))
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        img_path, label = self.samples[i]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label