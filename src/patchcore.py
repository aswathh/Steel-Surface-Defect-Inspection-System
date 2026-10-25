import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import roc_auc_score
from anomaly_data import load_folder, NORMAL

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def features(imgs):
    x = torch.tensor(imgs).unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(x))))
        f2 = backbone.layer2(backbone.layer1(x))
        f3 = backbone.layer3(f2)
    f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear")
    f = torch.cat([f2, f3], dim=1)
    f = F.avg_pool2d(f, 3, stride=1, padding=1)
    return f.permute(0, 2, 3, 1).reshape(len(imgs), -1, f.shape[1])

def scores(imgs, memory):
    out = []
    for i in range(0, len(imgs), 8):
        f = features(imgs[i:i + 8])
        for patches in f:
            d = torch.cdist(patches, memory).min(dim=1).values
            out.append(d.max().item())
    return np.array(out)

if __name__ == "__main__":
    train = load_folder("train", NORMAL)
    mem = torch.cat([features(train[i:i + 8]).reshape(-1, 384) for i in range(0, len(train), 8)])
    idx = torch.randperm(len(mem))[:20000]
    memory = mem[idx]
    print("memory bank:", memory.shape)

    all_scores, labels = [], []
    for c in CLASSES:
        s = scores(load_folder("validation", c), memory)
        print(f"{c:18s} mean score {s.mean():.3f}")
        all_scores.extend(s)
        labels.extend([0 if c == NORMAL else 1] * len(s))

    print("AUROC:", round(roc_auc_score(labels, all_scores), 3))