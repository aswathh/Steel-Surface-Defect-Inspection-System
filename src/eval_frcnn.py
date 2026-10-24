from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from frcnn_dataset import SteelDetDataset
from frcnn_model import build_model

ROOT = r"C:\Tasks\Steel Surface Defect Inspection System\data\NEU-DET"

def collate(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    val_ds = SteelDetDataset(ROOT, "validation")
    loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate, num_workers=0)

    model = build_model()
    weights = Path(__file__).resolve().parent.parent / "models" / "frcnn.pt"
    model.load_state_dict(torch.load(weights))
    model.eval()

    metric = MeanAveragePrecision(class_metrics=True)
    n_images = 0
    start = time.time()

    with torch.no_grad():
        for imgs, targets in loader:
            preds = model(list(imgs))
            metric.update(preds, list(targets))
            n_images += len(imgs)

    ms_per_image = (time.time() - start) / n_images * 1000
    result = metric.compute()

    print("mAP50:", round(result["map_50"].item(), 3))
    print("mAP50-95:", round(result["map"].item(), 3))
    print("per-class mAP:", [round(x, 3) for x in result["map_per_class"].tolist()])
    print("speed:", round(ms_per_image, 1), "ms per image")