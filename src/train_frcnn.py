from pathlib import Path
import torch
from torch.utils.data import DataLoader
from frcnn_dataset import SteelDetDataset
from frcnn_model import build_model

ROOT = r"C:\Tasks\Steel Surface Defect Inspection System\data\NEU-DET"

def collate(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    train_ds = SteelDetDataset(ROOT, "train")
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate, num_workers=0)

    model = build_model()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    for epoch in range(5):
        total = 0
        for imgs, targets in loader:
            loss_dict = model(list(imgs), list(targets))
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch+1}/5  loss {total/len(loader):.4f}")

    out = Path(__file__).resolve().parent.parent / "models" / "frcnn.pt"
    torch.save(model.state_dict(), out)
    print("saved", out)