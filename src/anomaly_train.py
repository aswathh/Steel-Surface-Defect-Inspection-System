from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from anomaly_data import load_folder, NORMAL
from anomaly_model import ConvAE

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

def score(model, imgs):
    x = torch.tensor(imgs).unsqueeze(1)
    with torch.no_grad():
        recon = model(x)
    return ((recon - x) ** 2).mean(dim=(1, 2, 3)).numpy()

if __name__ == "__main__":
    train = torch.tensor(load_folder("train", NORMAL)).unsqueeze(1)

    model = ConvAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(30):
        model.train()
        perm = torch.randperm(len(train))
        total = 0
        for i in range(0, len(train), 16):
            batch = train[perm[i:i + 16]]
            loss = loss_fn(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/30  loss {total / (len(train) / 16):.5f}")

    model.eval()
    scores, labels = [], []
    for c in CLASSES:
        s = score(model, load_folder("validation", c))
        print(f"{c:18s} mean score {s.mean():.5f}")
        scores.extend(s)
        labels.extend([0 if c == NORMAL else 1] * len(s))

    print("AUROC:", round(roc_auc_score(labels, scores), 3))

    out = Path(__file__).resolve().parent.parent / "models" / "conv_ae.pt"
    torch.save(model.state_dict(), out)
    print("saved", out)