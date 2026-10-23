import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import SteelDefectDataset
from transform import train_transform, Val_transform
from model import SimpleCNN

root = Path("C:/Tasks/Steel Surface Defect Inspection System/data/NEU-DET")
classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

train_ds = SteelDefectDataset(root, "train", classes, transform=train_transform)
val_ds = SteelDefectDataset(root, "validation", classes, transform=Val_transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

model = SimpleCNN(n_classes=6).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train_one_epoch():
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate():
    model.eval()
    correct, total = 0, 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


if __name__ == "__main__":
    epochs = 15
    for epoch in range(epochs):
        train_loss = train_one_epoch()
        val_acc = evaluate()
        print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_acc: {val_acc:.4f}")

   
    torch.save(model.state_dict(), Path(__file__).resolve().parent.parent / "models" / "simple_cnn.pt")
    print("Model saved.")