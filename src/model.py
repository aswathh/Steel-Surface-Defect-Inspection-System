import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64, 3, padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2)
            )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128*28*28, 6)
        )

    def forward(self,x):
        x= self.features(x)
        x= self.head(x)
        return x