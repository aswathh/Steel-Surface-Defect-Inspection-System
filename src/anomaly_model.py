import torch.nn as nn

class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,stride=2,padding=1,),nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1,),nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1,),nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1),nn.ReLU(),
            nn.ConvTranspose2d(32,16,3,stride=2,padding=1,output_padding=1),nn.ReLU(),
            nn.ConvTranspose2d(16,1,3,stride=2,padding=1,output_padding=1),nn.Sigmoid(),
        )

    def forward(self,x):
        return self.decoder(self.encoder(x))

