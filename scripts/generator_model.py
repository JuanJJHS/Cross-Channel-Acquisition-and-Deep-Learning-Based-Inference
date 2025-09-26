import torch
import torch.nn as nn
from torch.optim import Adam

"""
Implementation of U-Net. To be used both by the generator of Pix2Pix and the U-Net model.
"""

class Block(nn.Module):
    """
    A helpful class to be used by composition.
    Implements the functionality of both the downsampling and upsampling steps.
    """
    def __init__(self, in_channels, out_channels, down=True, action="relu"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1, padding=1, bias=False, padding_mode="reflect")
            if down else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU() if action=="relu" else nn.LeakyReLU(0.2)
        )


    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=16): #use two out_channels if using gamma 
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=features, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.down1 = Block(features, features*2, down=True, action="leaky")
        self.down2 = Block(features*2, features*4, down=True, action="leaky")
        self.down3 = Block(features*4, features*8, down=True, action="leaky")
        self.down4 = Block(features*8, features*8, down=True, action="leaky")
        self.down5 = Block(features*8, features*8, down=True, action="leaky")
        self.down6 = Block(features*8, features*8, down=True, action="leaky")

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=features*8, out_channels=features*8, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )

        self.up1 = Block(features*8, features*8, down=False, action="relu")
        self.up2 = Block(features*8*2, features*8, down=False, action="relu")
        self.up3 = Block(features*8*2, features*8, down=False, action="relu")
        self.up4 = Block(features*8*2, features*8, down=False, action="relu")
        self.up5 = Block(features*8*2, features*4, down=False, action="relu")
        self.up6 = Block(features*4*2, features*2, down=False, action="relu")
        self.up7 = Block(features*2*2, features, down=False, action="relu")

        self.final_up = nn.Sequential(                   
             nn.ConvTranspose2d(features*2, out_channels, kernel_size=4, stride=1, padding=1),
             nn.Sigmoid()
        )
    
    def forward(self, x): # U-Net forward pass
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)

        bottleneck = self.bottleneck(d7)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], dim=1))
        up3 = self.up3(torch.cat([up2, d6], dim=1))
        up4 = self.up4(torch.cat([up3, d5], dim=1))
        up5 = self.up5(torch.cat([up4, d4], dim=1))
        up6 = self.up6(torch.cat([up5, d3], dim=1))
        up7 = self.up7(torch.cat([up6, d2], dim=1))

        return self.final_up(torch.cat([up7, d1], dim=1))
    
