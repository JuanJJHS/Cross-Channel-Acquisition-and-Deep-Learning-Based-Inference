import torch
import torch.nn as nn

"""
Implementation of a Convolutional Neural Network (CNN) PatchGAN discriminator.
"""

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=stride, padding=1,   # <-- add padding=1
                bias=False, padding_mode="reflect"
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=2, features=(64, 128, 256, 512)):
        super().__init__()
        # stride=1 here → denser output map (~61x61 at 256x256), RF ~37x37
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels, features[0],
                kernel_size=4, stride=1, padding=1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_ch = features[0]
        for i, out_ch in enumerate(features[1:]):
            stride = 1 if i == len(features[1:]) - 1 else 2
            layers.append(CNNBlock(in_ch, out_ch, stride=stride))
            in_ch = out_ch

        layers.append(
            nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        h = self.initial(xy)
        return self.model(h)



if __name__ == "__main__":
    test()