
import torch
import torch.nn as nn
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class FusionNet(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.block1 = ConvBlock(in_channels, 64)
        self.block2 = ConvBlock(64, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 128)
        self.block5 = ConvBlock(128, 256)
        self.block6 = ConvBlock(256, 256)

        self.fusion_conv1 = nn.Sequential(
            nn.Conv2d(896, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.block1(x)     
        x2 = self.block2(x1)    
        x3 = self.block3(x2)    
        x4 = self.block4(x3)    
        x5 = self.block5(x4)    
        x6 = self.block6(x5)    

        z1 = torch.cat([x1, x3], dim=1)    
        z3 = torch.cat([z1, x5], dim=1)    
        z2 = torch.cat([x2, x4], dim=1)    
        z4 = torch.cat([z2, x6], dim=1)    
        fused = torch.cat([z3, z4], dim=1)  

        fusion = self.fusion_conv1(fused)
        fusion = self.fusion_conv2(fusion)
        out = self.out_conv(fusion)
        out = torch.sigmoid(out)
        return out

# print("✅ Model architecture defined!")