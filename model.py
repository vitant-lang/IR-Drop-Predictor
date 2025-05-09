import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
class EdgeAwareBlock(nn.Module):
    """feature edge for Chiplet"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.edge_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        identity = x
        x = self.conv(x)
        edges = F.relu(self.edge_conv(x))
        return F.relu(x + identity + edges)  # 增强边缘响应

class SmoothingBlock(nn.Module):
    """enhancing continuity for Interposer"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(channels)
        )
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        
    def forward(self, x):
        identity = x
        x = self.conv(x)
        smoothed = self.avg_pool(x)
        return F.relu(x + identity + smoothed)  # 增强平滑性

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False, block_type=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.attention = attention
        self.block_type = block_type
        
        if attention:
            self.att = AttentionBlock(F_g=out_channels, F_l=out_channels, F_int=out_channels//2)
        
     
        if block_type == "edge":
            self.special_block = EdgeAwareBlock(out_channels)
        elif block_type == "smooth":
            self.special_block = SmoothingBlock(out_channels)
        else:
            self.special_block = None

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        if self.attention:
            x2 = self.att(x1, x2)
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        
        if self.block_type is not None:
            x = self.special_block(x)
            
        return x

class InterposerUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # input channel 5
        self.inc = DoubleConv(5, 64)
        self.down1 = UNetDown(64, 128)
        self.down2 = UNetDown(128, 256)
        self.down3 = UNetDown(256, 512)
        self.down4 = UNetDown(512, 1024)
        
        # upsamping
        self.up1 = UNetUp(1024, 512, attention=True, block_type="edge")
        self.up2 = UNetUp(512, 256, attention=True, block_type="edge")
        self.up3 = UNetUp(256, 128, attention=True, block_type="edge")
        self.up4 = UNetUp(128, 64, attention=True, block_type="edge")
        
        self.outc = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.outc(x)
        return self.sigmoid(x)

class ChipletUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # input channel 3
        self.inc = DoubleConv(3, 64)
        self.down1 = UNetDown(64, 128)
        self.down2 = UNetDown(128, 256)
        self.down3 = UNetDown(256, 512)
        self.down4 = UNetDown(512, 1024)
        
        # upsampling 
        self.up1 = UNetUp(1024, 512, attention=True, block_type="smooth")
        self.up2 = UNetUp(512, 256, attention=True, block_type="smooth")
        self.up3 = UNetUp(256, 128, attention=True, block_type="smooth")
        self.up4 = UNetUp(128, 64, attention=True, block_type="smooth")
        
        self.outc = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.outc(x)
        return self.sigmoid(x)