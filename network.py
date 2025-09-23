import torch
import torch.nn as nn
import torch.nn.functional as F


class DAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.bn = nn.BatchNorm1d(1)
       
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),    
        )     

        self.pool1 = nn.AvgPool1d(kernel_size=2) # N/2

        self.enc2 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),     
        )

        self.pool2 = nn.AvgPool1d(kernel_size=2) # N/4

        self.enc3 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU()
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU()
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

        self.dec2 = nn.Sequential(
            nn.Conv1d(32 +64, 64, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU()
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

        self.dec3 = nn.Sequential(
            nn.Conv1d(64 + 128, 128, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU()
        )

        # Dilation
        self.dilation = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=3, dilation=3, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=8, dilation=8, padding_mode='reflect'),
            nn.ReLU(),
        )

        # Final layer
        self.output_layer = nn.Conv1d(128, 1, kernel_size=1)

    def forward(self, x):
        # x = self.bn(x)      # normalizes to mu = 0, std = 1

        # Encoder
        e1 = self.enc1(x)   # (B, 128, N)
        p1 = self.pool1(e1) # (B, 128, N/2)

        e2 = self.enc2(p1)  # (B, 64, N/2)
        p2 = self.pool2(e2) # (B, 64, N/4)

        e3 = self.enc3(p2)  # (B, 32, N/4)

        # Decoder
        d1 = self.dec1(e3)  # (B, 32, N/4)
        u1 = self.up1(d1)   # (B, 32, N/2)
        u1 = torch.cat([u1, e2], dim=1)  # (B, 96, N/2) Skip connection

        d2 = self.dec2(u1)  # (B, 64, N/2)
        u2 = self.up2(d2)   # (B, 64, N)
        u2 = torch.cat([u2, e1], dim=1)  # (B, 192, N) Skip connection

        d3 = self.dec3(u2)  # (B, 128, N)

        # Dilation and Output
        out = self.dilation(d3) # (B, 128, N)
        out = self.output_layer(out) # (B, 1, N)
        return out  # (B, 1, N)
    

class DoubleConv(nn.Module):
    """Two 3x3 convolutions with ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """UNet neural network from arXiv:1505.04597"""
    def __init__(self, in_ch=1, in_features=64, out_ch=1, depth=3):
        super().__init__()
        
        self.down = nn.ModuleList()
        self.up   = nn.ModuleList()

        curr_ch  = in_ch
        features = in_features

        for _ in range(depth):
            self.down.append(DoubleConv(curr_ch, features))
            curr_ch   = features
            features *= 2

        self.bottleneck = DoubleConv(curr_ch, features)

        curr_ch = features

        for _ in range(depth):
            features = features // 2
            self.up.append(nn.Sequential(
                nn.ConvTranspose1d(in_channels=curr_ch, out_channels=features, kernel_size=2, stride=2),
                DoubleConv(curr_ch, features)
            ))
            curr_ch = features

        self.dilation = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=3, dilation=3, padding_mode='reflect'),
            nn.ReLU())

        self.final = nn.Conv1d(in_features, out_ch, kernel_size=1)

    def forward(self, x):
        
        skip = []
        
        for down in self.down:
            x = down(x)
            skip.append(x)
            x = F.max_pool1d(x, 2)

        x    = self.bottleneck(x)
        skip = skip[::-1]

        for idx, upblock in enumerate(self.up):
            x = upblock[0](x)
            x = torch.cat((x, skip[idx]), dim=1)
            x = upblock[1](x)
    
        x = self.final(x)

        return x