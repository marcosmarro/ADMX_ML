import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        scale_factor = [0.1,0.01,0.001]

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, int(input_size*scale_factor[0])),
            torch.nn.ReLU(),
            torch.nn.Linear(int(input_size*scale_factor[0]), int(input_size*scale_factor[1])),
            torch.nn.ReLU(),
            torch.nn.Linear(int(input_size*scale_factor[1]), int(input_size*scale_factor[2])),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(int(input_size*scale_factor[2]), int(input_size*scale_factor[1])),
            torch.nn.ReLU(),
            torch.nn.Linear(int(input_size*scale_factor[1]), int(input_size*scale_factor[0])),
            torch.nn.ReLU(),
            torch.nn.Linear(int(input_size*scale_factor[0]), input_size)
        )

    def forward(self, input_seq):
        encoded = self.encoder(input_seq.float())
        decoded = self.decoder(encoded)

        return decoded
    

class DCAE(nn.Module):
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
            nn.Conv1d(32 , 64, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU()
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

        self.dec3 = nn.Sequential(
            nn.Conv1d(64 , 128, kernel_size=3, padding=1, padding_mode='reflect'),
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
        x = x.unsqueeze(1)  # (B, 1, N)
        x = self.bn(x)      # normalizes to mu = 0, std = 1

        # Encoder
        e1 = self.enc1(x)   # (B, 128, N)
        p1 = self.pool1(e1) # (B, 128, N/2)

        e2 = self.enc2(p1)  # (B, 64, N/2)
        p2 = self.pool2(e2) # (B, 64, N/4)

        e3 = self.enc3(p2)  # (B, 32, N/4)

        # Decoder
        d1 = self.dec1(e3)  # (B, 32, N/4)
        u1 = self.up1(d1)   # (B, 32, N/2)
        # u1 = torch.cat([u1, e2], dim=1)  # (B, 96, N/2) Skip connection

        d2 = self.dec2(u1)  # (B, 64, N/2)
        u2 = self.up2(d2)   # (B, 64, N)
        # u2 = torch.cat([u2, e1], dim=1)  # (B, 192, N) Skip connection

        d3 = self.dec3(u2)  # (B, 128, N)

        # Dilation and Output
        out = self.dilation(d3) # (B, 128, N)
        out = self.output_layer(out) # (B, 1, N)
        return out.squeeze(1)  # (B, N)



    
