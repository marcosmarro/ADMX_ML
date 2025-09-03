import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

SEQ_LEN = 100000


class FocalLoss1D(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss1D, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Assume inputs and targets are 1D with shape [batch_size, num_classes, length]
        # inputs are logits and targets are indices of the correct class
        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)
        
        # convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 2, 1).float()
        
        # Calculate Focal Loss
        alpha_t = self.alpha * targets_one_hot + (1 - self.alpha) * (1 - targets_one_hot)
        loss = -alpha_t * ((1 - pt) ** self.gamma) * log_pt
        loss = (targets_one_hot * loss).sum(dim=1)  # only keep loss where targets are not zero
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
        
class TransformerModel(nn.Module):

    def __init__(self):
        super(TransformerModel, self).__init__()
        
        
        ADC_channel = 16384 #ABRA ADC Channel
        Embedding_dim = 32
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(Embedding_dim)
        # d_model, nhead, d_hid, dropout
        encoder_layers = TransformerEncoderLayer(Embedding_dim, 2, 128, 0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)
        self.embedding = nn.Embedding(ADC_channel, Embedding_dim, scale_grad_by_freq=True)
        self.d_model = Embedding_dim
        self.linear = nn.Linear(Embedding_dim, ADC_channel)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
#     @torchsnooper.snoop()
    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len]``
            src_mask: Tensor, shape ``[batch_size, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = src.transpose(0,1) # (seq_len, batch_size)
        src = self.embedding(src) * math.sqrt(self.d_model) # (seq_len, batch_size, embedding_dim)
        src = self.pos_encoder(src.permute(1,2,0)).permute(2,0,1) # (batch_size, embedding_dim, seq_len) --> (seq_len, batch_size, embedding_dim)
        output = self.transformer_encoder(src, src_mask) # (seq_len, batch_size, embedding_dim)
        output = self.linear(output).permute(1,2,0) # (batch_size, ntoken, seq_len)
        # output = output.argmax(dim=1)
        return output
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, start=0, dropout=0.1, max_len=SEQ_LEN,factor=1.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.factor = factor

        pe = torch.zeros(max_len, d_model) # (seq_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (1, embedding_dim / 2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2) # (1, seq_len, embedding_dim) --> (1, embedding_dim, seq_len)
        self.register_buffer('pe', pe) # removes pe as a parameter but still shows in state_dict 
        self.start = start
    # @torchsnooper.snoop()
    def forward(self, x):
        x = x + self.factor*self.pe[:,:,self.start:(self.start+x.size(2))] # (batch_size, embedding_dim, seq_len)
        x = self.dropout(x)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        scale_factor = [0.1, 0.01 , 0.001]

        self.encoder = nn.Sequential(
            nn.Linear(input_size, int(input_size*scale_factor[0])),
            nn.ReLU(),
            nn.Dropout1d(.2),
            nn.Linear(int(input_size*scale_factor[0]), int(input_size*scale_factor[1])),
            nn.ReLU(),
            nn.Dropout1d(.2),
            nn.Linear(int(input_size*scale_factor[1]), int(input_size*scale_factor[2])),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
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
            nn.Conv1d(32 + 64, 64, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU()
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

        self.dec3 = nn.Sequential(
            nn.Conv1d(64 +128, 128, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU()
        )

        # Dilation
        self.dilation = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=3, dilation=3, padding_mode='reflect'),
            nn.ReLU())

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
        u1 = torch.cat([u1, e2], dim=1)  # (B, 96, N/2) Skip connection

        d2 = self.dec2(u1)  # (B, 64, N/2)
        u2 = self.up2(d2)   # (B, 64, N)
        u2 = torch.cat([u2, e1], dim=1)  # (B, 192, N) Skip connection

        d3 = self.dec3(u2)  # (B, 128, N)

        # Dilation and Output
        out = self.dilation(d3) # (B, 128, N)
        out = self.output_layer(out) # (B, 1, N)
        return out.squeeze(1)  # (B, N)


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
            x = F.avg_pool1d(x, 2)

        x    = self.bottleneck(x)
        skip = skip[::-1]

        for idx, upblock in enumerate(self.up):
            x = upblock[0](x)
            x = torch.cat((x, skip[idx]), dim=1)
            x = upblock[1](x)
    
        x = self.final(x)

        return x