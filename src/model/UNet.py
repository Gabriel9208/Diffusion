import torch
import torch.nn as nn

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        emb = torch.zeros(time.size(0), self.dim, device=device)
        idx = torch.arange(0, self.dim // 2, device=device)

        emb[:, 0::2] = torch.sin(time.float() / (10000 ** (2 * idx.unsqueeze(0) / self.dim)))
        emb[:, 1::2] = torch.cos(time.float() / (10000 ** (2 * idx.unsqueeze(0) / self.dim)))
        return emb

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.silu = nn.SiLU()
        

    def forward(self, x, emb=None):
        x = self.conv(x)
        x = self.gn(x)
        if emb is not None:
            gamma, beta = emb
            x = (1 + gamma) * x + beta
        x = self.silu(x)

        return x
        

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, label_emb_dim):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.label_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(label_emb_dim, out_channels)
        )

        self.label_film = nn.Linear(out_channels, 2 * out_channels) 

        self.block1 = Block(in_channels, out_channels)
        self.block2 = Block(out_channels, out_channels)
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb, label_emb):  

        t = self.time_mlp(time_emb)
        l = self.label_mlp(label_emb)
        emb = t + l
        gamma_beta = self.label_film(emb).unsqueeze(2).unsqueeze(3)
        gamma, beta = gamma_beta.chunk(2, dim=1)
            
        out = self.block1(x, (gamma, beta))
        out = self.block2(out)
        out = out + self.shortcut(x)
        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, label_emb_dim):
        super().__init__()
        
        self.block1 = ResBlock(in_channels, out_channels, time_emb_dim, label_emb_dim)
        self.block2 = ResBlock(out_channels, out_channels, time_emb_dim, label_emb_dim)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, x, time_emb, label_emb):
        x = self.upsample(x)
        x = self.block1(x, time_emb, label_emb)
        x = self.block2(x, time_emb, label_emb)
        return x
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, label_emb_dim):
        super().__init__()
        
        self.block1 = ResBlock(in_channels, out_channels, time_emb_dim, label_emb_dim)
        self.block2 = ResBlock(out_channels, out_channels, time_emb_dim, label_emb_dim)
        
        self.downsample = nn.MaxPool2d(kernel_size=2)

    def forward(self, x, time_emb, label_emb):
        x = self.block1(x, time_emb, label_emb)
        x = self.block2(x, time_emb, label_emb)
        x = self.downsample(x)
        return x

class UNet(nn.Module):
    def __init__(self, dim = 64, time_emb_dim = 256, label_emb_dim = 256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.down1 = DownBlock(3, dim, time_emb_dim, label_emb_dim)
        self.down2 = DownBlock(dim, dim * 2, time_emb_dim, label_emb_dim)
        self.down3 = DownBlock(dim * 2, dim * 4, time_emb_dim, label_emb_dim)
        self.down4 = DownBlock(dim * 4, dim * 8, time_emb_dim, label_emb_dim)

        self.bottleneck = nn.ModuleList([
            ResBlock(dim * 8, dim * 8, time_emb_dim, label_emb_dim),
            ResBlock(dim * 8, dim * 8, time_emb_dim, label_emb_dim)
        ])
 
        self.up1 = UpBlock(dim * 16, dim * 8, time_emb_dim, label_emb_dim)
        self.up2 = UpBlock(dim * 12, dim * 4, time_emb_dim, label_emb_dim)
        self.up3 = UpBlock(dim * 6, dim * 2, time_emb_dim, label_emb_dim)
        self.up4 = UpBlock(dim * 3, dim, time_emb_dim, label_emb_dim)
        self.up4_out = nn.Conv2d(dim, 3, kernel_size=1, padding=0)

    def forward(self, x, t, label_emb):        
        time_emb = self.time_mlp(t).to(x.device)

        x1 = self.down1(x, time_emb, label_emb)
        x2 = self.down2(x1, time_emb, label_emb)
        x3 = self.down3(x2, time_emb, label_emb)
        x4 = self.down4(x3, time_emb, label_emb)   
        
        out = x4
        for block in self.bottleneck:
            out = block(out, time_emb, label_emb)

        out = self.up1(torch.cat([x4, out], dim=1), time_emb, label_emb)
        out = self.up2(torch.cat([x3, out], dim=1), time_emb, label_emb)
        out = self.up3(torch.cat([x2, out], dim=1), time_emb, label_emb)
        out = self.up4(torch.cat([x1, out], dim=1), time_emb, label_emb)

        return self.up4_out(out)
        

    