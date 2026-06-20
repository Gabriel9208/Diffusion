import torch
import torch.nn as nn
import torch.nn.functional as F

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

class AdaGN(nn.Module):
    def __init__(self, num_channels, emb_dim, num_groups=32):
        super().__init__()

        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, affine=False)
        self.proj = nn.Linear(emb_dim, 2 * num_channels)

    def forward(self, x, emb=None):
        x = self.gn(x)
        if emb is not None:
            gamma, beta = self.proj(emb).unsqueeze(2).unsqueeze(3).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()

        self.norm = AdaGN(in_channels, emb_dim)
        self.silu = nn.SiLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x, emb=None):
        x = self.norm(x, emb)
        x = self.silu(x)
        x = self.conv(x)

        return x
        

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()

        self.block1 = Block(in_channels, out_channels, emb_dim)
        self.block2 = Block(out_channels, out_channels, emb_dim)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

        nn.init.zeros_(self.block2.conv.weight)   # identity-at-init: zero-init last conv of residual branch

    def forward(self, x, emb):
        out = self.block1(x, emb)
        out = self.block2(out, emb)
        out = out + self.shortcut(x)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.pre_norm = nn.GroupNorm(num_groups=32, num_channels=in_channel)

        self.Q = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)
        self.K = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)
        self.V = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)

        self.out_conv = nn.Conv2d(in_channel, in_channel, kernel_size=1)

    def forward(self, X):
        B, C, H, W = X.shape

        x = self.pre_norm(X)

        Q = self.Q(x).view(B, C, H * W).permute(0, 2, 1)
        K = self.K(x).view(B, C, H * W).permute(0, 2, 1)
        V = self.V(x).view(B, C, H * W).permute(0, 2, 1)

        attn = Q @ K.transpose(1, 2) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = attn @ V
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        out = self.out_conv(out) + X

        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, enable_attn=False):
        super().__init__()
        self.enable_attn = enable_attn
        if self.enable_attn:
            self.attn1 = AttentionBlock(out_channels)
            self.attn2 = AttentionBlock(out_channels)
        self.block1 = ResBlock(in_channels, out_channels, emb_dim)
        self.block2 = ResBlock(out_channels, out_channels, emb_dim)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, emb):
        x = self.upsample(x)
        x = self.block1(x, emb)
        if self.enable_attn:
            x = self.attn1(x)
        x = self.block2(x, emb)
        if self.enable_attn:
            x = self.attn2(x)

        return x
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, enable_attn=False):
        super().__init__()
        self.enable_attn = enable_attn
        if self.enable_attn:
            self.attn1 = AttentionBlock(out_channels)
            self.attn2 = AttentionBlock(out_channels)
        self.block1 = ResBlock(in_channels, out_channels, emb_dim)
        self.block2 = ResBlock(out_channels, out_channels, emb_dim)

        self.downsample = nn.MaxPool2d(kernel_size=2)

    def forward(self, x, emb):
        x = self.block1(x, emb)
        if self.enable_attn:
            x = self.attn1(x)

        x = self.block2(x, emb)
        if self.enable_attn:
            x = self.attn2(x)

        x = self.downsample(x)
        return x

class UNet(nn.Module):
    def __init__(self, dim = 64, emb_dim = 256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.init_conv = nn.Conv2d(3, dim, kernel_size=3, padding=1)

        self.down1 = DownBlock(dim, dim, emb_dim)
        self.down2 = DownBlock(dim, dim * 2, emb_dim)
        self.down3 = DownBlock(dim * 2, dim * 4, emb_dim, True)
        self.down4 = DownBlock(dim * 4, dim * 8, emb_dim, True)

        self.mid_block1 = ResBlock(dim * 8, dim * 8, emb_dim)
        self.mid_attn = AttentionBlock(dim * 8)
        self.mid_block2 = ResBlock(dim * 8, dim * 8, emb_dim)

        self.up1 = UpBlock(dim * 16, dim * 8, emb_dim, True)
        self.up2 = UpBlock(dim * 12, dim * 4, emb_dim, True)
        self.up3 = UpBlock(dim * 6, dim * 2, emb_dim)
        self.up4 = UpBlock(dim * 3, dim, emb_dim)
        self.up4_out = nn.Conv2d(dim, 3, kernel_size=1, padding=0)

    def forward(self, x, t, label_emb):
        time_emb = self.time_mlp(t).to(x.device)
        emb = time_emb + label_emb

        x = self.init_conv(x)
        x1 = self.down1(x, emb)
        x2 = self.down2(x1, emb)
        x3 = self.down3(x2, emb)
        x4 = self.down4(x3, emb)

        out = self.mid_block1(x4, emb)
        out = self.mid_attn(out)
        out = self.mid_block2(out, emb)

        out = self.up1(torch.cat([x4, out], dim=1), emb)
        out = self.up2(torch.cat([x3, out], dim=1), emb)
        out = self.up3(torch.cat([x2, out], dim=1), emb)
        out = self.up4(torch.cat([x1, out], dim=1), emb)

        return self.up4_out(out)
        

    