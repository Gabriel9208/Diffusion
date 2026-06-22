import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.UNet import UNet

class DDPM(nn.Module):
    def __init__(self,
                 beta_start=1e-4,
                 beta_end=0.02,
                 T=1000,
                 device='cuda',
                 cfg_scale=2.0,
                 thresholding="static",
                 dynamic_thresholding_ratio=0.995,
                 dim=64,
                 emb_dim=256,
                 ):
        super().__init__()
        self.model = UNet(dim, emb_dim)
        self.T = T
        self.device = device
        self.cfg_scale = cfg_scale
        self.thresholding = thresholding                              # "static" | "dynamic" | "none"
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio

        betas = torch.linspace(beta_start, beta_end, T)
        #betas = self.cosine_beta_schedule()
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0).to(device)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        one_minus_alpha_bar = 1 - alpha_bar
        sqrt_one_minus_alpha_bar = torch.sqrt(one_minus_alpha_bar)
        
        self.register_buffer('sqrt_alpha_bar', sqrt_alpha_bar)
        self.register_buffer('one_minus_alpha_bar', one_minus_alpha_bar)
        self.register_buffer('sqrt_one_minus_alpha_bar', sqrt_one_minus_alpha_bar)

    def cosine_beta_schedule(self, s=0.008):
        steps = self.T + 1
        t = torch.linspace(0, self.T, steps, dtype=torch.float64, device=self.device)

        alphas_cumprod = torch.cos(((t / self.T) + s) / (1 + s) * (torch.pi / 2)) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.999).float()

    def uniform_t(self, N:int):     
        return torch.randint(0, self.T, (N, 1), device=self.device)

    def gaussian_noise(self, shape):
        return torch.randn(shape, device=self.device)

    def q_forward(self, x_0, t, epsilon):
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].unsqueeze(2).unsqueeze(3)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].unsqueeze(2).unsqueeze(3)
        
        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * epsilon

    def _threshold_x0(self, x0):
        if self.thresholding == "dynamic":
            B = x0.shape[0]
            s = torch.quantile(x0.reshape(B, -1).abs(),
                               self.dynamic_thresholding_ratio, dim=1)   # per-image threshold
            s = s.clamp(min=1.0).view(B, 1, 1, 1)                        # no outliers -> s=1 -> no-op
            return x0.clamp(-s, s) / s                                   # clamp to [-s,s] then rescale to [-1,1]
        if self.thresholding == "static":
            return x0.clamp(-1, 1)
        return x0  # "none"

    def forward(self, x_0, label_emb):
        N = x_0.shape[0]
        t = self.uniform_t(N)
        eps = self.gaussian_noise(x_0.shape)

        x_t = self.q_forward(x_0, t, eps)
        pred_noise = self.model(x_t, t, label_emb)

        loss = F.mse_loss(eps, pred_noise, reduction='mean')
        return loss

    @torch.no_grad()
    def sample(self, batch_size, label_emb):
        shape = (batch_size, 3, 64, 64)
        x_t = self.gaussian_noise(shape)

        print(f"=== Sample debug, cfg_scale={self.cfg_scale} ===")
        print(f"init x_t: mean={x_t.mean():+.3f} std={x_t.std():.3f} "
              f"min={x_t.min():+.2f} max={x_t.max():+.2f}")

        timesteps = np.linspace(999, 0, 57, dtype=int)
        denoise_process = []
        for idx, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size, 1), t, device=self.device, dtype=torch.long)

            cond_eps = self.model(x_t, t_tensor, label_emb)
            uncond_eps = self.model(x_t, t_tensor, torch.zeros_like(label_emb))
            eps = (1 + self.cfg_scale) * cond_eps - self.cfg_scale * uncond_eps

            sqrt_alpha_bar_t = self.sqrt_alpha_bar[t]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t]

            if t > 0:
                t_minus_one = timesteps[idx + 1]
                sqrt_alpha_bar_t_minus_1 = self.sqrt_alpha_bar[t_minus_one]
                sqrt_one_minus_alpha_bar_t_minus_one = self.sqrt_one_minus_alpha_bar[t_minus_one]
            else:
                sqrt_alpha_bar_t_minus_1 = 1
                sqrt_one_minus_alpha_bar_t_minus_one = 0
            
            predicted_x_0 = (x_t - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t
            predicted_x_0 = self._threshold_x0(predicted_x_0)
            predicted_x_0 *= sqrt_alpha_bar_t_minus_1
            direction_to_x_t = sqrt_one_minus_alpha_bar_t_minus_one * eps
            x_t = predicted_x_0 + direction_to_x_t 

            if idx % 8 == 0:
                denoise_process.append(x_t.squeeze(0))

            if idx % 5 == 0 or idx == len(timesteps) - 1:
                print(f"step {idx:2d} (t={t:4d}): "
                    f"x_t std={x_t.std():.3f} min={x_t.min():+.2f} max={x_t.max():+.2f} | "
                    f"cond_eps std={cond_eps.std():.3f} | "
                    f"|cond-uncond|={(cond_eps - uncond_eps).abs().mean():.4f} | "
                    f"eps std={eps.std():.3f} | "
                    f"pred_x0 std={predicted_x_0.std():.3f}")
            
        denoise_process = torch.stack(denoise_process, dim=0)
        return x_t.clamp(-1, 1), denoise_process
    
