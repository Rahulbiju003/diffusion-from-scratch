import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class DiffusionUtils:
    def __init__(self, timesteps, start=0.0001, end=0.02):
        self.timesteps = timesteps
        self.betas = self.cosine_beta_schedule(timesteps, start, end)
        self.alphas = 1. - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, axis=0)
        self.alphas_prod_prev = F.pad(self.alphas_prod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_prod = torch.sqrt(self.alphas_prod)
        self.sqrt_one_minus_alphas_prod = torch.sqrt(1. - self.alphas_prod)
        self.posterior_variance = self.betas * (1. - self.alphas_prod_prev) / (1. - self.alphas_prod)

    def cosine_beta_schedule(self, timestep, start, end):
        t = torch.linspace(0, 1, timestep)
        alphas_cumprod = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = 1 - alphas_cumprod / torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        betas = start + (end - start) * (betas - betas.min()) / (betas.max() - betas.min())
        return betas.clamp(start, end)

    def get_index_from_list(self, vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device):
        noise = torch.randn_like(x_0)
        sqrt_alphas_prod_t = self.get_index_from_list(self.sqrt_alphas_prod, t, x_0.shape)
        sqrt_one_minus_alphas_prod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_prod, t, x_0.shape)
        return (
            sqrt_alphas_prod_t.to(device) * x_0.to(device) +
            sqrt_one_minus_alphas_prod_t.to(device) * noise.to(device),
            noise.to(device)
        )

    def get_loss(self, model, x_0, t):
        x_noisy, noise = self.forward_diffusion_sample(x_0, t, x_0.device)
        noise_pred = model(x_noisy, t)
        return F.mse_loss(noise, noise_pred)

    def sample(self, model, n_samples, device, img_size=64):
        model.eval()
        with torch.no_grad():
            x = torch.randn(n_samples, 3, img_size, img_size).to(device)
            for i in reversed(range(self.timesteps)):
                t = torch.full((n_samples,), i, dtype=torch.long, device=device)
                noise_pred = model(x, t)
                alpha = self.alphas[t][:, None, None, None]
                alpha_prod = self.alphas_prod[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_prod)) * noise_pred) + torch.sqrt(beta) * noise
        model.train()
        return x

    def show_tensor_image(self, image):
        reverse_transform = lambda t: ((t + 1) / 2 * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        if len(image.shape) == 4:
            image = image[0]
        img = Image.fromarray(reverse_transform(image))
        plt.imshow(img)
        plt.axis('off')
        plt.show()