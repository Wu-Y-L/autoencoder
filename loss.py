import math

import torch
from torch import nn
# https://arxiv.org/abs/1806.03589 SNPatchGAN


def spectral_norm(module):
    return nn.utils.spectral_norm(module)


class snPATCH_discriminator(nn.Module):
    def __init__(self, in_channel=1, base_ch=64, n_layers=3):
        super().__init__()

        layers = [
            spectral_norm(
                nn.Conv2d(in_channel, base_ch, kernel_size=4, stride=2, padding=1)
            ),  # 64 + 2 - 4 / 2 + 1= 32
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]

        ch = base_ch

        for _ in range(n_layers - 1):
            layers += [
                spectral_norm(
                    nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1)
                ),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ]
            ch *= 2

        layers += [
            spectral_norm(nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(ch * 2, 1, kernel_size=4, stride=2, padding=1)),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def hinge_discriminator(real_scores, fake_scores):
    loss_real = torch.nn.functional.relu(1.0 - real_scores).mean()
    loss_fake = torch.nn.functional.relu(1.0 + fake_scores).mean()
    return loss_real, loss_fake


def hinge_generator_loss(fake_scores):
    return -fake_scores.mean()


class VAE_loss(nn.Module):
    def __init__(self, l1_w, kl_w, adv_w, adv_start_step, adv_ramp_steps=2640):
        super().__init__()
        self.l1_w = l1_w
        self.kl_w = kl_w
        self.adv_w = adv_w
        self.adv_start_step = adv_start_step
        self.adv_ramp_steps = adv_ramp_steps

    def forward(self, recon, target, mu, log_var, disc, global_step, kl_weight=1.0):
        l1 = torch.nn.functional.l1_loss(recon, target)
        kl = -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp())

        # adversarial term, only after reconstruction has settled a bit
        adv_loss = torch.tensor(0.0, device=recon.device)
        if global_step >= self.adv_start_step:
            fake_scores = disc(recon)

            # background-masked adversarial loss: mask out disc gradient
            # contribution in background regions (per-image 5th percentile)
            # to prevent hallucination and dim-signal suppression in bg
            with torch.no_grad():
                p5 = torch.quantile(
                    target.view(target.size(0), -1), 0.05, dim=1
                ).view(-1, 1, 1, 1)
                fg_mask = (target > p5).float()

            # downsample mask to disc output resolution
            # disc has 5 stride-2 layers: 512 -> 16, so pool by 32
            mask_pooled = torch.nn.functional.avg_pool2d(
                fg_mask, kernel_size=32, stride=32
            ).clamp(min=0.0, max=1.0)

            # masked mean (not full-tensor mean) to avoid diluting fg
            # adv signal when background dominates the patch
            raw_adv = -(fake_scores * mask_pooled).sum() / (
                mask_pooled.sum() + 1e-8
            )

            # cosine ramp: smooth onset from 0 to full over adv_ramp_steps
            progress = min(
                1.0,
                (global_step - self.adv_start_step)
                / max(self.adv_ramp_steps, 1),
            )
            adv_scale = 0.5 * (1.0 - math.cos(progress * math.pi))
            adv_loss = raw_adv * adv_scale

        total = l1 * self.l1_w + adv_loss * self.adv_w + kl * kl_weight * self.kl_w

        return total, {
            "l1": l1,
            "adversarial_loss": adv_loss,
            "kl_loss": kl,
        }
