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
    def __init__(self, l1_w, kl_w, adv_w, adv_start_step):
        super().__init__()
        self.l1_w = l1_w
        self.kl_w = kl_w
        self.adv_w = adv_w
        self.adv_start_step = adv_start_step

    def forward(self, recon, target, mu, log_var, disc, global_step, kl_weight=1.0):
        l1 = torch.nn.functional.l1_loss(recon, target)
        kl = -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp())

        # adversarial term, only after reconstruction has settled a bit
        adv_loss = torch.tensor(0.0, device=recon.device)
        if global_step >= self.adv_start_step:
            fake_scores = disc(recon)
            adv_loss = hinge_generator_loss(fake_scores)

        total = l1 * self.l1_w + adv_loss * self.adv_w + kl * kl_weight * self.kl_w

        return total, {
            "l1": l1,
            "adversarial_loss": adv_loss,
            "kl_loss": kl,
        }
