from rich import padding
import torch
from tqdm import tqdm
from torch import nn 

device = "cuda" if torch.cuda.is_available() else "cpu"


def positional_encoding(t, enc_dim):
    """encodes position with a sinusoid"""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, enc_dim, 2).float() / enc_dim)).to(
        t.device
    )

    pos_enc_a = torch.sin(t.repeat(1, enc_dim // 2) * inv_freq)
    pos_enc_b = torch.cos(t.repeat(1, enc_dim // 2) * inv_freq)
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc


class diffusion:
    """the forward diffusion process
    recreation of the equation
    t = √¯αt x0 + √1 − ¯αt ϵ (3)
    where ¯αt = ∏t, s=1 αs, αt = 1− βt, and ϵ ∼ N (0, 1).
    ( look at equation 3 of https://arxiv.org/abs/2409.16488)
    I know that this is calculating the sum of gaussians instead of performing
    gaussian noise steps in series. but im too retarded to understand the
    equations
    """

    def __init__(
        self,
        noise_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.2,
        image_size=(128, 128), # redefine image_size as a tuple 
        device=device,
    ):
        """initialize diffusion model"""
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.image_size = image_size

    def prepare_noise_schedule(self):
        """creates noise schedule
        creates an 1000 steps of evenly spaced range of beta
        """
        return torch.linspace(
            start=self.beta_start, end=self.beta_end, steps=self.noise_steps
        )

    def forward_diffusion(self, x0, t):
        # add noise using fast forward diffusion algorithm
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[
            :, None, None, None
        ]
        noise = torch.randn_like(x0)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise, noise

    def reverse_diffusion(
        self,
        model,
        n_images,
        n_channels,
        position_encoding_dim,
        position_encoding_function,
        saved_time_steps=None,
        input_image=None,
    ):
        """Reverse diffusion process"""
        with torch.inference_mode():
            x = torch.randn((n_images, n_channels, self.image_size[0], self.image_size[1]))
            x = x.to(self.device)

            denoised_images = []
            for i in tqdm(
                reversed(range(0, self.noise_steps)),
                desc="U-Net inference",
                total=self.noise_steps,
            ):
                t = (torch.ones(n_images) * i).long()
                t_pos_enc = position_encoding_function(
                    t.unsqueeze(1), position_encoding_dim
                ).to(device)

                predict_noise = model(
                    torch.cat((input_image.to(self.device), x), dim=1), t_pos_enc
                )
                alpha = self.alpha[t][:, None, None, None]
                alpha_bar = self.alpha_bar[t][:, None, None, None]

                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = (
                    1
                    / torch.sqrt(alpha)
                    * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predict_noise)
                    + torch.sqrt(1 - alpha) * noise
                )

                if i in saved_time_steps:
                    denoised_images.append(x)

            denoised_images = torch.stack(denoised_images)
            denoised_images = denoised_images.swapaxes(0, 1)
            return denoised_images

    def reverse_diffusion_implicit(
        self,
        model,
        n_images,
        n_channels,
        position_encoding_dim,
        position_encoding_function,
        saved_time_steps=None,
        input_image=None,
        ddim_steps=50,
        eta=0.0,
    ):
        with torch.inference_mode():
            x = torch.randn((n_images, n_channels, self.image_size[0], self.image_size[1]))
            x = x.to(self.device)

            denoised_image = []

            time_steps = torch.linspace(0, self.noise_steps - 1, ddim_steps).long()

            for i in tqdm(
                reversed(range(len(time_steps))),
                desc="Unet inference",
                total=len(time_steps),
            ):
                t = time_steps[i]
                t_prev = time_steps[i - 1] if i - 1 > 0 else torch.tensor(0)

                t_batch = (torch.ones(n_images) * t).long()
                t_pos_enc = position_encoding_function(
                    t_batch.unsqueeze(1), position_encoding_dim
                ).to(device)

                predict_noise = model(
                    torch.cat((input_image.to(self.device), x), dim=1), t_pos_enc
                )

                alpha_bar = self.alpha_bar[int(t.item())]
                alpha_bar_prev = self.alpha_bar[int(t_prev.item())]

                # predict x0 from xt and predicted noise
                x0_pred = (x - torch.sqrt(1 - alpha_bar) * predict_noise) / torch.sqrt(
                    alpha_bar
                )
                x0_pred = x0_pred.clamp(-1, 1)

                sigma = (
                    eta
                    * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                    * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
                )

                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = (
                    torch.sqrt(alpha_bar_prev) * x0_pred
                    + torch.sqrt(1 - alpha_bar_prev - sigma**2) * predict_noise
                    + sigma * noise
                )

                if i in saved_time_steps:
                    denoised_image.append(x)

            denoised_image = torch.stack(denoised_image)
            denoised_image = denoised_image.swapaxes(0, 1)

            return denoised_image


    def sample_latent_ddim(model, vae, diffusion, cond_images, mean, std, device,
                       ddim_steps=25, eta=0.0, latent_dim=8):

      """cond_images: [B,1,512,512] low-SNR inputs -> returns restored [B,1,512,512] in [0,1]."""
        model.eval(); vae.eval()
        mean, std = mean.to(device), std.to(device)
        cond_images = cond_images.to(device)
        # 1. encode + normalize the conditioning ONCE (fixed across all steps)
        # mu_cond, _ = vae.encode(cond_images)
        # z_cond = normalize_latent(mu_cond, mean, std)          # [B, 8, 64, 64]
        # 2. start the target latent from pure noise
        # B, _, hL, wL = z_cond.shape
        # x = torch.randn(B, latent_dim, hL, wL, device=device)
        # 3. DDIM reverse loop over a sub-sampled schedule
        time_steps = torch.linspace(0, diffusion.noise_steps - 1, ddim_steps).long()
        for i in reversed(range(len(time_steps))):
            t      = time_steps[i]
            t_prev = time_steps[i-1] if i > 0 else torch.tensor(0)
            # t_batch = torch.full((B,), int(t), device=device, dtype=torch.long)
            # pred_noise = model(torch.cat([z_cond, x], dim=1), t_batch)   # <-- concat cond each step -> 16ch
            # ab, ab_prev = diffusion.alpha_bar[t], diffusion.alpha_bar[t_prev]
            # x0 = (x - torch.sqrt(1-ab)*pred_noise) / torch.sqrt(ab)
            # ---- see GOTCHA below about clamping x0 ----
            # sigma = eta * torch.sqrt((1-ab_prev)/(1-ab)) * torch.sqrt(1 - ab/ab_prev)
            # noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            # x = torch.sqrt(ab_prev)*x0 + torch.sqrt(1-ab_prev-sigma**2)*pred_noise + sigma*noise
            ...
        # 4. denormalize the final latent, then decode with the (EMA) VAE
        # z0 = denormalize_latent(x, mean, std)
        # return vae.decode(z0)      # [B,1,512,512] in [0,1]
        ...

# create time embedding 

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, time_dim: int, silu : bool):
        super().__init__()

        assert dim % 2 == 0, "Dim must be even"

        self.dim = dim 
        self.time_dim = time_dim 

        self.MLP = nn.Sequential( 
            nn.Linear(dim, time_dim),
            nn.SiLU() if silu else nn.GELU(),
            nn.Linear(time_dim, time_dim)
         )

    def forward(self, t):  # t: LongTensor [B]  ->  returns [B, time_dim]

        half = self.dim // 2 
        freqs = torch.exp( - torch.log(torch.tensor(10000.0)) * torch.arange(half, device = t.device, dtype = torch.float) / half)  # [half], build on t.device
        args = t[:, None].float() * freqs[None, :]                      # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim = -1)   # [B, dim]
        return self.MLP(emb)

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 32, silu: bool = True):
        super().__init__()
        act = nn.SiLU() if silu else nn.GELU()
        # n_groups = min(groups, out_ch)    

        self.res_block1 = nn.Sequential(
            nn.GroupNorm(min(groups, in_ch) , in_ch),
            act,
            nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1)
        )
        self.time_proj = nn.Linear(time_dim, out_ch)
        
        self.res_block2 = nn.Sequential(
            nn.GroupNorm(min(groups, out_ch) , out_ch),
            act, 
            nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1)
        )

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size = 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):   # x: [B, in_ch, H, W],  t_emb: [B, time_dim]  ->  [B, out_ch, H, W]
        h = self.res_block1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]     # broadcast [B,out_ch] -> [B,out_ch,1,1]
        h = self.res_block2(h)
        return h + self.skip(x)
    
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8, groups : int = 32):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dims = channels // num_heads
        # self.use_pos_enc = use_pos_enc

        # query key values
        self.norm = nn.GroupNorm(min(groups, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)



    def forward(self, x, t_embed):

        B, C, H, W = x.shape

        h = self.norm(x)

        # calculate qkv
        qkv = self.qkv(h)
        # separate qkv into heads
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dims, H * W)
        q, k, v = qkv.unbind(1)  # 3 * (B, self_num_heads, self_head_dims, H*W)

        # memory-efficient attention: scaled_dot_product_attention uses a fused/flash
        # kernel and never materializes the (B, heads, N, N) matrix, which is the
        # O(N^2) memory blow-up at the 64x64 bottleneck (N = 4096).
        # SDPA expects (B, heads, seq, head_dims), so transpose the last two dims.
        q = q.transpose(-2, -1).contiguous()
        k = k.transpose(-2, -1).contiguous()
        v = v.transpose(-2, -1).contiguous()

        # default scale is 1/sqrt(head_dims), matching self.scale
        out = nn.functional.scaled_dot_product_attention(q, k, v)

        # back to (B, heads, head_dims, N) -> (B, C, H, W)
        out = out.transpose(-2, -1).reshape(B, C, H, W)

        return self.proj(out) + x


class AdaLNAttentionBlock(nn.Module):
    def __init__(self, channels, time_dim, num_heads=8, groups=32):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dims = channels // num_heads
        self.norm = nn.GroupNorm(min(groups, channels), channels, affine=False)   # no learned affine
        self.qkv  = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        #  modulation: t_emb -> scale, shift, gate  (each [B, C])
        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 3 * channels),
        )
        # (the -Zero) zero-init the final Linear so the block starts as identity
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, t_emb):                 # t_emb REQUIRED here [B, time_dim]
        B, C, H, W = x.shape
        scale, shift, gate = self.ada(t_emb).chunk(3, dim=1)     # each [B, C]
        h = self.norm(x)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]   # (3) modulate, broadcast [B,C]->[B,C,1,1]

        # same as the multihead attention as before 
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dims, H * W)
        q, k, v = qkv.unbind(1) 

        q = q.transpose(-2, -1).contiguous()
        k = k.transpose(-2, -1).contiguous()
        v = v.transpose(-2, -1).contiguous()

        # default scale is 1/sqrt(head_dims), matching self.scale
        out = nn.functional.scaled_dot_product_attention(q, k, v)

        out = out.transpose(-2, -1).reshape(B, C, H, W)

        return x + gate[:, :, None, None] * self.proj(out)



def make_attention(kind, channels, time_dim, num_heads=8, groups=32):
    if kind == "vanilla": return AttentionBlock(channels, num_heads, groups)
    if kind == "adaln":   return AdaLNAttentionBlock(channels, time_dim, num_heads, groups)
    raise ValueError(f"unknown attention kind: {kind}")


class Downsample(nn.Module):
    """Downsample Block"""
    def __init__(self, channels, silu : bool = True, group = 32):
        super().__init__()
        
        act = nn.SiLU() if silu else nn.GELU()

        self.downsample = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size = 3, stride = 2, padding = 1),
            nn.GroupNorm(min(group, channels), channels),
            act,
        )

    def forward(self, x):   # [B,C,H,W] -> [B,C,H/2,W/2]
        return self.downsample(x)


class Upsample(nn.Module):
    """Upsample block"""
    def __init__(self, channels, groups : int = 32, silu : bool = True):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(groups, channels), channels),
            nn.SiLU() if silu else nn.GELU(),
        )
       
    def forward(self, x):   # [B,C,H,W] -> [B,C,2H,2W]
        return self.up(x)
         

class ResAttnBlock(nn.Module):
    """One ResBlock + optional attention. The atom of every UNet level."""
    def __init__(self, in_ch, out_ch, time_dim, use_attn=False, attn_kind="vanilla",
                 num_heads=8, groups=32, silu=True):
        super().__init__()
        self.res  = ResBlock(in_ch, out_ch, time_dim, groups=groups, silu=silu)
        self.attn = make_attention(attn_kind, out_ch, time_dim, num_heads, groups) if use_attn else None

    def forward(self, x, t_emb):        # [B,in_ch,H,W] -> [B,out_ch,H,W]  (spatial unchanged)

        x = self.res(x, t_emb)

        if self.attn is not None:
            x = self.attn(x, t_emb)     

        return x


class Level(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, num_res_blocks, use_attn, attn_kind, num_heads=8, groups=32, silu=True):
        super().__init__()

        self.blocks = nn.ModuleList()

        ch = in_ch

        for _ in range(num_res_blocks):

            self.blocks.append(ResAttnBlock(ch, out_ch, time_dim, use_attn, attn_kind, num_heads, groups, silu))
            ch = out_ch                       # only the FIRST block changes channels; rest are out_ch->out_ch

    def forward(self, x, t_emb):

        for b in self.blocks:

            x = b(x, t_emb)

        return x


class DiffusionUNet(nn.Module):
    def __init__(self, in_channels, out_channels, channels=[128,256,384,512],
                 attn_flags=[False,False,True,True], num_res_blocks=2,
                 attn_kind="vanilla", time_dim=512, num_heads=8, groups=32, silu=True):
        super().__init__()

        self.L = len(channels)

        act = nn.SiLU() if silu else nn.GELU()

        self.time_embed = TimeEmbedding(dim=channels[0], time_dim=time_dim)

        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        # DOWN: down_levels[i] indexed 0..L-1 ; downsamplers[i] indexed 0..L-2
        self.down_levels = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        ch = channels[0]

        for i in range(self.L):

            self.down_levels.append(Level(ch, channels[i], time_dim, num_res_blocks,
                                          attn_flags[i], attn_kind, num_heads, groups, silu))

            ch = channels[i]

            if i < self.L - 1:
                self.downsamplers.append(Downsample(ch))   # keeps ch = channels[i]

        # MID (attention on)
        self.mid = Level(channels[-1], channels[-1], time_dim, num_res_blocks,
                         use_attn=True, attn_kind=attn_kind, num_heads=num_heads, groups=groups, silu=silu)

        # UP: up_levels[i] indexed 0..L-1 ; upsamplers[i] indexed 0..L-2
        self.up_levels = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for i in range(self.L):
            h_ch = channels[i+1] if i < self.L - 1 else channels[-1]   # channels arriving
            in_ch = h_ch + channels[i]                                  # + matching skip

            self.up_levels.append(Level(in_ch, channels[i], time_dim, num_res_blocks,
                                        attn_flags[i], attn_kind, num_heads, groups, silu))

            if i < self.L - 1:
                self.upsamplers.append(Upsample(channels[i+1]))        # keeps ch = channe

        # OUT
        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(groups, channels[0]), channels[0]),
            act,
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1),
        )


    def forward(self, x, t):        # x: [B, in_channels, H, W],  t: LongTensor [B]
        t_emb = self.time_embed(t)
        h = self.conv_in(x)
        # --- down: run each level, SAVE its output as a skip, then downsample ---
        skips = []
        for i in range(self.L):
            # h = down_levels[i](h, t_emb)
            # skips.append(h)
            # if i < self.L - 1: h = downsamplers[i](h)
            h = self.down_levels[i](h, t_emb)
            skips.append(h) # save skips for skip connection
            if i < self.L - 1:
                h = self.downsamplers[i](h) # downsample until last level( 4 channels in list = 3 downsample blocks)   

        # --- mid ---
        # h = self.mid(h, t_emb)
        # bottleneck 
        h = self.mid(h, t_emb)

        # --- up: upsample (except deepest), concat matching skip, run up level ---
        for i in reversed(range(self.L)):
            # if i < self.L - 1: h = upsamplers[i](h)
            # h = torch.cat([h, skips[i]], dim=1)
            # h = up_levels[i](h, t_emb)
            if i < self.L - 1: 
                h = self.upsamplers[i](h)

            h = torch.cat([h, skips[i]], dim = 1)

            h = self.up_levels[i](h, t_emb)

        return self.conv_out(h) # final conv

