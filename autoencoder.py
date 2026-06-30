import torch
from torch import nn


class MultiheadSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=8, use_pos_enc=True):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dims = channels // num_heads
        self.scale = self.head_dims**-0.5
        self.use_pos_enc = use_pos_enc

        # query key values
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        # learnable positional encodings
        if use_pos_enc:
            self.pos_enc = nn.Parameter(torch.zeros(1, channels, 64, 64))

    def forward(self, x):

        B, C, H, W = x.shape

        # calculate qkv
        qkv = self.qkv(x)
        # separate qkv into heads
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dims, H * W)
        q, k, v = qkv.unbind(1)  # 3 * (B, self_num_heads, self_head_dims, H*W)

        # add 2d embeddings
        if (
            self.use_pos_enc
            and H <= self.pos_enc.shape[-2]
            and W <= self.pos_enc.shape[-1]
        ):
            pos = self.pos_enc[:, :, :H, :W]
            pos = pos.reshape(1, self.num_heads, self.head_dims, H * W)

            q = q + pos
            k = k + pos

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


class downsample_block(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        groups: int = 32,
        use_attn: bool = False,
        silu: bool = False,
        int_res: bool = False,
    ):
        super().__init__()
        self.int_res = int_res
        # heard group norm is better than batchnorm
        n_groups = min(groups, out_channel)

        # conv block, conv => norm => activation
        self.conv = nn.Sequential(
            # kernel stride and padding calculated by
            # W_out = ( W + 2p - k ) / s + 1
            # W = input dimension, p = padding, k = kernel_size, s = stride
            # say if W = 64, p = 1, s = 2, k = 3
            # 64 + 2 - 3 = 63. 63 / 2 = 31.5 rounds down 31, 31 + 1 = 32, exact factor of 2 down sample
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(n_groups, out_channel),
            nn.SiLU() if silu else nn.GELU(),
        )

        if int_res:
            self.res = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1),
                nn.GroupNorm(n_groups, out_channel),
                nn.SiLU() if silu else nn.GELU(),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(n_groups, out_channel),
            )
            self.act = nn.SiLU() if silu else nn.GELU()

        self.attn = (
            MultiheadSelfAttention(out_channel, num_heads=8, use_pos_enc=True)
            if use_attn
            else nn.Identity()
        )

    def forward(self, x):
        x = self.conv(x)

        # if we want internal residuals
        if self.int_res:
            x = self.act(self.res(x) + x)
        return self.attn(x)


class upsample_block(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        skip_channel,
        groups: int = 32,
        use_attn: bool = False,
        silu: bool = False,
        int_res: bool = False,
    ):
        super().__init__()

        self.int_res = int_res

        n_groups = min(groups, out_channel)

        # upscale before skip connection
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                out_channel + skip_channel, out_channel, kernel_size=3, padding=1
            ),
            nn.GroupNorm(n_groups, out_channel),
            nn.SiLU() if silu else nn.GELU(),
        )

        self.attn = (
            MultiheadSelfAttention(out_channel, num_heads=8, use_pos_enc=True)
            if use_attn
            else nn.Identity()
        )

        if int_res:
            self.res = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1),
                nn.GroupNorm(n_groups, out_channel),
                nn.SiLU() if silu else nn.GELU(),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(n_groups, out_channel),
            )
            self.act = nn.SiLU() if silu else nn.GELU()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)

        x = self.conv(x)
        # residuals
        if self.int_res:
            x = self.act(self.res(x) + x)
        return self.attn(x)


class upsample_block_no_skip(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        groups=32,
        use_attn=False,
        int_res=False,
        silu=False,
    ):
        super().__init__()
        n_groups = min(groups, out_channel)

        self.int_res = int_res

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, out_channel),
            nn.SiLU() if silu else nn.GELU(),
        )

        self.attn = (
            MultiheadSelfAttention(out_channel, num_heads=8, use_pos_enc=True)
            if use_attn
            else nn.Identity()
        )

        if int_res:
            self.res = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1),
                nn.GroupNorm(n_groups, out_channel),
                nn.SiLU() if silu else nn.GELU(),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(n_groups, out_channel),
            )
            self.act = nn.SiLU() if silu else nn.GELU()

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)

        if self.int_res:
            x = self.act(self.res(x) + x)

        return self.attn(x)


class VAE(nn.Module):
    def __init__(
        self, channel_list, latent_dim, attn_flag, silu=False, int_res=False, groups=32
    ):
        """

        channel_list : num channels for each block, example: [3, 32, 64, 128, 256] # must be divisible by 8 ( for attention )
        latent_dim : dim of latent space : 64
        attn_flag : which channels MHA is wanted, example: [ True, True, False, False ] # must be smaller than channel list by 1
        silu : if we want silu, false by default. GELU is used if silu is false
        int_res : if we want internal residuals
        group: group for groupnorm

        """

        super().__init__()

        assert len(attn_flag) == len(channel_list) - 1, (
            "Number of channels must match the number of attention flags"
        )

        self.channel_list = channel_list
        self.attn_flag = attn_flag

        # build encoder
        self.enc_block = nn.ModuleList()
        in_c = channel_list[0]
        for i, out_c in enumerate(channel_list[1:]):
            self.enc_block.append(
                downsample_block(
                    in_c,
                    out_c,
                    groups=groups,
                    use_attn=attn_flag[i],
                    silu=silu,
                    int_res=int_res,
                )
            )
            in_c = out_c

        # spatial conv
        final_channel = channel_list[-1]
        self.conv_mu = nn.Conv2d(final_channel, latent_dim, kernel_size=1)
        self.conv_log_var = nn.Conv2d(final_channel, latent_dim, kernel_size=1)

        # decode first latent dim
        # first decoder layer has no skip connection
        self.dec_initial = nn.Sequential(
            nn.Conv2d(latent_dim, channel_list[-1], kernel_size=3, padding=1),
            nn.GroupNorm(min(groups, channel_list[-1]), channel_list[-1]),
            nn.SiLU() if silu else nn.GELU(),
        )

        # build decoder
        self.decode_block = nn.ModuleList()
        reverse_channel = list(reversed(channel_list))

        for i in range(len(reverse_channel) - 2):
            in_c = reverse_channel[i]
            out_c = reverse_channel[i + 1]

            skip_idx = len(reverse_channel) - 2 - i

            # skip connections removed for latent diffusion: the decoder must
            # reconstruct from z alone, since no encoder activations are available
            # at sampling time. Using the skip-free block instead of upsample_block.
            # skip_c = channel_list[skip_idx]
            use_attn = attn_flag[skip_idx - 1]
            self.decode_block.append(
                upsample_block_no_skip(
                    in_c,
                    out_c,
                    groups=groups,
                    use_attn=use_attn,
                    silu=silu,
                    int_res=int_res,
                )
            )

        # final block is 3, 3 no skip connection
        in_c = reverse_channel[-2]  # 32
        out_c = reverse_channel[-1]  # 3

        # attn off for last up sample as it would be on full res, but its optional
        attn = False
        self.decode_block.append(
            upsample_block_no_skip(
                in_c, out_c, groups, attn, int_res=int_res, silu=silu
            )
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(channel_list[0], channel_list[0], kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        # skips = []  # skip connections removed for latent diffusion
        for block in self.enc_block:
            x = block(x)

            # store skip connections
            # skips.append(x)

        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)

        return mu, log_var

    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        # we write latent space representation as p(z)
        x = self.dec_initial(z)
        # skip connections removed for latent diffusion: every block is skip-free
        # skips_for_decoder = list(reversed(skips[:-1]))

        for block in self.decode_block:
            x = block(x)  # no skip connection

        return self.final_conv(x)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterization(mu, log_var)
        reconstruct = self.decode(z)
        return reconstruct, mu, log_var
