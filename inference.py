import numpy as np
import torch
from tqdm import tqdm

import vae_dataset_loader as vae_dataloader


def make_starts(length, patch_size, overlap):
    """
    Creates patch start positions so the whole image is covered.
    Always includes the final patch position if needed.
    """

    step = patch_size - overlap

    if step <= 0:
        raise ValueError("overlap must be smaller than patch_size")

    starts = list(range(0, length - patch_size + 1, step))

    if len(starts) == 0:
        starts = [0]
    elif starts[-1] < length - patch_size:
        starts.append(length - patch_size)

    return starts


def make_gaussian_weight(patch_size):
    """
    Creates a 2D Gaussian blending mask.
    High weight in the centre, low weight near patch edges.
    """

    xs = np.arange(patch_size) - (patch_size - 1) / 2.0
    ys = np.arange(patch_size) - (patch_size - 1) / 2.0

    xx, yy = np.meshgrid(xs, ys)

    sigma = patch_size / 6.0

    weight = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    weight = weight.astype(np.float32)

    return weight


def normalize_2d_patch(patch):
    """
    Uses the same normalization function as the VAE dataloader.

    patch:
        [H, W] numpy array

    returns:
        [H, W] float32 numpy array
    """

    out = vae_dataloader._normalize_per_slice(patch)

    # In case your dataloader normalization returns a tuple
    if isinstance(out, tuple):
        out = out[0]

    if torch.is_tensor(out):
        out = out.detach().cpu().numpy()

    return out.astype(np.float32)


def inference_2d_ldm(
    model: torch.nn.Module,
    vae: torch.nn.Module,
    noisy_image: np.ndarray,
    diffusion,
    mean: torch.Tensor,
    std: torch.Tensor,
    patch_size: int = 512,
    xy_overlap: int = 64,
    ddim_steps: int = 25,
    eta: float = 0.0,
    device: str = "cuda",
    batch_size: int = 4,
    latent_dims: int = 8,
    blend_mode: str = "gaussian",
    transform=None,
) -> np.ndarray:
    """
    2D latent diffusion inference for microscopy restoration.

    Parameters
    ----------
    model:
        Trained diffusion U-Net, preferably EMA model.

    vae:
        Trained VAE used to encode/decode image patches.

    noisy_image:
        Input noisy microscopy image.
        Shape can be [H, W], [1, H, W], or [H, W, 1].

    diffusion:
        Diffusion class containing sample_latent_ddim.

    mean, std:
        Latent mean/std from compute_latent_stats.

    patch_size:
        Patch size used during training, usually 512.

    xy_overlap:
        Overlap between neighbouring patches.

    ddim_steps:
        Number of DDIM sampling steps.

    eta:
        DDIM stochasticity.
        eta=0.0 gives deterministic DDIM.

    blend_mode:
        "gaussian" or "average".

    transform:
        Optional torchvision transform.
        Use None if training used transform=None.

    Returns
    -------
    clean_img:
        Denoised image with original [H, W] shape.
    """

    model.eval()
    vae.eval()

    # --------------------------------------------------
    # 1. Ensure image is [H, W]
    # --------------------------------------------------
    if noisy_image.ndim == 3:
        if noisy_image.shape[0] == 1:
            noisy_image = noisy_image[0]
        elif noisy_image.shape[-1] == 1:
            noisy_image = noisy_image[..., 0]
        else:
            raise ValueError(
                f"Expected grayscale image with shape [H,W], [1,H,W], or [H,W,1], got {noisy_image.shape}"
            )

    if noisy_image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {noisy_image.shape}")

    h, w = noisy_image.shape
    noisy_image = noisy_image.astype(np.float32)

    # --------------------------------------------------
    # 2. Pad image so 512x512 patches cover everything
    # --------------------------------------------------
    step = patch_size - xy_overlap

    if step <= 0:
        raise ValueError("xy_overlap must be smaller than patch_size")

    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)

    if (h + pad_h - patch_size) % step != 0:
        pad_h += step - ((h + pad_h - patch_size) % step)

    if (w + pad_w - patch_size) % step != 0:
        pad_w += step - ((w + pad_w - patch_size) % step)

    img_padded = np.pad(
        noisy_image,
        ((0, pad_h), (0, pad_w)),
        mode="reflect",
    ).astype(np.float32)

    h_pad, w_pad = img_padded.shape

    # --------------------------------------------------
    # 3. Normalize
    # --------------------------------------------------

    img_for_patches = normalize_2d_patch(img_padded)

    # --------------------------------------------------
    # 4. Prepare output buffers
    # --------------------------------------------------
    clean_img = np.zeros_like(img_padded, dtype=np.float32)

    if blend_mode == "gaussian":
        weight_img = np.zeros_like(img_padded, dtype=np.float32)
        weight_mask = make_gaussian_weight(patch_size)

    elif blend_mode == "average":
        count_img = np.zeros_like(img_padded, dtype=np.float32)

    else:
        raise ValueError("blend_mode must be either 'gaussian' or 'average'")

    # --------------------------------------------------
    # 5. Extract all patch coordinates
    # --------------------------------------------------
    y_starts = make_starts(h_pad, patch_size, xy_overlap)
    x_starts = make_starts(w_pad, patch_size, xy_overlap)

    patches = []
    positions = []

    for y0 in y_starts:
        for x0 in x_starts:
            patch = img_for_patches[
                y0 : y0 + patch_size,
                x0 : x0 + patch_size,
            ]

            patch = vae_dataloader._normalize_per_slice(patch)

            patches.append(patch)
            positions.append((y0, x0))

    # --------------------------------------------------
    # 6. Run batched latent DDIM inference
    # --------------------------------------------------
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(patches), batch_size),
            desc="2D LDM inference",
        ):
            batch_patches = patches[start : start + batch_size]
            batch_positions = positions[start : start + batch_size]

            batch_tensors = []

            for patch in batch_patches:
                tensor = torch.from_numpy(patch).float().unsqueeze(0)
                # [1, H, W]

                if transform is not None:
                    tensor = transform(tensor)

                batch_tensors.append(tensor)

            batch_tensors = torch.stack(batch_tensors, dim=0).to(device)
            # [B, 1, 512, 512]

            denoised = diffusion.sample_latent_ddim(
                model=model,
                vae=vae,
                diffusion=diffusion,
                cond_images=batch_tensors,
                mean=mean,
                std=std,
                device=device,
                ddim_steps=ddim_steps,
                eta=eta,
                latent_dim=latent_dims,
            )

            denoised = denoised.detach().cpu().float().numpy()

            # [B, 1, H, W] -> [B, H, W]
            if denoised.ndim == 4:
                denoised = denoised[:, 0]

            # --------------------------------------------------
            # 7. Stitch patches back into full image
            # --------------------------------------------------
            for idx, (y0, x0) in enumerate(batch_positions):
                patch_out = denoised[idx].astype(np.float32)

                if blend_mode == "gaussian":
                    clean_img[
                        y0 : y0 + patch_size,
                        x0 : x0 + patch_size,
                    ] += patch_out * weight_mask

                    weight_img[
                        y0 : y0 + patch_size,
                        x0 : x0 + patch_size,
                    ] += weight_mask

                else:
                    clean_img[
                        y0 : y0 + patch_size,
                        x0 : x0 + patch_size,
                    ] += patch_out

                    count_img[
                        y0 : y0 + patch_size,
                        x0 : x0 + patch_size,
                    ] += 1.0

    # --------------------------------------------------
    # 8. Normalize overlap blending
    # --------------------------------------------------
    if blend_mode == "gaussian":
        weight_img[weight_img == 0] = 1.0
        clean_img = clean_img / weight_img

    else:
        count_img[count_img == 0] = 1.0
        clean_img = clean_img / count_img

    # --------------------------------------------------
    # 9. Remove padding
    # --------------------------------------------------
    clean_img = clean_img[:h, :w]

    return clean_img.astype(np.float32)