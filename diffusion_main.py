import autoencoder
import train
import loss
import vae_dataset_loader

import torch
from pathlib import Path

from torchvision import transforms
from sklearn.model_selection import train_test_split
import torchinfo

import diffusion_model 

torch.cuda.manual_seed(42)
torch.manual_seed(42)

# define data path
DATA_PATH = Path("train_autoencoder")
transform = None
N_PATCHES = 8
PATCH_SIZE = 512
CACHE_SIZE = 2  # per DataLoader worker now, so keep small to avoid OOM/shm blowups
BATCH_SIZE = 16

CHANNEL_LIST = [1, 128, 256, 512]
LATENT_DIMS = 8
ATTN_FLAGS = [False, False, True]

EPOCHS = 100

DIFFUSION_CHANNEL_LIST = [128,256,384,512]
RES_BLOCKS = 2
DIFFUSION_ATTN_FLAG = [False, False, True, True]

NOISE_STEPS =1000

device = "cuda" if torch.cuda.is_available() else "cpu"

vae_dataset = vae_dataset_loader.ConditionalPatchDataset(
    data_dir=DATA_PATH,
    transform=transform,
    n_patches=N_PATCHES,
    patch_size=PATCH_SIZE,
    cache_size=CACHE_SIZE,
)

print(len(vae_dataset))
indices = list(range(len(vae_dataset)))
print("dataset loaded!")

print("spliting into train and test...")
train_indices, test_indices = train_test_split(
    indices, train_size=0.8, shuffle=True, random_state=42
)

train_dataset = torch.utils.data.Subset(vae_dataset, train_indices)
test_dataset = torch.utils.data.Subset(vae_dataset, test_indices)
print("done!")

print("loading data into DataLoader...")
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
print("Done!")

vae = autoencoder.VAE(
    channel_list=CHANNEL_LIST,
    latent_dim=LATENT_DIMS,
    attn_flag=ATTN_FLAGS,
    silu=True,
    int_res=True,
).to(device)

diffusion_unet = diffusion_model.DiffusionUNet(
    in_channels = LATENT_DIMS*2,
    out_channels = LATENT_DIMS,
    channels = DIFFUSION_CHANNEL_LIST,
    attn_flags = DIFFUSION_ATTN_FLAG,
    num_res_blocks = RES_BLOCKS,
    attn_kind = "adaln",
    time_dim=512, 
    num_heads=8, 
    groups=32, 
    silu=True
).to(device)

# load EMA vae checkpoint here 

checkpoint = ""
ema_vae_checkpoint = torch.load(checkpoint, map_location = device)
vae.load_state_dict(ema_vae_checkpoint["ema_vae_state_dict"])

for p in vae.parameters():
    p.requires_grad_(False)

torchinfo.summary(vae, input_size=(BATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE))
torchinfo.summary(diffusion_unet, input_size = (BATCH_SIZE, 16, PATCH_SIZE / (2 ** (len(CHANNEL_LIST)-1)), PATCH_SIZE / (2 ** (len(CHANNEL_LIST)-1))))

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.AdamW(
    diffusion_unet.parameters(),
    lr=5.34e-05,
    betas=(0.512, 0.972),
    weight_decay=9.38e-06,
    eps=9.69e-09,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS, eta_min = 5.34e-05 * 0.01)

diffusion = diffusion_model.diffusion(
    noise_steps = NOISE_STEPS,
    beta_start = 1e-5,
    beta_end = 0.02,
    cosine_s = 0.008,
    image_size=(8, 64, 64), 
    device=device,
    schedule_type = "cosine"
)

mean, std = train.compute_latent_stats(vae, dataloader = train_dataloader, device = device, latent_dim= 8)

results = train.train_diffusion(
    model = diffusion_unet,
    vae = vae, 
    optimizer = optimizer,
    loss_fn = loss_fn,
    train_dataloader = train_dataloader,
    test_dataloader = test_dataloader,
    scheduler = scheduler,
    epochs = EPOCHS,
    mean = mean,
    std = std,
    noise_steps = NOISE_STEPS,
    diffusion = diffusion,
    model_name = "LDM_run_1",
    device = device,
    ema_decay = 0.999,
    latent_dims = LATENT_DIMS
)
