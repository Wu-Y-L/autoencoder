import autoencoder
import train
import loss
import vae_dataset_loader

import torch
from pathlib import Path

from torchvision import transforms
from sklearn.model_selection import train_test_split
import torchinfo

torch.cuda.manual_seed(10)
torch.manual_seed(10)

# define data path
DATA_PATH = Path("train_autoencoder")
transform = None
NUM_WORKERS = -1
N_PATCHES = 8
PATCH_SIZE = 512
CACHE_SIZE = 10
BATCH_SIZE = 8

CHANNEL_LIST = [1, 128, 256, 512]
LATENT_DIMS = 8
ATTN_FLAGS = [False, False, True]

EPOCHS = 100

device = "cuda" if torch.cuda.is_available() else "cpu"

vae_dataset = vae_dataset_loader.load_vae_dataset(
    input_dir=DATA_PATH,
    transform=transform,
    n_jobs=NUM_WORKERS,
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
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)
print("Done!")

vae = autoencoder.VAE(
    channel_list=CHANNEL_LIST,
    latent_dim=LATENT_DIMS,
    attn_flag=ATTN_FLAGS,
    silu=True,
    int_res=True,
).to(device)
discriminator = loss.snPATCH_discriminator(in_channel=1, base_ch=64, n_layers=3).to(
    device
)

torchinfo.summary(vae, input_size=(BATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE))
torchinfo.summary(discriminator, input_size=(BATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE))

loss_fn = loss.VAE_loss(l1_w=1.0, kl_w=1.0, adv_w=0.5, adv_start_step=10000)

vae_opt = torch.optim.AdamW(
    vae.parameters(),
    lr=1.34e-04,
    betas=(0.516, 0.972),
    weight_decay=9.38e-06,
    eps=9.69e-09,
)

disc_opt = torch.optim.AdamW(
    discriminator.parameters(),
    lr=1.34e-04,
    betas=(0.512, 0.972),
    weight_decay=9.38e-06,
    eps=9.69e-09,
)

steps_per_epoch = len(train_dataloader)
total_steps = steps_per_epoch * EPOCHS

kl_start = int(0.05 * total_steps)
kl_end = int(0.40 * total_steps)

kl_annealing_scheduler = train.KL_cosine_annealing(start_step=kl_start, end_step=kl_end)

result = train.train_vae(
    vae=vae,
    discriminator=discriminator,
    vae_opt=vae_opt,
    disc_opt=disc_opt,
    vae_loss=loss_fn,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    epochs=EPOCHS,
    vae_lr_scheduler=None,
    disc_lr_scheduler=None,
    model_name="VAE_run_1",
    kl_annealing_scheduler=kl_annealing_scheduler,
    device=device,
)

