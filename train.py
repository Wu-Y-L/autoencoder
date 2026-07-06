# add KL annealing scheduler
import copy
import math
import loss
import torch
from tqdm import tqdm
from alive_progress import alive_bar
from pathlib import Path
import pickle as pypickle
import matplotlib.pyplot as plt


class KL_cosine_annealing:
    def __init__(self, start_step, end_step):
        self.start_step = start_step
        self.end_step = end_step

    def get_value(self, current_step):
        if current_step <= self.start_step:
            return 0.0
        elif current_step >= self.end_step:
            return 1.0

        else:
            progress = (current_step - self.start_step) / (
                self.end_step - self.start_step
            )
            current_val = 0.5 * (1 - math.cos(progress * math.pi))
            return current_val


def train_step(
    vae,
    discriminator,
    disc_opt,
    vae_opt,
    loss_fn,
    train_dataloader,
    kl_scheduler: KL_cosine_annealing,
    step,
    device,
    scaler,
    ema_vae=None,
    ema_decay=0.999,
):

    total_disc_loss, total_vae_loss = 0.0, 0.0

    # bf16 has fp32's exponent range, so activations can't overflow to inf at the
    # fp16 ceiling (65504) -- which is what was producing NaN recon a few steps in.
    # Fall back to fp16 only on GPUs without bf16 support.
    amp_dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    for i, X in enumerate(train_dataloader):
        vae.train()
        discriminator.train()

        X = X.to(device)

        with torch.autocast(device_type=device, dtype=amp_dtype):
            reconstruct, mu, log_var = vae(X)

        # train discriminator (gated: prevents over-specialization before gen feels adv)
        disc_loss = torch.tensor(0.0, device=device)
        if step >= loss_fn.adv_start_step:
            disc_opt.zero_grad()

            disc_recon = reconstruct.detach()

            with torch.autocast(device_type=device, dtype=amp_dtype):
                fake_score = discriminator(disc_recon)
                real_score = discriminator(X)

                loss_real, loss_fake = loss.hinge_discriminator(
                    real_score, fake_score
                )
                disc_loss = loss_real + loss_fake

            scaler.scale(disc_loss).backward()
            scaler.step(disc_opt)

        # train vae
        vae_opt.zero_grad()

        kl_w = kl_scheduler.get_value(current_step=step)

        with torch.autocast(device_type=device, dtype=amp_dtype):
            vae_loss, loss_dict = loss_fn(
                recon=reconstruct,
                target=X,
                mu=mu,
                log_var=log_var,
                disc=discriminator,
                global_step=step,
                kl_weight=kl_w,
            )

        scaler.scale(vae_loss).backward()

        scaler.step(vae_opt)

        # vae_opt.step()

        total_disc_loss += disc_loss.cpu().item()
        total_vae_loss += vae_loss.cpu().item()

        if i % 100 == 0:
            print(
                f"currently at : {i} / {len(train_dataloader)} \n"
                f"avg_vae_loss : {total_vae_loss / max(i, 1)}\n"
                f"loss_dict : {loss_dict} \n",
                f"discriminator loss : {total_disc_loss / max(i, 1)}\n",
            )

        step += 1

        avg_disc_loss = total_disc_loss / len(train_dataloader)
        avg_vae_loss = total_vae_loss / len(train_dataloader)

        scaler.update()

        if ema_vae is not None:
            with torch.no_grad():
                for ema_p, p in zip(ema_vae.parameters(), vae.parameters()):
                    ema_p.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

    return avg_vae_loss, avg_disc_loss, step


def test_step(vae, discriminator, test_dataloader, loss_fn, kl_scheduler, step, device):

    vae.eval()
    discriminator.eval()

    total_disc_loss, total_vae_loss = 0.0, 0.0

    with torch.inference_mode():
        for i, X in enumerate(test_dataloader):
            X = X.to(device)

            # get discriminator loss
            reconstruct, mu, log_var = vae(X)

            fake_score = discriminator(reconstruct.detach())
            real_score = discriminator(X)

            loss_real, loss_fake = loss.hinge_discriminator(real_score, fake_score)

            disc_loss = loss_real + loss_fake

            # get vae loss
            kl_w = kl_scheduler.get_value(current_step=step)

            vae_loss, loss_dict = loss_fn(
                recon=reconstruct,
                target=X,
                mu=mu,
                log_var=log_var,
                disc=discriminator,
                global_step=step,
                kl_weight=kl_w,
            )

            total_disc_loss += disc_loss.cpu().item()
            total_vae_loss += vae_loss.cpu().item()

            with torch.no_grad():
                p5 = torch.quantile(
                    X.view(X.size(0), -1), 0.05, dim=1
                ).view(-1, 1, 1, 1)
                p10 = torch.quantile(
                    X.view(X.size(0), -1), 0.10, dim=1
                ).view(-1, 1, 1, 1)

                p10_target = torch.quantile(
                    X.view(X.size(0), -1), 0.10, dim=1
                )
                p10_recon = torch.quantile(
                    reconstruct.view(reconstruct.size(0), -1), 0.10, dim=1
                )
                loss_dict["dim_preservation_ratio"] = (
                    p10_recon / (p10_target + 1e-8)
                ).mean()

                bg_mask = (X <= p5).float()
                bg_recon_var = ((reconstruct * bg_mask) ** 2).sum() / (
                    bg_mask.sum() + 1e-8
                )
                bg_target_var = ((X * bg_mask) ** 2).sum() / (
                    bg_mask.sum() + 1e-8
                )
                loss_dict["bg_hallucination_ratio"] = bg_recon_var / (
                    bg_target_var + 1e-8
                )

                bottom_mask = (X <= p10).float()
                recon_bottom = (
                    reconstruct.abs() * bottom_mask
                ).sum() / (bottom_mask.sum() + 1e-8)
                target_bottom = (X.abs() * bottom_mask).sum() / (
                    bottom_mask.sum() + 1e-8
                )
                loss_dict["deletion_ratio"] = recon_bottom / (
                    target_bottom + 1e-8
                )

            if i % 100 == 0:
                print(
                    f"test step currently at : {i} / {len(test_dataloader)} \n"
                    f"avg_vae_loss : {total_vae_loss / max(i, 1)}\n"
                    f"loss_dict : {loss_dict} \n",
                    f"discriminator loss : {total_disc_loss / max(i, 1)}\n",
                )

    return total_vae_loss / len(test_dataloader), total_disc_loss / len(test_dataloader)


def train_vae(
    vae,
    discriminator,
    vae_opt,
    disc_opt,
    vae_loss,
    train_dataloader,
    test_dataloader,
    vae_lr_scheduler,
    disc_lr_scheduler,
    epochs,
    model_name,
    kl_annealing_scheduler,
    device,
    ema_decay=0.999,
):

    # implement caching here
    checkpoint_dir = Path("model_checkpoint")
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    result_checkpoint = checkpoint_dir / f"{model_name}_results_checkpoint.pkl"

    if result_checkpoint.is_file():
        with open(result_checkpoint, "rb") as rc:
            print("found results checkpoint, continuing")
            results = pypickle.load(rc)
    else:
        print("no checkpoint found, initializing empty results dictionary")
        results = {
            "vae_loss": [],
            "disc_loss": [],
            "test_vae_loss": [],
            "test_disc_loss": [],
        }

    model_checkpoint_path = checkpoint_dir / f"{model_name}_latest_epoch.pt"

    if model_checkpoint_path.exists():
        print("model checkpoint found, continuing from last saved statedict")
        checkpoint_states = torch.load(model_checkpoint_path, map_location=device)

        vae.load_state_dict(checkpoint_states["vae_state_dict"])

        discriminator.load_state_dict(checkpoint_states["disc_state_dict"])

        vae_opt.load_state_dict(checkpoint_states["vae_optimizer_state_dict"])

        disc_opt.load_state_dict(checkpoint_states["disc_optimizer_state_dict"])

        if (
            vae_lr_scheduler is not None
            and checkpoint_states.get("vae_lr_scheduler_state_dict") is not None
        ):
            vae_lr_scheduler.load_state_dict(
                checkpoint_states["vae_lr_scheduler_state_dict"]
            )
        if (
            disc_lr_scheduler is not None
            and checkpoint_states.get("disc_lr_scheduler_state_dict") is not None
        ):
            disc_lr_scheduler.load_state_dict(
                checkpoint_states["disc_lr_scheduler_state_dict"]
            )

        global_step = checkpoint_states["global_step"]

        checkpoint_epoch = checkpoint_states["epoch"] + 1

    else:
        print("no checkpoint found, starting training ")
        checkpoint_epoch = 0
        global_step = 0

    # instantiate scaler

    scaler = torch.amp.GradScaler("cuda")

    if (
        model_checkpoint_path.exists()
        and checkpoint_states.get("scaler_state_dict") is not None
    ):
        scaler.load_state_dict(checkpoint_states["scaler_state_dict"])

    # EMA copy of VAE weights for stable test-time metrics and previews
    ema_vae = copy.deepcopy(vae)
    ema_vae.eval()
    for p in ema_vae.parameters():
        p.requires_grad_(False)

    if (
        model_checkpoint_path.exists()
        and checkpoint_states.get("ema_vae_state_dict") is not None
    ):
        ema_vae.load_state_dict(checkpoint_states["ema_vae_state_dict"])

    # training loop
    with alive_bar(epochs - checkpoint_epoch, bar="fish") as bar:
        for epoch in range(checkpoint_epoch, epochs):
            vae_loss_avg, disc_loss, global_step = train_step(
                vae=vae,
                discriminator=discriminator,
                disc_opt=disc_opt,
                vae_opt=vae_opt,
                loss_fn=vae_loss,
                train_dataloader=train_dataloader,
                kl_scheduler=kl_annealing_scheduler,
                step=global_step,
                device=device,
                scaler=scaler,
                ema_vae=ema_vae,
                ema_decay=ema_decay,
            )

            # generate a few preview images on test set

            with torch.inference_mode():
                plt.figure(figsize=(16, 12))

                ema_vae.eval()
                x = next(iter(test_dataloader)).to(device)
                gen_img, _, _ = ema_vae(x)

                # latent space generation: sample z ~ N(0,1) and decode
                mu, log_var = ema_vae.encode(x)
                z_sampled = torch.randn_like(mu)
                latent_gen = ema_vae.decode(z_sampled)

                for i in range(4):
                    orig = x[i].cpu().permute(1, 2, 0).squeeze()
                    recon = gen_img[i].cpu().permute(1, 2, 0).squeeze()
                    lat = latent_gen[i].cpu().permute(1, 2, 0).squeeze()

                    plt.subplot(4, 3, i * 3 + 1)
                    plt.imshow(recon)
                    plt.title("vae reconstruction")
                    plt.axis("off")

                    plt.subplot(4, 3, i * 3 + 2)
                    plt.imshow(orig)
                    plt.title("input img")
                    plt.axis("off")

                    plt.subplot(4, 3, i * 3 + 3)
                    plt.imshow(lat)
                    plt.title("latent sampled")
                    plt.axis("off")

                plt.show(block=False)
                plt.pause(10)
                plt.close("all")

            test_vae_loss, test_disc_loss = test_step(
                vae=ema_vae,
                discriminator=discriminator,
                test_dataloader=test_dataloader,
                loss_fn=vae_loss,
                kl_scheduler=kl_annealing_scheduler,
                step=global_step,
                device=device,
            )

            print(
                f"epoch         : {epoch} ------------------------------------------\n",
                f"train vae_loss: {vae_loss_avg}     | train disc_loss : {disc_loss}\n",
                f"------------------------------------------------------------------\n",
                f"test vae_loss : {test_vae_loss} | test disc_loss : {test_disc_loss}",
            )

            results["vae_loss"].append(vae_loss_avg)
            results["disc_loss"].append(disc_loss)
            results["test_vae_loss"].append(test_vae_loss)
            results["test_disc_loss"].append(test_disc_loss)

            # scheduler step
            if vae_lr_scheduler is not None:
                vae_lr_scheduler.step()
            if disc_lr_scheduler is not None:
                disc_lr_scheduler.step()

            # checkpointing

            checkpoint_states = {
                "vae_state_dict": vae.state_dict(),
                "disc_state_dict": discriminator.state_dict(),
                "ema_vae_state_dict": ema_vae.state_dict(),
                "vae_optimizer_state_dict": vae_opt.state_dict(),
                "disc_optimizer_state_dict": disc_opt.state_dict(),
                "vae_lr_scheduler_state_dict": vae_lr_scheduler.state_dict()
                if vae_lr_scheduler is not None
                else None,
                "disc_lr_scheduler_state_dict": disc_lr_scheduler.state_dict()
                if disc_lr_scheduler is not None
                else None,
                "global_step": global_step,
                "epoch": epoch,
                "scaler_state_dict": scaler.state_dict(),
            }

            torch.save(checkpoint_states, model_checkpoint_path)

            if epoch >= 10:
                save_model_path = checkpoint_dir / "best_models" / model_name

                if not save_model_path.exists():
                    save_model_path.mkdir(parents=True, exist_ok=True)

                if min(results["test_vae_loss"]) == test_vae_loss:
                    torch.save(
                        checkpoint_states,
                        f"{save_model_path}/{model_name}_epoch_{epoch}_vae_test_loss_{test_vae_loss:.2f}.pt",
                    )

            with open(result_checkpoint, "wb") as f:
                pypickle.dump(results, f)
                print("saved results")

            bar()

    return results


def compute_latent_stats(vae, dataloader, device, latent_dim=8):
    """One-time pass. Returns per-channel (mean, std) of the GT latents, each shape [latent_dim]."""
    vae.eval()
    # accumulate in float64 for precision; reduce over batch + spatial, keep the CHANNEL dim
    ch_sum   = torch.zeros(latent_dim, dtype=torch.float64, device=device)
    ch_sumsq = torch.zeros(latent_dim, dtype=torch.float64, device=device)
    count = 0

    with torch.no_grad():
        for _inp, gt in dataloader:
            mu, log_var = vae.encode(gt.to(device))
            ch_sum += mu.sum(dim=(0,2,3)).double()
            ch_sumsq += (mu ** 2 ).sum(dim = (0,2,3)).double()
            count += mu.shape[0] * mu.shape[2] * mu.shape[3]

    mean = ch_sum / count
    var  = ch_sumsq / count - mean**2
    std  = var.clamp_min(1e-12).sqrt()
    return mean.float(), std.float()


def normalize_latent(z, mean, std):        # z:[B,C,H,W], mean/std:[C]
    return (z - mean[None, :, None, None]) / std[None, :, None, None]


def denormalize_latent(z_tilde, mean, std):  # inverse, for decoding samples later
    return z_tilde * std[None, :, None, None] + mean[None, :, None, None]


def latent_train_step(model, vae, diffusion, dataloader, optimizer, loss_fn,
                      mean, std, device, noise_steps, ema_model, decay):
    model.train()
    vae.eval()
    mean, std = mean.to(device), std.to(device)
    running = 0.0
    for i, data in enumerate(dataloader):
        
        inp, gt = data

        inp, gt = inp.to(device), gt.to(device)
        
        with torch.no_grad():
            mu_cond, _ = vae.encode(inp)
            mu_gt, log_var_gt = vae.encode(gt)
            z_gt = vae.reparameterization(mu_gt, log_var_gt)
        
        # normalize latent space
        z_cond = normalize_latent(mu_cond, mean, std)
        
        z_target = normalize_latent(z_gt, mean, std)

        # prepare latent data 
        t = torch.randint(0, noise_steps, (z_target.shape[0],), device = device)
        x_t, noise = diffusion.forward_diffusion(z_target, t)
        model_in = torch.cat([z_cond, x_t], dim = 1)
        
        preds = model(model_in, t)
        
        loss = loss_fn(preds, noise)
        
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        for ep, p in zip(ema_model.parameters(), model.parameters()): 
            ep.mul_(decay).add_(p, alpha=1-decay)


        running += loss.cpu().item()

        if i % 100 == 0:
            print(
                f"currently at : {i} / {len(dataloader)} \n"
                f"avg_train_loss : {running / max(i, 1)}\n"
            )

    return running / len(dataloader)

def latent_test_step(model, vae, diffusion, dataloader, loss_fn, mean, std, device, noise_steps):
    model.eval() ; vae.eval() 

    mean, std = mean.to(device), std.to(device)

    running = 0.0 
    with torch.inference_mode():
        for i, data in enumerate(dataloader):
            X, y = data 

            X, y = X.to(device), y.to(device)

            mu_cond, _ = vae.encode(X)
            mu_target, log_var_gt = vae.encode(y)
            z_target = vae.reparameterization(mu_target, log_var_gt)

            # normalize latent space
            z_cond = normalize_latent(mu_cond, mean, std)
            
            z_target = normalize_latent(z_target, mean, std)

            t = torch.randint(0, noise_steps, (z_target.shape[0],), device = device)
            x_t, noise = diffusion.forward_diffusion(z_target, t)
            model_in = torch.cat([z_cond, x_t], dim = 1)
            
            preds = model(model_in, t)
            
            loss = loss_fn(preds, noise)

            running += loss.cpu().item()

            if i % 100 == 0:
                print(
                f"currently at : {i} / {len(dataloader)} \n"
                f"avg_test_loss : {running / max(i, 1)}\n"
            ) 
    
    return running / len(dataloader)

def train_diffusion(
    model,
    vae,
    optimizer,
    loss_fn,
    train_dataloader,
    test_dataloader,
    scheduler,
    epochs,
    mean,
    std,
    noise_steps,
    diffusion,
    model_name,
    device,
    ema_decay = 0.999 
):

    # implement caching here
    checkpoint_dir = Path("model_checkpoint")
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    result_checkpoint = checkpoint_dir / f"{model_name}_results_checkpoint.pkl"

    if result_checkpoint.is_file():
        with open(result_checkpoint, "rb") as rc:
            print("found results checkpoint, continuing")
            results = pypickle.load(rc)
    else:
        print("no checkpoint found, initializing empty results dictionary")
        results = {
            "train_loss": [],
            "test_loss" : []
        }

    model_checkpoint_path = checkpoint_dir / f"{model_name}_latest_epoch.pt"

    if model_checkpoint_path.exists():
        print("model checkpoint found, continuing from last saved statedict")
        checkpoint_states = torch.load(model_checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint_states["model_state_dict"])

        optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])

        if scheduler:
            scheduler.load_state_dict(checkpoint_states["scheduler_state_dict"])

        checkpoint_epoch = checkpoint_states["epoch"] + 1

    else:
        print("no checkpoint found, starting training ")
        checkpoint_epoch = 0


    ema_model = copy.deepcopy(vae)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    if (
        model_checkpoint_path.exists()
        and checkpoint_states.get("ema_vae_state_dict") is not None
    ):
        ema_model.load_state_dict(checkpoint_states["ema_model_state_dict"])

    # training loop
    with alive_bar(epochs - checkpoint_epoch, bar="fish") as bar:
        for epoch in range(checkpoint_epoch, epochs):
            train_loss = latent_train_step(
                model = model,
                vae=vae,
                optimizer = optimizer,
                loss_fn=loss_fn,
                dataloader=train_dataloader,
                mean = mean,
                std = std,
                noise_steps = noise_steps,
                diffusion = diffusion,
                device=device,
                ema_model = ema_model,
                decay = ema_decay
            )

            # generate a few preview images on test set
            # INFERENCE NEEDS TO BE REWRITTEN
            

            test_loss = latent_test_step(
                model = model, 
                vae = vae, 
                diffusion = diffusion,
                loss_fn = loss_fn,
                dataloader = test_dataloader,
                mean = mean,
                std = std,
                noise_steps = noise_steps,
                device = device
            )

            print(
                f"epoch         : {epoch} ------------------------------------------\n",
                f"train loss: {train_loss}\n",
                f"------------------------------------------------------------------\n",
                f"test loss : {test_loss}\n",
            )

            results["train_loss"].append(train_loss)
            results["test_loss"].append(test_loss)


            # scheduler step

            if scheduler:
                scheduler.step()

            # checkpointing

            checkpoint_states = {
                "model_state_dict"    : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "ema_model_state_dict": ema_model.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
                if scheduler is not None
                else None,
                "epoch": epoch,

            }

            torch.save(checkpoint_states, model_checkpoint_path)

            if epoch >= 10:
                save_model_path = checkpoint_dir / "best_models" / model_name

                if not save_model_path.exists():
                    save_model_path.mkdir(parents=True, exist_ok=True)

                if min(results["test_loss"]) == test_loss:
                    torch.save(
                        checkpoint_states,
                        f"{save_model_path}/{model_name}_epoch_{epoch}_vae_test_loss_{test_loss:.2f}.pt",
                    )

            with open(result_checkpoint, "wb") as f:
                pypickle.dump(results, f)
                print("saved results")

            bar()

    return results