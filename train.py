# add KL annealing scheduler
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
):

    total_disc_loss, total_vae_loss = 0.0, 0.0

    for i, X in tqdm(enumerate(train_dataloader)):
        vae.train()
        discriminator.train()

        X = X.to(device)

        with torch.autocast(device_type=device):
            reconstruct, mu, log_var = vae(X)

        # train discriminator
        disc_opt.zero_grad()

        disc_recon = reconstruct.detach()

        with torch.autocast(device_type=device):
            fake_score = discriminator(disc_recon)
            real_score = discriminator(X)

            loss_real, loss_fake = loss.hinge_discriminator(real_score, fake_score)
            disc_loss = loss_real + loss_fake

        scaler.scale(disc_loss).backward()
        scaler.step(disc_opt)

        # disc_opt.step()

        # train vae
        vae_opt.zero_grad()

        kl_w = kl_scheduler.get_value(current_step=step)

        with torch.autocast(device_type=device):
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
                f"avg_vae_loss : {total_vae_loss / max(i, 1)}"
                f"loss_dict : {loss_dict} \n",
                f"discriminator loss : {total_disc_loss / max(i, 1)}",
            )

        step += 1

        avg_disc_loss = total_disc_loss / len(train_dataloader)
        avg_vae_loss = total_vae_loss / len(train_dataloader)

        scaler.update()

    return avg_vae_loss, avg_disc_loss, step


def test_step(vae, discriminator, test_dataloader, loss_fn, kl_scheduler, step, device):

    vae.eval()
    discriminator.eval()

    total_disc_loss, total_vae_loss = 0.0, 0.0

    with torch.inference_mode():
        for i, X in tqdm(enumerate(test_dataloader)):
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

            if i % 100 == 0:
                print(
                    f"test step currently at : {i} / {len(test_dataloader)} \n"
                    f"avg_vae_loss : {total_vae_loss / max(i, 1)}"
                    f"loss_dict : {loss_dict} \n",
                    f"discriminator loss : {total_disc_loss / max(i, 1)}",
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
            )

            # generate a few preview images on test set

            with torch.inference_mode():
                plt.figure(figsize=(16, 8))

                vae.eval()
                x = next(iter(test_dataloader)).to(device)
                gen_img, _, _ = vae(x)

                for i in range(4):
                    orig = x[i].cpu().permute(1, 2, 0).squeeze()
                    recon = gen_img[i].cpu().permute(1, 2, 0).squeeze()

                    plt.subplot(4, 2, i * 2 + 1)
                    plt.imshow(recon)
                    plt.title("generated by vae")
                    plt.axis("off")

                    plt.subplot(4, 2, i * 2 + 2)
                    plt.imshow(orig)
                    plt.title("input img")
                    plt.axis("off")

                plt.show(block=False)
                plt.pause(10)
                plt.close("all")

            test_vae_loss, test_disc_loss = test_step(
                vae=vae,
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
