import logging

import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import mlvision.distributed

from mlvision.distributed import AllReduce
from mlvision.logger import AverageMeter, CSVLogger, gpu_timer

from mlvision.augmentation import build_transforms
from mlvision.datasets import init_dataloaders
from mlvision.optim import init_optimizers
from mlvision.models import build_model

logger = logging.getLogger()


def save_checkpoint(
    rank,
    checkpoint_path,
    model,
    optimizer,
    scaler,
    epoch,
    loss_meter,
    world_size,
    lr,
):
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "loss": loss_meter.avg,
        "world_size": world_size,
        "lr": lr,
    }

    if rank == 0:
        torch.save(save_dict, checkpoint_path)


def train_loop(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    dataloaders: dict[str, DataLoader],
    device: torch.device,
    epochs: int,
    rank: int,
    world_size: int,
    checkpoint_path: str,
    log_freq: int,
    log_file: str | None,
):
    logger.info(f"start training for {epochs} epochs")

    csv_logger = None
    if log_file:
        csv_logger = CSVLogger(
            log_file,
            ("%d", "epoch"),
            ("%d", "itr"),
            ("%.5f", "loss"),
            ("%s", "stage"),
        )

    start_epoch = 0
    num_epochs = epochs - start_epoch

    for epoch in range(start_epoch, num_epochs):
        for stage in ["train", "val"]:
            loss_meter = AverageMeter()
            time_meter = AverageMeter()

            dataloaders[stage].sampler.set_epoch(epoch)

            def is_train():
                return stage == "train"

            def autocast_enabled():
                return scaler is not None

            for i, (x, y) in enumerate(dataloaders[stage]):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                def step():
                    def compute_loss():
                        out = model(x)
                        loss = criterion(out, y)
                        loss = AllReduce.apply(loss)
                        return loss

                    if is_train():
                        model.train()
                        new_lr = scheduler.step()
                        optimizer.zero_grad()
                    else:
                        model.eval()
                        new_lr = 0.0

                    with torch.inference_mode(not is_train()), torch.autocast(
                        device_type=device.type, enabled=autocast_enabled()
                    ):
                        loss = compute_loss()

                    if is_train():
                        if autocast_enabled():
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                    return loss.item(), new_lr

                (loss, new_lr), etime = gpu_timer(step)

                loss_meter.update(loss)
                time_meter.update(etime)

                if (i % log_freq) == log_freq - 1:
                    logger.info(
                        "[%d %5d] %s_loss: %.3f [lr: %.2e] (%.1f ms)"
                        % (epoch + 1, i, stage, loss_meter.avg, new_lr, time_meter.avg)
                    )

                    if csv_logger:
                        csv_logger.log(epoch + 1, i, loss, stage)

            if not is_train():
                save_checkpoint(
                    rank=rank,
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    loss_meter=loss_meter,
                    world_size=world_size,
                    lr=new_lr,
                )


def main(params):
    experiment_name = params["experiment"]

    # -- Data Augmentation
    use_horizontal_flip = params["data"].get("use_horizontal_flip", False)
    use_vertical_flip = params["data"].get("use_vertical_flip", False)
    crop_size = params["data"].get("crop_size", 224)
    # --

    # -- Dataset
    batch_size = params["data"]["batch_size"]
    train_root = params["data"]["train_root"]
    val_root = params["data"]["val_root"]
    num_workers = params["data"]["num_workers"]
    # --

    # -- Model
    architecture = params["model"]["architecture"]
    model_args = params["model"]["args"]
    # --

    # -- Optimization
    epochs = params["optimization"]["epochs"]
    start_lr = params["optimization"].get("start_lr", 2e-4)
    ref_lr = params["optimization"].get("lr", 1e-3)
    final_lr = params["optimization"].get("final_lr", 1e-6)
    warmup_steps = params["optimization"].get("warmup", 40)
    ipe_scale = params["optimization"].get("ipe_scale", 1.0)
    use_float16 = params["optimization"]["use_float16"]
    compile_model = params["optimization"].get("compile", False)
    # --

    # -- Logging
    log_freq = params["logging"]["log_frequency"]
    log_file = params["logging"].get("log_file")
    # --

    world_size, rank = mlvision.distributed.initialize()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # == Creating data loaders ==
    transforms = build_transforms(crop_size, use_horizontal_flip, use_vertical_flip)
    train_loader, val_loader = init_dataloaders(
        train_root,
        val_root,
        batch_size,
        num_workers,
        transforms,
        rank,
        world_size,
    )

    # == Creating Model ==
    model = build_model(architecture, model_args).to(device)
    model = DistributedDataParallel(model, static_graph=True)
    if compile_model:
        model = torch.compile(model)

    # == Creating Optimizer ==
    (optimizer, scheduler, scaler) = init_optimizers(
        model,
        start_lr,
        ref_lr,
        num_epochs=epochs,
        warmup_steps=warmup_steps,
        iterations_per_epoch=len(train_loader),
        final_lr=final_lr,
        use_float16=use_float16,
        ipe_scale=ipe_scale,
    )

    # == Train ==
    train_loop(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        dataloaders={"train": train_loader, "val": val_loader},
        device=device,
        epochs=epochs,
        rank=rank,
        world_size=world_size,
        checkpoint_path=f"./trained_models/{experiment_name}_train_checkpoint.tar",
        log_freq=log_freq,
        log_file=log_file,
    )
