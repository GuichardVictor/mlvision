from typing import Any

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder


def init_dataloaders(
    train_root: str,
    val_root: str,
    batch_size: int,
    num_workers: int,
    transforms: dict[str, Any],
    rank: int,
    world_size: int,
):
    train_dataset = ImageFolder(train_root, transform=transforms["train"])
    val_dataset = ImageFolder(val_root, transform=transforms["val"])

    train_sampler = DistributedSampler(
        dataset=train_dataset, num_replicas=world_size, rank=rank
    )

    val_sampler = DistributedSampler(
        dataset=train_dataset, num_replicas=world_size, rank=rank
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
        prefetch_factor=2,
    )

    return train_loader, val_loader
