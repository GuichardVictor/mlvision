import os
import logging

import torch
import torch.distributed as dist

logger = logging.getLogger()


def initialize(port=40000, rank_and_world_size=(None, None)):
    # If Using DDP multi Node
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    # If using local multi processing
    rank, world_size = rank_and_world_size
    os.environ["MASTER_ADDR"] = "localhost"

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
        except Exception:
            logger.info("SLURM vars not set (distributed training not available)")
            world_size, rank = 1, 0
            return world_size, rank

    try:
        os.environ["MASTER_PORT"] = str(port)
        backend = "gloo" if os.environ["CUDA_VISIBLE_DEVICES"] == "" else "nccl"
        torch.distributed.init_process_group(
            backend=backend, world_size=world_size, rank=rank
        )
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f"distributed training not available {e}")

    return world_size, rank


class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
