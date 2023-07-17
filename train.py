import argparse
import multiprocessing as mp

import yaml

import mlvision.distributed
import mlvision.train

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="path of config file to load")
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cuda:0"],
    help="which devices to use on the local machine",
)


def main(rank, config_path, world_size, devices):
    import os

    # Keep only one gpu or cpu
    if str(devices[rank]) == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

    import logging

    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    params = None
    with open(config_path, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    world_size, rank = mlvision.distributed.initialize(
        rank_and_world_size=(rank, world_size)
    )
    logger.info(f"Running on rank: {rank}/{world_size}")

    mlvision.train.main(params)


if __name__ == "__main__":
    args = parser.parse_args()

    num_gpus = len(args.devices)
    mp.set_start_method("spawn")  # Fork does not allow gpu

    for rank in range(num_gpus):
        mp.Process(
            target=main, args=(rank, args.config, num_gpus, args.devices)
        ).start()
