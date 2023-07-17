import argparse
import yaml
import multiprocessing as mp

from mlvision.models import build_model
import torch
import torch.nn

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, help="Path to config file")
parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")


def load_checkpoint(checkpoint_path, map_location):
    state_dict = torch.load(checkpoint_path, map_location=map_location)["model"]

    # Remove DDP
    return {key.replace("module.", ""): param for key, param in state_dict.items()}


def main(config_path, checkpoint_path):
    params = None
    with open(config_path, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(params["model"]["architecture"], params["model"]["args"]).to(
        device
    )
    model.load_state_dict(load_checkpoint(checkpoint_path, map_location=device))

    print(model)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.config, args.checkpoint)
