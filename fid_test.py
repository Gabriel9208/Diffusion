"""
Recompute val-set FID for a checkpoint at one or more cfg scales, replicating
Runner._validate's FID (same transform, same val split, same denormalize).

    uv run python fid_test.py --ckpt results/epoch1000.pt --cfgs 1 3
"""
import os
import sys
import json
import contextlib
import io
from argparse import ArgumentParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance

from model.DDPM import DDPM
from model.condition_encoder.MultiHotEncoder import MultiHotEncoder
from dataset.ICLVERDataset import ICLVERDataset
from utils.LetterBox import LetterBox
from utils.tarin_valid_split import split


def collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), list(labels)


def denorm(x):
    return ((x + 1) / 2 * 255).clamp(0, 255).round().to(torch.uint8)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}, ckpt = {args.ckpt}, cfgs = {args.cfgs}")

    transform = transforms.Compose([
        LetterBox(target_size=(64, 64), fill_color=(255, 255, 255)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    _, val_dict = split(json.load(open("train.json")))
    val_loader = DataLoader(ICLVERDataset(val_dict, "iclevr", transforms=transform),
                            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=4)
    print(f"val images: {len(val_dict)}")

    enc = MultiHotEncoder("objects.json", 256).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    enc.load_state_dict(ck["condition_encoder"]); enc.eval()

    for cfg in args.cfgs:
        ddpm = DDPM(device=str(device), cfg_scale=cfg, thresholding=args.thresholding, dim=args.dim).to(device)
        ddpm.load_state_dict(ck["model"]); ddpm.eval()
        fid = FrechetInceptionDistance(feature=2048).to(device)

        with torch.no_grad():
            for x0, prompts in val_loader:
                x0 = x0.to(device)
                cond = enc(prompts)
                with contextlib.redirect_stdout(io.StringIO()):   # silence sample debug
                    gen, _ = ddpm.sample(x0.shape[0], cond)
                fid.update(denorm(x0), real=True)
                fid.update(denorm(gen), real=False)

        score = fid.compute().item()
        print(f"  cfg={cfg}: FID = {score:.2f}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--ckpt", default="results/epoch1000.pt")
    p.add_argument("--cfgs", type=float, nargs="+", default=[1, 3])
    p.add_argument("--thresholding", default="static", choices=["static", "dynamic", "none"])
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    main(p.parse_args())
