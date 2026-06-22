"""
Run the REAL sampler with a real checkpoint + real conditioning, and measure
accuracy. Confirms whether the trained model actually explodes / gives bad FID,
or whether the earlier FID-293 run was just a wrong/half-loaded checkpoint.

    uv run python sample_test.py --ckpt results/epoch1000.pt --cfg 3
"""
import os
import sys
import json
from argparse import ArgumentParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch

from model.DDPM import DDPM
from model.condition_encoder.MultiHotEncoder import MultiHotEncoder
from evaluator import evaluation_model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}, cfg_scale = {args.cfg}")

    ddpm = DDPM(device=str(device), cfg_scale=args.cfg, thresholding=args.thresholding, dim=args.dim).to(device)
    cond_enc = MultiHotEncoder("objects.json", 256).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    ddpm.load_state_dict(ckpt["model"])           # strict: must match arch
    cond_enc.load_state_dict(ckpt["condition_encoder"])
    ddpm.eval(); cond_enc.eval()
    print(f"loaded {args.ckpt} (strict)")

    with open(args.prompts, "r") as f:
        labels = json.load(f)
    print(f"{len(labels)} prompts from {args.prompts}, e.g. {labels[0]}")

    with torch.no_grad():
        cond_emb = cond_enc(labels)
        onehot = cond_enc(labels, pure=True)
        imgs, _ = ddpm.sample(len(labels), cond_emb)   # the real sampler (prints x_t trajectory)

    print(f"\nfinal imgs: std={imgs.std().item():.3f} "
          f"min={imgs.min().item():+.2f} max={imgs.max().item():+.2f}")

    evaluator = evaluation_model()
    acc = evaluator.eval(imgs, onehot)
    print(f"\n*** classification accuracy = {acc:.4f} ***")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--ckpt", default="results/epoch1000.pt")
    p.add_argument("--prompts", default="new_test.json")
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--thresholding", default="static", choices=["static", "dynamic", "none"])
    p.add_argument("--dim", type=int, default=64)
    main(p.parse_args())
