"""
Diagnostic for the FID-293 / exploding-sampler problem.

It bisects the question: "did the model learn to denoise, or is the sampler
feeding it something different from training?" by replaying the *training*
forward path (q_forward + model) at several fixed timesteps and reporting:

  - eps_pred std  (at high t the model should predict eps with std ~= 1)
  - MSE(eps_pred, eps_true)               -> the actual training objective
  - reconstructed x0 std / MSE            -> is denoising usable for sampling
  - per-t loss sweep                      -> is the 0.0115 "average" hiding bad high-t?
  - fp32 vs bf16-autocast                 -> train(bf16) / sample(fp32) mismatch?
  - |model(label) - model(0)|             -> is conditioning actually wired in?

Run from repo root:
    uv run python diagnose.py --ckpt checkpoint.pth --n 8
"""
import os
import sys
import json
from argparse import ArgumentParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from model.DDPM import DDPM
from model.condition_encoder.MultiHotEncoder import MultiHotEncoder
from dataset.ICLVERDataset import ICLVERDataset
from utils.LetterBox import LetterBox
from utils.tarin_valid_split import split


def collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), list(labels)


def load_real_batch(n, device):
    transform = transforms.Compose([
        LetterBox(target_size=(64, 64), fill_color=(255, 255, 255)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    with open("train.json", "r") as f:
        json_file = json.load(f)
    _, val_dict = split(json_file)
    ds = ICLVERDataset(val_dict, "iclevr", transforms=transform)
    loader = DataLoader(ds, batch_size=n, shuffle=False, collate_fn=collate_fn)
    x0, labels = next(iter(loader))
    return x0.to(device), labels


def noised(ddpm, x0, t_int, eps):
    """Replay training's q_forward at a single integer timestep."""
    t = torch.full((x0.shape[0], 1), int(t_int), device=x0.device, dtype=torch.long)
    x_t = ddpm.q_forward(x0, t, eps)
    return x_t, t


@torch.no_grad()
def per_t_report(ddpm, x0, label_emb, ts, autocast):
    mode = "bf16-autocast (training condition)" if autocast else "fp32 (sampling condition)"
    print(f"\n=== eps-prediction replay  [{mode}] ===")
    print(f"{'t':>5} | {'eps_pred.std':>12} | {'eps_true.std':>12} | "
          f"{'eps_MSE':>9} | {'x0hat.std':>9} | {'x0_MSE':>8}")
    for t_int in ts:
        eps = torch.randn_like(x0)
        x_t, t = noised(ddpm, x0, t_int, eps)
        ctx = torch.autocast(device_type=x0.device.type, dtype=torch.bfloat16) if autocast \
            else torch.autocast(device_type=x0.device.type, enabled=False)
        with ctx:
            eps_pred = ddpm.model(x_t, t, label_emb).float()
        sab = ddpm.sqrt_alpha_bar[t_int]
        somab = ddpm.sqrt_one_minus_alpha_bar[t_int]
        x0_hat = (x_t - somab * eps_pred) / sab
        eps_mse = torch.mean((eps_pred - eps) ** 2).item()
        x0_mse = torch.mean((x0_hat - x0) ** 2).item()
        print(f"{t_int:>5} | {eps_pred.std().item():>12.3f} | {eps.std().item():>12.3f} | "
              f"{eps_mse:>9.4f} | {x0_hat.std().item():>9.3f} | {x0_mse:>8.3f}")


@torch.no_grad()
def loss_sweep(ddpm, x0, label_emb, n_buckets=10):
    """Bucketed training loss over the whole t range -> is high-t actually bad?"""
    print("\n=== training loss vs timestep (is 0.0115 average hiding bad high-t?) ===")
    edges = torch.linspace(0, ddpm.T, n_buckets + 1, dtype=torch.long)
    print(f"{'t range':>14} | {'eps_MSE':>9}")
    for i in range(n_buckets):
        lo, hi = int(edges[i]), int(edges[i + 1])
        mids = torch.randint(lo, max(hi, lo + 1), (x0.shape[0], 1), device=x0.device)
        eps = torch.randn_like(x0)
        sab = ddpm.sqrt_alpha_bar[mids].unsqueeze(2).unsqueeze(3)
        somab = ddpm.sqrt_one_minus_alpha_bar[mids].unsqueeze(2).unsqueeze(3)
        x_t = sab * x0 + somab * eps
        eps_pred = ddpm.model(x_t, mids, label_emb).float()
        mse = torch.mean((eps_pred - eps) ** 2).item()
        print(f"{lo:>5}-{hi:<5} | {mse:>9.4f}")


@torch.no_grad()
def conditioning_report(ddpm, x0, label_emb, ts):
    print("\n=== conditioning effect: |model(label) - model(0)| ===")
    print(f"label_emb: std={label_emb.std().item():.3f} "
          f"abs_mean={label_emb.abs().mean().item():.3f} "
          f"(all-zero={torch.count_nonzero(label_emb).item() == 0})")
    zero_emb = torch.zeros_like(label_emb)
    print(f"{'t':>5} | {'|cond-uncond| mean':>18}")
    for t_int in ts:
        eps = torch.randn_like(x0)
        x_t, t = noised(ddpm, x0, t_int, eps)
        cond = ddpm.model(x_t, t, label_emb)
        uncond = ddpm.model(x_t, t, zero_emb)
        print(f"{t_int:>5} | {(cond - uncond).abs().mean().item():>18.5f}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    ddpm = DDPM(device=str(device), dim=args.dim).to(device)
    cond_enc = MultiHotEncoder("objects.json", 256).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model_sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    try:
        ddpm.load_state_dict(model_sd)
        print(f"loaded model weights from {args.ckpt} (strict)")
    except RuntimeError as e:
        print("!! strict load FAILED -- checkpoint likely predates the UNet refactor.")
        print("!! Retrying strict=False; results may be meaningless if many keys differ.\n", str(e)[:400])
        ddpm.load_state_dict(model_sd, strict=False)
    if isinstance(ckpt, dict) and ckpt.get("condition_encoder") is not None:
        try:
            cond_enc.load_state_dict(ckpt["condition_encoder"])
            print("loaded condition_encoder weights")
        except RuntimeError:
            print("!! condition_encoder weights did not match; using fresh init")

    ddpm.eval()
    cond_enc.eval()

    x0, labels = load_real_batch(args.n, device)
    print(f"real batch: x0 shape={tuple(x0.shape)} std={x0.std().item():.3f} "
          f"min={x0.min().item():+.2f} max={x0.max().item():+.2f}")
    print(f"example label[0] = {labels[0]}")
    with torch.no_grad():
        label_emb = cond_enc(labels)

    ts = [999, 800, 500, 200, 100, 50, 10]
    per_t_report(ddpm, x0, label_emb, ts, autocast=False)
    if device.type == "cuda":
        per_t_report(ddpm, x0, label_emb, ts, autocast=True)
    loss_sweep(ddpm, x0, label_emb)
    conditioning_report(ddpm, x0, label_emb, ts)

    print("\n=== how to read this ===")
    print("- At t=999, eps_pred.std should be ~1 and eps_MSE small. If eps_pred.std<<1")
    print("  (like 0.09) the model ignores the input at high t -> sampler will explode.")
    print("- If fp32 and bf16 columns differ a lot -> train(bf16)/sample(fp32) mismatch.")
    print("- If loss sweep is low for ALL t buckets but high-t replay still fails ->")
    print("  the bug is how the sampler feeds t/conditioning, not the weights.")
    print("- If |cond-uncond|~0 with a non-zero label_emb -> conditioning not wired into AdaGN.")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--ckpt", default="checkpoint.pth")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--dim", type=int, default=64)
    main(p.parse_args())
