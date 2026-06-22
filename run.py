import os
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import cv2

from engine.Runner import Runner
from engine.Callback import CheckpointCallback, LoggerCallback
from model.DDPM import DDPM
from model.condition_encoder.MultiHotEncoder import MultiHotEncoder
from dataset.ICLVERDataset import ICLVERDataset
from utils.LetterBox import LetterBox
from utils.tarin_valid_split import split


def collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), list(labels)

def sample(runner):
    img_tensor = runner.sample([["red cube", "green cube"]])
    img = img_tensor.numpy()
    
    cropped_img = img[8:56, :, :]
    
    restored_img = cv2.resize(cropped_img, (320, 240), interpolation=cv2.INTER_CUBIC)
    
    restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
    
    cv2.imshow("Image", restored_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sweep_test(runner):
    with open("test.json", "r") as f:
        json_file = json.load(f)
    
    maximum = 0
    for i in range(882, 957):
        runner.load_checkpoint(f"./results/latest_new_{i}.pt")
        
        acc = runner.test(json_file)

        if acc > maximum:
            maximum = acc
            print(f"New maximum: {acc} from {i}")

def make_grid(runner, file_name):
    with open(file_name, "r") as f:
        json_file = json.load(f)
    
    runner.make_grid_img(json_file, f"grid.png")

    
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    transform = transforms.Compose([
        LetterBox(target_size=(64, 64), fill_color=(255, 255, 255)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])  

    with open("train.json", "r") as f:
        json_file = json.load(f)

    train_dict, val_dict = split(json_file)

    dataset = ICLVERDataset(train_dict, "iclevr", transforms=transform)
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_dataset = ICLVERDataset(val_dict, "iclevr", transforms=transform)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    model = DDPM(
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        T=args.T,
        device=device,
        cfg_scale=args.cfg_scale,
        dim=args.dim,
    ).to(device)

    ema_model = None
    if args.use_ema:
        ema_model = DDPM(
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            T=args.T,
            device=device,
            cfg_scale=args.cfg_scale,
            dim=args.dim,
        ).to(device)

    condition_encoder = MultiHotEncoder("objects.json", 256).to(device)
 
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(condition_encoder.parameters()), 
        lr=args.lr,
        weight_decay=args.weight_decay,
    ) 

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    scaler = torch.amp.GradScaler(device)

    callbacks = [
        CheckpointCallback(args.latest_path, args.best_path),
        LoggerCallback(args.run_name, args, wandb_id=args.wandb_id)
    ]

    runner = Runner(
        model=model, 
        ema_model=ema_model,
        condition_encoder=condition_encoder,
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        scaler=scaler,        
        device=device, 
        total_epoch=args.epochs, 
        validate_every_epoch=args.validate_every_epoch,
        cfg_p_uncond=args.cfg_p_uncond,
        ema_beta=args.ema_beta,
        callbacks=callbacks,
        save_img_dir=args.save_img_dir,
        resume=args.resume,
        resume_path=args.resume_path,
    )

    if args.mode == "train":
        runner.run()
    elif args.mode == "sweep":
        sweep_test(runner)
    elif args.mode == "grid":
        make_grid(runner, args.test_path)
    elif args.mode == "denoising":
        runner.make_denoising_img([["red sphere", "cyan cylinder", "cyan cube"]], "denoising.png")

    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--ema-beta', type=float, default=0.995)
    parser.add_argument('--beta-start', type=float, default=1e-4)
    parser.add_argument('--beta-end', type=float, default=0.02)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--dim', type=int, default=128)            # UNet base width (multiple of 32)
    parser.add_argument('--cfg-scale', type=float, default=1)
    parser.add_argument('--cfg-p-uncond', type=float, default=0.1)
    parser.add_argument('--validate-every-epoch', type=int, default=1)
    

    parser.add_argument('--latest-path', type=str, default='./results/latest.pt')
    parser.add_argument('--best-path', type=str, default='./results/best.pt')
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--save-img-dir', type=str, default='./results_img/')
    
    parser.add_argument('--wandb-id', type=str, default=None, help='WandB run ID to resume')
    parser.add_argument('--resume', action='store_true') 
    parser.add_argument('--resume-path', type=str, default="./results/latest.pt")
    
    parser.add_argument('--mode', type=str, default="train") # train, sweep, grid, denoising
    parser.add_argument('--test-path', type=str, default="")
    args = parser.parse_args()
        
    if not os.path.exists(args.save_img_dir):
        os.makedirs(args.save_img_dir)

    main(args)
    
# example to train:
# python run.py --latest-path ./results/latest_cfg_film.pt --best-path ./results/best_cfg_film.pt --save-img-dir ./images/run/ --validate-every-epoch 10 --mode train
