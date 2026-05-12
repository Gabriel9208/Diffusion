from dataclasses import dataclass
import os
import sys

# Ensure the project root is on sys.path so that evaluator.py can be imported
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from torchmetrics.image.fid import FrechetInceptionDistance
from model.condition_encoder import BaseConditionEncoder
import tqdm
import torch
from torchvision.utils import save_image, make_grid

from evaluator import evaluation_model

@dataclass
class RunnerContext:
    model: torch.nn.Module | None = None
    condition_encoder: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    epoch: int = 0
    total_epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    fid: float = 0.0
    best_fid: float = 0.0

class Runner:
    def __init__(self, 
                model,
                condition_encoder: BaseConditionEncoder,
                train_loader,
                val_loader,
                #test_loader,
                optimizer,
                scheduler,
                device,
                total_epoch,
                validate_every_epoch,
                cfg_p_uncond,
                callbacks = None,
                resume = False,
                resume_path = None,
                save_img_dir=None
                ):
        self.model = model
        self.condition_encoder = condition_encoder
        self.train_loader = train_loader
        self.valid_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.callbacks = callbacks or []
        self.fid = FrechetInceptionDistance(feature=2048).to(self.device)
        
        self.evaluator = evaluation_model()

        self.resume = resume
        self.total_epoch = total_epoch
        self.cfg_p_uncond = cfg_p_uncond
        self.validate_every_epoch = validate_every_epoch

        self.context = RunnerContext(
            model=model, 
            condition_encoder=condition_encoder,
            optimizer=optimizer, 
            scheduler=scheduler, 
            epoch=0, 
            total_epoch=total_epoch, 
            train_loss=0.0,
            val_loss=0.0,
            fid=float('inf'),
            best_fid=float('inf')
        )

        self.save_img_dir = save_img_dir

        if resume:
            self.load_checkpoint(resume_path)

    def _emit(self, hook_name, *args, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, hook_name, None)

            if method is not None:
                method(*args, **kwargs)            

    def _forward(self, x_0, text_embedding):
        x_0 = x_0.to(self.device)
        loss = self.model(x_0, text_embedding)
            
        return loss

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _embed_text(self, text_prompt, training=True, pure=False):        
        embeddings = self.condition_encoder(text_prompt, pure=pure)
        embeddings = embeddings.to(self.device)

        if training and self.cfg_p_uncond > 0:
            B = embeddings.shape[0]
            mask = (torch.rand(B, 1, device=self.device) < self.cfg_p_uncond).float()
            embeddings = embeddings * (1 - mask)

        return embeddings
    
    def _denormalize(self, x):
        x = (x + 1) / 2 * 255
        return x.clamp(0, 255).round().to(torch.uint8)

    def calculate_fid(self, real_imgs, gen_imgs, batch_size=32):        
        N = real_imgs.shape[0]

        for i in range(0, N, batch_size):
            end = i + batch_size
            if end > N:
                end = N
                
            r = real_imgs[i:end].to(self.device)
            g = gen_imgs[i:end].to(self.device)
            self.fid.update(r, real=True)
            self.fid.update(g, real=False)

        total_score = self.fid.compute().item()
        self.fid.reset()
        
        return total_score

    def _train(self):
        total_loss = 0
        
        for x_0, text_prompt in tqdm.tqdm(self.train_loader):
            x_0 = x_0.to(self.device)
            self._emit("on_step_begin", self.context)
            
            cond_emb = self._embed_text(text_prompt)
                
            loss = self._forward(x_0, cond_emb)
            total_loss += loss.item()

            self._backward(loss)

            
            self._emit("on_step_end", self.context)
        
        total_loss /= len(self.train_loader)
        self.context.train_loss = total_loss

        print(f"Train Loss: {total_loss:.4f}")
        
    @torch.no_grad()
    def _validate(self):
        total_loss = 0
        real_imgs = []
        gen_imgs = []
        
        for x_0, text_prompt in tqdm.tqdm(self.valid_loader):
            x_0 = x_0.to(self.device)
            self._emit("on_step_begin", self.context)
            
            cond_emb = self._embed_text(text_prompt, training=False)

            loss = self._forward(x_0, cond_emb)
            total_loss += loss.item()
            
            imgs, _ = self.model.sample(x_0.shape[0], cond_emb)

            x_0 = self._denormalize(x_0).cpu()
            imgs = self._denormalize(imgs).cpu()
            
            real_imgs.append(x_0)
            gen_imgs.append(imgs)

            self._emit("on_step_end", self.context)
        
        total_loss /= len(self.valid_loader)

        self.context.val_loss = total_loss

        real_imgs = torch.cat(real_imgs, dim=0)
        gen_imgs = torch.cat(gen_imgs, dim=0)

        score = self.calculate_fid(real_imgs, gen_imgs)
        self.context.fid = score
        print(f"Val Loss: {total_loss:.4f}, FID: {score:.4f}")            
      
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        if 'condition_encoder' in checkpoint and checkpoint['condition_encoder'] is not None:
            self.condition_encoder.load_state_dict(checkpoint['condition_encoder'])
            print("condition_encoder loaded successfully")
        else:
            print("condition_encoder not found")
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.context.epoch = checkpoint['epoch']
        if 'best_fid' in checkpoint:
            self.context.best_fid = checkpoint['best_fid']
        print(f"Checkpoint {path} loaded successfully")
        return self.context.epoch
        
    def run(self):
            
        self._emit("on_run_begin", self.context)
        for epoch in range(self.context.epoch + 1, self.total_epoch + 1):
            print(f'Running Epoch: {epoch}/{self.total_epoch}')
            self._emit("on_epoch_begin", self.context)

            self.context.epoch = epoch

            self._train() 

            if epoch % self.validate_every_epoch == 0 and epoch != 0:
                self._validate()

            self.scheduler.step()
                            
            self._emit("on_epoch_end", self.context, isBest = (self.context.fid < self.context.best_fid))
            self.context.best_fid = min(self.context.best_fid, self.context.fid)

        self._emit("on_run_end", self.context)

    def test(self, text_prompts):
        total_acc = 0
        N = len(text_prompts)

        cond_emb = self._embed_text(text_prompts, training=False)
        imgs, _ = self.model.sample(1, cond_emb)
        onehot = self._embed_text(text_prompts, training=False, pure=True)
        acc = self.evaluator.eval(imgs, onehot)

        return acc
            
    def sample(self, prompts, file_name):
        label_emb = self._embed_text(prompts, training=False)
        print(f"cond_emb: sum={label_emb.sum().item():.4f}, std={label_emb.std().item():.4f}")

        imgs, _ = self.model.sample(1, label_emb)

        save_image(
                    imgs.cpu(), 
                    os.path.join(self.save_img_dir, file_name),
                    normalize=True,
                    value_range=(-1, 1),
                )
        return self._denormalize(imgs[0]).permute(1, 2, 0).cpu()

    def uncoinditional_sample(self):

        imgs, _ = self.model.sample(1, torch.zeros((1, 256), device=self.device))
        return self._denormalize(imgs[0]).permute(1, 2, 0).cpu()

    def make_grid_img(self, prompts, file_name):
        label_emb = self._embed_text(prompts, training=False)
        print(f"cond_emb: sum={label_emb.sum().item():.4f}, std={label_emb.std().item():.4f}")

        imgs, _ = self.model.sample(len(prompts), label_emb)
        for i, img in enumerate(imgs):
            save_image(
                img.cpu(),
                os.path.join(self.save_img_dir, f"{i}.png"),
                normalize=True,
                value_range=(-1, 1),
            )
            
        onehot = self._embed_text(prompts, training=False, pure=True)
        acc = self.evaluator.eval(imgs, onehot)
        print(f"Accuracy: {acc:.4f}")

        imgs_grid = make_grid(imgs.cpu(), nrow=8, padding=2, normalize=True, value_range=(-1, 1))
        save_image(
                    imgs_grid, 
                    os.path.join(self.save_img_dir, file_name),
                )
        
        return acc

    def make_denoising_img(self, prompts, file_name):
        label_emb = self._embed_text(prompts, training=False)
        print(f"cond_emb: sum={label_emb.sum().item():.4f}, std={label_emb.std().item():.4f}")

        imgs, denoise_process = self.model.sample(len(prompts), label_emb)

        denoise_process = make_grid(denoise_process.cpu(), nrow=8, padding=2, normalize=True, value_range=(-1, 1))
        save_image(
                    denoise_process, 
                    os.path.join(self.save_img_dir, file_name),
                )