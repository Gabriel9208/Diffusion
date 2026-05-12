from typing import Any
import os
import torch
import wandb

from engine.Runner import RunnerContext


class Callback():    
    def on_run_begin(self, ctx: RunnerContext, **kwargs): ...
    def on_run_end(self, ctx: RunnerContext, **kwargs): ...
    def on_epoch_begin(self, ctx: RunnerContext, **kwargs): ...
    def on_epoch_end(self, ctx: RunnerContext, **kwargs): ...
    def on_step_begin(self, ctx: RunnerContext, **kwargs): ...
    def on_step_end(self, ctx: RunnerContext, **kwargs): ...

class CheckpointCallback(Callback):
    def __init__(self, path, best_path):
        super().__init__()
        self.path = path
        self.best_path = best_path

    def _update_path(self, epoch:int):
        self.path = os.path.join(os.path.dirname(self.path), f"epoch{epoch}.pt")

    def _save_checkpoint(self, ctx: RunnerContext):
        if ctx.epoch <= 600 and ctx.epoch % 10 != 0:
            return

        self._update_path(ctx.epoch)
        torch.save({
            'model': ctx.model.state_dict() if ctx.model is not None else None,
            'condition_encoder': ctx.condition_encoder.state_dict() if ctx.condition_encoder is not None else None,
            'optimizer': ctx.optimizer.state_dict() if ctx.optimizer is not None else None,
            'scheduler': ctx.scheduler.state_dict() if ctx.scheduler is not None else None,
            'epoch': ctx.epoch,
            'train_loss': ctx.train_loss,
            'val_loss': ctx.val_loss,
            'fid': ctx.fid,
            'best_fid': ctx.best_fid
        }, self.path)

        print("Checkpoint saved at: ", self.path)
    
        
    def on_epoch_end(self, ctx: RunnerContext, isBest=False, **kwargs):
        self._save_checkpoint(ctx)
        
        
        if isBest:
            self._save_checkpoint(ctx)
            print("Best checkpoint saved at: ", self.best_path)

class LoggerCallback(Callback):
    def __init__(self, run_name, config, wandb_id=None):
        super().__init__()
        self.run_name = None
        if run_name:
            self.run_name = run_name
            self.config = config
            wandb.init(
                entity="gabrieee-national-taiwan-university-of-science-and-techn",
                project="LAB6",
                config=config,
                name=run_name,
                id=wandb_id,
                resume="allow" if wandb_id else None
            )

    def on_epoch_end(self, ctx: RunnerContext, **kwargs):
        if not self.run_name:
            return
            
        wandb.log({
            "train_loss": ctx.train_loss,
            "val_loss": ctx.val_loss,
            "fid": ctx.fid
        })

class MetricCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []
        
    def on_epoch_end(self, ctx: RunnerContext, **kwargs):
        self.metrics.append(ctx.metrics['fid'])

    