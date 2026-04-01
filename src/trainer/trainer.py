import torch
import torch.optim as optim
import logging
import os
from collections import defaultdict
import math
from src.models.charting_ae import ChartingAE

log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, dataloader, cfg):
        self.cfg = cfg
        self.device = self.cfg.device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)

    def train(self):
        self.model.train()
        history = defaultdict(list)

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_metrics = defaultdict(float)
            if isinstance(self.model, ChartingAE):
                self.model.beta = math.exp(-epoch / 20)

            for batch in self.dataloader:
                x = batch[0].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(x)

                loss = outputs['total_loss']
                loss.backward()
                self.optimizer.step()

                for k, v in outputs.items():
                    if 'loss' in k:
                        epoch_metrics[k] += v.item()

            log_str = f"Epoch {epoch:03d} "
            for k, v in epoch_metrics.items():
                avg_val = v / len(self.dataloader)
                history[k].append(avg_val)
                log_str += f"| {k}: {avg_val:.4f} "

            log.info(log_str)
            
        return history

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        log.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log.info(f"Checkpoint loaded from {path}")
