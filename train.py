import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import json
from sklearn.preprocessing import StandardScaler

from src.trainer.trainer import Trainer

log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    
    data = np.load(cfg.data_path)
    tensor_data = torch.from_numpy(data).float()
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=cfg.trainer.batch_size, shuffle=True)
    
    model = instantiate(cfg.model)
    trainer = Trainer(model, dataloader, cfg.trainer)
    
    if cfg.get("load_path"):
        trainer.load_checkpoint(cfg.load_path)

    history = trainer.train()
    trainer.save_checkpoint(cfg.save_path)
    
    with open("metrics.json", "w") as f:
        json.dump(history, f, indent=4)
    log.info("Metrics saved to metrics.json")

if __name__ == "__main__":
    main()
