import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = instantiate(cfg.model)
    checkpoint = torch.load(cfg.save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    test_data_path = cfg.data_path.replace("_train.npy", "_test.npy")
    data = np.load(test_data_path)
    tensor_data = torch.from_numpy(data).float()

    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    total_mse = 0.0
    total_mae = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            outputs = model(x)
            recon = outputs['recon']
            total_mse += F.mse_loss(recon, x, reduction='sum').item()
            total_mae += F.l1_loss(recon, x, reduction='sum').item()

    n_samples = len(tensor_data) * cfg.model.input_dim
    final_mse = total_mse / n_samples
    final_mae = total_mae / n_samples
    
    print("\n" + "="*40)
    print(f"Dataset: {test_data_path}")
    print(f"Model: {cfg.model._target_}")
    print(f"TEST MSE: {final_mse:.4f}")
    print(f"TEST MAE: {final_mae:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
