import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import numpy as np
import matplotlib.pyplot as plt

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = instantiate(cfg.model)
    checkpoint = torch.load(cfg.save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    data = np.load(cfg.data_path)
    tensor_data = torch.from_numpy(data).float()

    with torch.no_grad():
        outputs = model(tensor_data.to(device))

    recon = outputs['recon'].cpu().numpy()
    is_charting = 'probabilities' in outputs

    plt.figure(figsize=(10, 4))
    plt.plot(data[0], label="Original")
    plt.plot(recon[0], label="Reconstructed", linestyle="--")
    plt.legend()
    plt.savefig("recon_plot.png")
    plt.close()

    if is_charting:
        z = outputs['latent'].cpu().numpy()
        p = outputs['probabilities'].cpu().numpy()
        
        n_charts = p.shape[1]
        best_chart_idx = np.argmax(p, axis=1)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        z_best = z[np.arange(len(z)), best_chart_idx]
        
        scatter = ax.scatter(z_best[:, 0], z_best[:, 1], z_best[:, 2], 
                             c=best_chart_idx, cmap='tab10', s=2, alpha=0.8)
        plt.legend(*scatter.legend_elements(), title="Charts")
        plt.savefig("latent_argmax.png")
        plt.close()

        fig = plt.figure(figsize=(15, 10))
        cols = min(n_charts, 4)
        rows = int(np.ceil(n_charts / cols))

        for i in range(n_charts):
            ax = fig.add_subplot(rows, cols, i+1, projection='3d')
            mask = best_chart_idx == i 

            if np.any(mask):
                z_chart = z[mask, i, :]
                time_color = np.arange(len(z))[mask] 

                ax.scatter(z_chart[:, 0], z_chart[:, 1], z_chart[:, 2], 
                           c=time_color, cmap='viridis', s=5, alpha=0.9)
                ax.set_title(f"Chart {i} (N={np.sum(mask)} pts)")
            else:
                ax.set_title(f"Chart {i} (Empty)")

        plt.tight_layout()
        plt.savefig("latent_atlas_split.png")
        plt.close()
        
        print(f"Plots saved: recon_plot.png, latent_argmax.png, latent_atlas_split.png")

    else:
        z_global = outputs['latent'].cpu().numpy()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        time_color = np.arange(len(z_global))
        scatter = ax.scatter(z_global[:, 0], z_global[:, 1], z_global[:, 2], c=time_color, cmap='viridis', s=2, alpha=0.5)

        plt.savefig("latent_3d_vae.png")
        plt.close()
        print(f"Plots saved: recon_plot.png, latent_3d_vae.png")

if __name__ == "__main__":
    main()
