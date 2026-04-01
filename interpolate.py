import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import numpy as np
import matplotlib.pyplot as plt


def interpolate_latent(z_a, z_b, steps=10):
    device = z_a.device
    alphas = torch.linspace(0, 1, steps, device=device)
    view_shape = [steps] + [1] * z_a.dim()
    alphas = alphas.view(*view_shape)
    return z_a.unsqueeze(0) * (1 - alphas) + z_b.unsqueeze(0) * alphas


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

    idx_a, idx_b = 60, 72
    x_a = tensor_data[idx_a:idx_a+1].to(device)
    x_b = tensor_data[idx_b:idx_b+1].to(device)
    
    steps = 10
    
    with torch.no_grad():
        out_a = model(x_a)
        out_b = model(x_b)
        
        is_charting = 'probabilities' in out_a
        
        if is_charting:
            z_interp = interpolate_latent(out_a['latent'][0], out_b['latent'][0], steps)
            p_interp = interpolate_latent(out_a['probabilities'][0], out_b['probabilities'][0], steps)

            trajectories = []
            for i in range(steps):
                z_t = z_interp[i].unsqueeze(0)
                p_t = p_interp[i].unsqueeze(0)
                p_t = p_t / p_t.sum(dim=-1, keepdim=True)
                
                x_recons = torch.stack([dec(z_t[:, j]) for j, dec in enumerate(model.decoders)], dim=1)
                recon_x = torch.sum(x_recons * p_t.unsqueeze(-1), dim=1)
                trajectories.append(recon_x.squeeze().cpu().numpy())
                title = "Interpolation (Charting AE)"
        else:
            z_interp = interpolate_latent(out_a['latent'][0], out_b['latent'][0], steps)
            trajectories = []
            for i in range(steps):
                recon_x = model.decoder(z_interp[i].unsqueeze(0))
                trajectories.append(recon_x.squeeze().cpu().numpy())
                title = "Interpolation (VAE)"
                
    fig = plt.figure(figsize=(14, 6))
    time_steps = np.arange(trajectories[0].shape[0])

    ax1 = fig.add_subplot(121, projection='3d')
    for i, traj in enumerate(trajectories):
        ax1.plot(time_steps, traj, zs=i, zdir='y', color=plt.cm.viridis(i / steps), lw=2)
        
    ax1.set_xlabel('Time (Window)')
    ax1.set_ylabel('Interpolation Step')
    ax1.set_zlabel('Amplitude')
    ax1.view_init(elev=25, azim=-55) 
    ax1.set_title(f"{title} (3D)")

    ax2 = fig.add_subplot(122)
    for i, traj in enumerate(trajectories):
        label = "Start" if i == 0 else ("End" if i == steps-1 else None)
        alpha = 1.0 if (i == 0 or i == steps-1) else 0.5
        ax2.plot(time_steps, traj, color=plt.cm.viridis(i / steps), lw=2, alpha=alpha, label=label)
        
    ax2.set_xlabel('Time (Window)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f"{title} (2D Overlay)")
    ax2.legend()
    
    plt.tight_layout()
    save_name = "interp_charting.png" if is_charting else "interp_vae.png"
    plt.savefig(save_name, dpi=300)
    print(f"saved {save_name}")

if __name__ == "__main__":
    main()
