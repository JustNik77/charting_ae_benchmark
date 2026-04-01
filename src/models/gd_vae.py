import torch
import torch.nn as nn
import torch.nn.functional as F

class GDVAE(nn.Module):
    def __init__(self, input_dim=24, latent_dim=3, hidden_dim=64, alpha=1.0):
        super().__init__()
        self.window_size = input_dim - 1 
        
        self.encoder = nn.Sequential(
            nn.Linear(self.window_size, hidden_dim),
            nn.ELU(alpha=alpha),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(alpha=alpha),
            nn.Linear(hidden_dim, input_dim)
        )

        self.dynamics_A = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, x):
        x_past = x[:, :-1]
        x_future = x[:, 1:]

        z_past = self.encoder(x_past)
        z_future_target = self.encoder(x_future.detach())

        z_future_pred = self.dynamics_A(z_past)
        recon_x = self.decoder(z_past)
        
        recon_loss = F.mse_loss(recon_x, x)

        dynamics_loss = F.mse_loss(z_future_pred, z_future_target)
        total_loss = recon_loss + 10.0 * dynamics_loss
        
        return {
            'recon': recon_x,
            'latent': z_past,
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'dynamics_loss': dynamics_loss
        }
