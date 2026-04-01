import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ToricAE(nn.Module):
    def __init__(self, input_dim=24, latent_angles=2, hidden_dim=64, alpha=1.0):
        super().__init__()
        self.latent_angles = latent_angles
        self.coord_dim = latent_angles * 2 
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(alpha=alpha),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(alpha=alpha),
            nn.Linear(hidden_dim, latent_angles)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.coord_dim, hidden_dim),
            nn.ELU(alpha=alpha),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(alpha=alpha),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        angles = self.encoder(x)
        z_coords = []
        for i in range(self.latent_angles):
            z_coords.append(torch.cos(angles[:, i]).unsqueeze(1))
            z_coords.append(torch.sin(angles[:, i]).unsqueeze(1))
            
        z = torch.cat(z_coords, dim=1) # shape: [batch, 4]
        recon_x = self.decoder(z)
        
        mse_loss = F.mse_loss(recon_x, x)
        
        return {
            'recon': recon_x,
            'latent': z,
            'angles': angles,
            'total_loss': mse_loss,
            'recon_loss': mse_loss
        }
