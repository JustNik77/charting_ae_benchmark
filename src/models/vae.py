import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, alpha=1.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(alpha=alpha),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(alpha=alpha)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(alpha=alpha),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(alpha=alpha),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.shape[0]
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
        total_loss = recon_loss + 0.001 * kld_loss  # 0.1 for Beta-VAE effect
        
        return {
            'recon': recon_x,
            'latent': mu, # return mu for visualization
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kld_loss': kld_loss
        }
