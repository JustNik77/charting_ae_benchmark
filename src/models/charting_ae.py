import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsemax import Sparsemax

def get_encoder(d_input, d_latent, layer_sz, alpha=1.0):
    return nn.Sequential(
        nn.Linear(d_input, layer_sz),
        nn.ELU(alpha=alpha),
        nn.Linear(layer_sz, layer_sz),
        nn.ELU(alpha=alpha),
        nn.Linear(layer_sz, layer_sz),
        nn.ELU(alpha=alpha),
        nn.Linear(layer_sz, d_latent)
    )

def get_decoder(d_latent, d_output, layer_sz, alpha=1.0):
    return nn.Sequential(
        nn.Linear(d_latent, layer_sz),
        nn.ELU(alpha=alpha),
        nn.Linear(layer_sz, layer_sz),
        nn.ELU(alpha=alpha),
        nn.Linear(layer_sz, layer_sz),
        nn.ELU(alpha=alpha),
        nn.Linear(layer_sz, d_output)
    )

class ChartingAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, n_charts, alpha=1.0, gamma=1.0, beta=1.0):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.encoders = nn.ModuleList([get_encoder(input_dim, latent_dim, hidden_dim, alpha) for _ in range(n_charts)])
        self.decoders = nn.ModuleList([get_decoder(latent_dim, input_dim, hidden_dim, alpha) for _ in range(n_charts)])
        self.proba = nn.Sequential(nn.Linear(input_dim, n_charts), Sparsemax(1))

    def forward(self, x, *args):
        z = torch.stack([enc(x) for enc in self.encoders], dim=1)
        p = self.proba(x)

        x_recons = torch.stack([dec(z[:, i]) for i, dec in enumerate(self.decoders)], dim=1)
        recon_x = torch.sum(x_recons * p.unsqueeze(-1), dim=1)

        recon_errors = ((x.unsqueeze(1) - x_recons)**2).sum(dim=-1)
        recon_loss = (p * recon_errors).sum(dim=-1).mean()

        q = F.softmax(-recon_errors, dim=-1)
        trans_loss = -(q * torch.log(p + 1e-8)).sum(dim=-1).mean()

        mean_p = p.mean(dim=0)
        target = torch.ones_like(mean_p) / len(self.encoders)
        nondom_loss = ((mean_p - target)**2).sum()
        total_loss = recon_loss + self.gamma * trans_loss + self.beta * nondom_loss
        mse_loss = F.mse_loss(recon_x, x)

        return {
            'recon': recon_x,
            'latent': z,
            'probabilities': p,
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'nondom_loss': nondom_loss,
            'transition_loss': trans_loss,
            'mse_loss': mse_loss
        }
