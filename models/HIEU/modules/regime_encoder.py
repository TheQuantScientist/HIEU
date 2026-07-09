import torch
import torch.nn as nn
import torch.nn.functional as F

class RegimeEncoder(nn.Module):
    def __init__(self, in_channels: int, num_regimes: int, latent_dim: int, temperature: float = 1.0):
        super().__init__()
        self.num_regimes = num_regimes
        self.temperature = temperature
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Linear(64, latent_dim)
        self.logits_head = nn.Linear(latent_dim, num_regimes)
        # lightweight decoder for reconstruction (SSL)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU())

    def forward(self, x: torch.Tensor):
        # x: [B, L, C] -> [B, C, L]
        h = self.encoder(x.permute(0, 2, 1)).squeeze(-1)   # [B, 64]
        z = self.proj(h)                                   # [B, D]
        logits = self.logits_head(z)                       # [B, K]
        # gumbel-softmax routing (soft gate)
        gate = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)
        # recon signal (placeholder: map back to feature space center)
        recon = self.decoder(z)                            # [B, 64]
        return z, logits, gate, recon

    @staticmethod
    def ssl_losses(z_q: torch.Tensor, z_k: torch.Tensor, recon: torch.Tensor):
        # Simple InfoNCE-style contrastive with cosine similarity
        z_q = F.normalize(z_q, dim=-1)
        z_k = F.normalize(z_k, dim=-1)
        logits = z_q @ z_k.t()
        labels = torch.arange(z_q.size(0), device=z_q.device)
        contrastive_loss = F.cross_entropy(logits, labels)
        # simple reconstruction target = zero (placeholder)
        recon_loss = (recon.pow(2)).mean()
        return contrastive_loss, recon_loss
