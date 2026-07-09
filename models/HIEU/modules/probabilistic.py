import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantileHead(nn.Module):
    def __init__(self, pred_len: int, quantiles):
        super().__init__()
        self.quantiles = quantiles
        self.head = nn.Linear(pred_len, pred_len * len(quantiles))

    def forward(self, y_point: torch.Tensor):
        # y_point: [B, S, 1]
        B, S, _ = y_point.shape
        q = self.head(y_point.squeeze(-1))  # [B, S*q]
        q = q.view(B, S, len(self.quantiles))
        # ensure monotonic quantiles via cumulative sort
        q, _ = torch.sort(q, dim=2)
        return q

def pinball_loss(pred_q: torch.Tensor, y: torch.Tensor, quantiles):
    # pred_q: [B, S, Q], y: [B, S, 1]
    loss = 0.0
    for i, tau in enumerate(quantiles):
        e = y.squeeze(-1) - pred_q[:, :, i]
        loss += torch.mean(torch.maximum(tau * e, (tau - 1) * e))
    return loss / len(quantiles)

# CRPS approximation from discrete quantiles (Jordan et al. 2019 style):
# CRPS ≈ 2 * sum_k w_k * pinball(q_k)
# with trapezoidal weights over quantile grid

def crps_from_quantiles(pred_q: torch.Tensor, y: torch.Tensor, quantiles):
    B, S, Q = pred_q.shape
    taus = torch.tensor(quantiles, device=pred_q.device, dtype=pred_q.dtype)
    # trapezoidal weights
    w = torch.zeros_like(taus)
    w[0] = (taus[1] - taus[0]) / 2
    for k in range(1, Q - 1):
        w[k] = (taus[k + 1] - taus[k - 1]) / 2
    w[-1] = (taus[-1] - taus[-2]) / 2
    # accumulate pinball at each tau with weights
    crps = 0.0
    for i in range(Q):
        e = y.squeeze(-1) - pred_q[:, :, i]
        pin = torch.mean(torch.maximum(taus[i] * e, (taus[i] - 1) * e))
        crps = crps + w[i] * pin
    return 2.0 * crps
