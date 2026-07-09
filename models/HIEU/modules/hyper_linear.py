import torch
import torch.nn as nn

class LowRankAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.randn(out_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, in_dim) * 0.01)

    def forward(self, W_base: torch.Tensor):
        # W_base: [out_dim, in_dim]
        return W_base + self.A @ self.B

class HyperLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, rank: int, ctx_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(pred_len, seq_len))
        nn.init.xavier_uniform_(self.W)
        self.adapter = LowRankAdapter(seq_len, pred_len, rank)
        self.hyper = nn.Sequential(
            nn.Linear(ctx_dim, 64), nn.ReLU(),
            nn.Linear(64, rank * (seq_len + pred_len))
        )
        self.rank = rank
        self.seq_len = seq_len
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor, ctx: torch.Tensor):
        # x: [B, L, 1] -> treat channel=1, per coin
        B, L, C = x.shape
        assert C == 1
        h = self.hyper(ctx)  # [B, rank*(L+S)]
        A_delta = h[:, : self.rank * self.pred_len].view(B, self.pred_len, self.rank)
        B_delta = h[:, self.rank * self.pred_len :].view(B, self.rank, self.seq_len)
        W_eff = self.W.unsqueeze(0) + A_delta @ B_delta  # [B, S, L]
        y = (W_eff @ x.squeeze(-1).unsqueeze(-1)).squeeze(-1)  # [B, S]
        return y.unsqueeze(-1), W_eff
