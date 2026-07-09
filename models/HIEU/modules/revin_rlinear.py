import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(1, 1, num_features))
        self.register_buffer('running_std', torch.ones(1, 1, num_features))

    def forward(self, x: torch.Tensor, mode: str):
        if mode == 'norm':
            mean = x.mean(dim=(1), keepdim=True)
            std = x.std(dim=(1), keepdim=True, unbiased=False) + self.eps
            self.running_mean = mean.detach()
            self.running_std = std.detach()
            x = (x - mean) / std
            if self.affine:
                x = x * self.weight + self.bias
            return x
        elif mode == 'denorm':
            x = x.clone()
            if self.affine:
                x = (x - self.bias) / (self.weight + self.eps)
            x = x * self.running_std + self.running_mean
            return x
        else:
            raise ValueError('mode must be norm or denorm')

class RLinearCore(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, channels: int, individual: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.individual = individual
        if individual:
            self.heads = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(channels)])
        else:
            self.head = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.individual:
            out = torch.zeros(x.size(0), self.pred_len, self.channels, device=x.device, dtype=x.dtype)
            for i in range(self.channels):
                out[:, :, i] = self.heads[i](x[:, :, i])
            return out
        else:
            return self.head(x.permute(0, 2, 1)).permute(0, 2, 1)

class RGRLCore(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, channels: int = 1):
        super().__init__()
        self.revin = RevIN(channels, affine=True)
        self.linear = RLinearCore(seq_len, pred_len, channels, individual=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.revin(x, mode='norm')
        y = self.linear(x)
        y = self.revin(y, mode='denorm')
        return y
