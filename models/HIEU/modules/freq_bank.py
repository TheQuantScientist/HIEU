import torch
import torch.nn as nn
import torch.nn.functional as F

class FIRBand(nn.Module):
    def __init__(self, channels: int, kernel: int):
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel, padding=padding, groups=channels, bias=False)

    def forward(self, x):
        # x: [B, L, C]
        y = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        return y

class FrequencyBank(nn.Module):
    def __init__(self, channels: int, num_bands: int, kernel: int):
        super().__init__()
        self.bands = nn.ModuleList([FIRBand(channels, kernel) for _ in range(num_bands)])
        self.gate = nn.Parameter(torch.ones(num_bands))

    def forward(self, x):
        outs = [band(x) for band in self.bands]
        w = torch.softmax(self.gate, dim=0)
        y = 0
        for i, o in enumerate(outs):
            y = y + w[i] * o
        return y, outs, w
