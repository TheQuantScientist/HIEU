import torch
import torch.nn as nn
import torch.nn.functional as F

class TentAdapter:
    def __init__(self, params, lr=1e-4, steps=1):
        self.params = list(params)
        self.opt = torch.optim.Adam(self.params, lr=lr)
        self.steps = steps

    def entropy_min_step(self, outputs: torch.Tensor):
        # outputs: [B, S, 1] -> turn into pseudo-prob via softmax across horizon
        p = torch.softmax(outputs.squeeze(-1), dim=-1)
        ent = -(p * (p + 1e-8).log()).sum(dim=-1).mean()
        loss = ent
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
