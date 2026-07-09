import torch
from .modules.probabilistic import pinball_loss, crps_loss

class HIEULoss:
    def __init__(self, lambda_q=0.2, lambda_crps=0.0, lambda_lap=1e-3, lambda_ssl=0.1):
        self.lq = lambda_q
        self.lc = lambda_crps
        self.ll = lambda_lap
        self.ls = lambda_ssl

    def __call__(self, y_point, y_true, pred_q=None, quantiles=None, lap_smooth=None, ssl_terms=None):
        # point loss (MSE)
        point = torch.mean((y_point - y_true) ** 2)
        loss = point
        if pred_q is not None and quantiles is not None:
            loss += self.lq * pinball_loss(pred_q, y_true, quantiles)
            if self.lc > 0:
                loss += self.lc * crps_loss(pred_q, y_true, quantiles)
        if lap_smooth is not None:
            loss += self.ll * lap_smooth
        if ssl_terms is not None:
            loss += self.ls * sum(ssl_terms)
        return loss, {
            'point': point.item(),
            'lap': float(lap_smooth) if lap_smooth is not None else 0.0,
            'ssl': sum([t.item() for t in ssl_terms]) if ssl_terms is not None else 0.0
        }
