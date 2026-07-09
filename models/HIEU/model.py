
import torch
import torch.nn as nn
import torch.nn.functional as F


class HIEUConfig:

    def __init__(self):
        # IO
        self.model_version = 'HIEU-v3-gated-experts'
        self.seq_len = 96
        self.pred_len = 96
        self.num_nodes = 5  # number of assets
        self.batch_size = 32
        
        # Training
        self.learning_rate = 5e-4
        self.weight_decay = 2e-4
        self.epochs = 80
        self.huber_beta = 0.5
        self.pinball_weight = 0.05
        self.direction_weight = 0.0
        
        # Regime encoder
        self.num_regimes = 4
        self.regime_dim = 64
        self.regime_temp = 1.0
        
        # Graph
        self.graph_hidden = 128
        
        # Frequency bank
        self.num_bands = 5
        self.band_kernel = 15
        self.temporal_hidden = 128
        self.temporal_dropout = 0.15
        
        # HyperLinear
        self.linear_rank = 12
        self.asset_dim = 24
        self.gate_hidden = 128
        self.ma_kernel = 25
        
        # Probabilistic
        self.quantiles = [0.1, 0.5, 0.9]


class RevIN(nn.Module):
    """Reversible Instance Normalization for distribution shift handling"""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True, unbiased=False) + self.eps
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.gamma + self.beta
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            x = x * self.std + self.mean
        return x


class IndividualLinear(nn.Module):
    """Independent temporal projection per asset."""
    def __init__(self, seq_len, pred_len, channels):
        super().__init__()
        self.pred_len = pred_len
        self.channels = channels
        self.heads = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(channels)
        ])
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)

    def forward(self, x):
        # x: [B, L, N]
        ys = [self.heads[i](x[:, :, i]).unsqueeze(-1) for i in range(self.channels)]
        return torch.cat(ys, dim=-1)


class RGRLCore(nn.Module):
    """Reversible Graph-Regularized Linear Core"""
    def __init__(self, seq_len, pred_len, channels):
        super().__init__()
        self.revin = RevIN(channels)
        self.linear = IndividualLinear(seq_len, pred_len, channels)
    
    def forward(self, x):
        # x: [B, L, N]
        x = self.revin(x, mode='norm')
        x = self.linear(x)
        x = self.revin(x, mode='denorm')
        return x


class NLinearCore(nn.Module):
    """NLinear-style expert: predict deviations from the last context value."""
    def __init__(self, seq_len, pred_len, channels):
        super().__init__()
        self.pred_len = pred_len
        self.linear = IndividualLinear(seq_len, pred_len, channels)

    def forward(self, x):
        last = x[:, -1:, :].detach()
        y = self.linear(x - last)
        return y + last.expand(-1, self.pred_len, -1)


class MovingAverage(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        return self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)


class DLinearCore(nn.Module):
    """DLinear-style expert with separate trend and residual projections."""
    def __init__(self, seq_len, pred_len, channels, kernel_size=25):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)
        self.seasonal = IndividualLinear(seq_len, pred_len, channels)
        self.trend = IndividualLinear(seq_len, pred_len, channels)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return self.seasonal(seasonal) + self.trend(trend)


class RegimeEncoder(nn.Module):
    """
    Regime Encoder - Identifies latent market states
    Outputs interpretable regime probabilities via Gumbel-Softmax
    """
    def __init__(self, in_channels, num_regimes=4, latent_dim=64, temperature=1.0):
        super().__init__()
        self.num_regimes = num_regimes
        self.temperature = temperature
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.proj = nn.Linear(64, latent_dim)
        self.logits_head = nn.Linear(latent_dim, num_regimes)
        self.decoder = nn.Linear(latent_dim, 64)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: [B, L, C] -> [B, C, L]
        h = self.encoder(x.permute(0, 2, 1)).squeeze(-1)
        h = self.dropout(h)
        z = self.proj(h)
        logits = self.logits_head(z)
        gate = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)
        recon = self.decoder(z)
        return z, logits, gate, recon


class DynamicGraph(nn.Module):
    """
    Dynamic Graph - Learns time-evolving cross-asset adjacency matrix
    Provides interpretable graph heatmaps
    """
    def __init__(self, num_nodes, hidden_dim=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        self.raw_A_base = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.05)
        
        self.adj_generator = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_nodes * num_nodes)
        )
        
        self.ctx_mlp = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.dynamic_scale = nn.Parameter(torch.tensor(0.1))
    
    def get_adjacency(self, features):
        B = features.shape[0]
        
        A_base = torch.tanh(self.raw_A_base)
        A_base = A_base - torch.diag_embed(torch.diag(A_base))
        
        delta = self.adj_generator(features).view(B, self.num_nodes, self.num_nodes)
        delta = torch.tanh(delta) * self.dynamic_scale
        
        A = A_base.unsqueeze(0) + delta
        mask = 1.0 - torch.eye(self.num_nodes, device=A.device).unsqueeze(0)
        A = A * mask
        
        return A
    
    def forward(self, features):
        ctx = self.ctx_mlp(features)
        A = self.get_adjacency(features)
        return ctx, A


class FrequencyBank(nn.Module):
    """
    Frequency Bank - Multi-scale temporal decomposition
    Provides interpretable frequency gating weights
    """
    def __init__(self, channels, num_bands=5, kernel=15):
        super().__init__()
        self.num_bands = num_bands
        
        self.filters = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=kernel, 
                     padding=kernel//2, groups=channels, bias=False)
            for _ in range(num_bands)
        ])
        
        self.gate = nn.Parameter(torch.ones(num_bands) / num_bands)
        
        for i, filt in enumerate(self.filters):
            nn.init.normal_(filt.weight, mean=0, std=0.1 / (i + 1))
    
    def forward(self, x):
        B, L, N = x.shape
        x_t = x.permute(0, 2, 1)
        
        bands = []
        for filt in self.filters:
            band = filt(x_t)
            bands.append(band.permute(0, 2, 1))
        
        bands = torch.stack(bands, dim=1)
        weights = F.softmax(self.gate, dim=0)
        x_fused = (weights.view(1, -1, 1, 1) * bands).sum(dim=1)
        
        return x_fused, bands, weights


class TemporalRefiner(nn.Module):
    """Depthwise temporal residual block for local momentum and volatility cues."""
    def __init__(self, channels, hidden_dim=128, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, channels, kernel_size=1),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        y = self.net(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x + self.scale * y


class HyperLinear(nn.Module):
    """
    HyperLinear - Context-adaptive weight generation via low-rank decomposition
    Core innovation: generates sample-specific weights conditioned on context
    """
    def __init__(self, seq_len, pred_len, rank=8, ctx_dim=128):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.rank = rank
        
        self.W_base = nn.Parameter(torch.zeros(pred_len, seq_len))
        nn.init.xavier_uniform_(self.W_base, gain=0.1)
        
        self.hyper = nn.Sequential(
            nn.Linear(ctx_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, rank * (seq_len + pred_len))
        )
        
        self.adapt_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x, ctx):
        B, L, _ = x.shape
        
        h = self.hyper(ctx)
        A_delta = h[:, :self.rank * self.pred_len].view(B, self.pred_len, self.rank)
        B_delta = h[:, self.rank * self.pred_len:].view(B, self.rank, self.seq_len)
        
        delta_W = self.adapt_scale * (A_delta @ B_delta)
        W_eff = self.W_base.unsqueeze(0) + delta_W
        
        y = (W_eff @ x.squeeze(-1).unsqueeze(-1)).squeeze(-1)
        
        return y.unsqueeze(-1), W_eff


class QuantileHead(nn.Module):
    """Quantile Head for probabilistic forecasting"""
    def __init__(self, pred_len, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.linear = nn.Linear(pred_len, pred_len * self.num_quantiles)
    
    def forward(self, x):
        B, S, _ = x.shape
        out = self.linear(x.squeeze(-1))
        out = out.view(B, S, self.num_quantiles)
        out = torch.sort(out, dim=-1)[0]
        return out


class HIEUModel(nn.Module):
    """
    HIEU - Hypernetwork-Integrated Expert Unit
    
    A regime-aware hypernetwork that dynamically generates context-conditioned
    low-rank weight adaptations for multi-asset cryptocurrency forecasting.
    
    Key features:
    - Regime-aware: Detects market states (bull, bear, volatile, sideways)
    - Cross-asset: Learns dynamic inter-asset dependencies
    - Multi-scale: Decomposes temporal patterns across frequency bands
    - Interpretable: Glass-box explainability via auxiliary outputs
    - Probabilistic: Quantile predictions for uncertainty estimation
    """
    
    def __init__(self, config: HIEUConfig = None):
        super().__init__()
        if config is None:
            config = HIEUConfig()
        
        self.cfg = config
        N = config.num_nodes
        
        # Core linear with RevIN
        self.core = RGRLCore(config.seq_len, config.pred_len, channels=N)
        self.nlinear = NLinearCore(config.seq_len, config.pred_len, channels=N)
        self.decomp_core = DLinearCore(
            config.seq_len,
            config.pred_len,
            channels=N,
            kernel_size=getattr(config, 'ma_kernel', 25),
        )
        
        # Regime encoder
        self.regime = RegimeEncoder(
            in_channels=N, 
            num_regimes=config.num_regimes,
            latent_dim=config.regime_dim,
            temperature=config.regime_temp
        )
        
        # Dynamic graph
        self.graph = DynamicGraph(num_nodes=N, hidden_dim=config.graph_hidden)
        
        # Frequency bank
        self.freq = FrequencyBank(
            channels=N, 
            num_bands=config.num_bands,
            kernel=config.band_kernel
        )
        self.temporal_refiner = TemporalRefiner(
            channels=N,
            hidden_dim=getattr(config, 'temporal_hidden', 128),
            dropout=getattr(config, 'temporal_dropout', 0.15),
        )
        self.asset_embed = nn.Embedding(N, getattr(config, 'asset_dim', 24))
        self.graph_to_pred = IndividualLinear(config.seq_len, config.pred_len, N)
        
        # HyperLinear
        ctx_dim = (
            config.regime_dim
            + config.graph_hidden
            + config.num_bands
            + getattr(config, 'asset_dim', 24)
        )
        self.hyper = HyperLinear(
            config.seq_len, config.pred_len, 
            rank=config.linear_rank, ctx_dim=ctx_dim
        )

        self.expert_names = (
            'revin_linear',
            'decomposition',
            'nlinear',
            'hyper',
            'graph',
            'zero',
            'persistence',
        )
        gate_dim = (
            config.regime_dim
            + config.graph_hidden
            + config.num_bands
            + getattr(config, 'asset_dim', 24)
            + 5
        )
        self.expert_gate = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Linear(gate_dim, getattr(config, 'gate_hidden', 128)),
            nn.GELU(),
            nn.Dropout(getattr(config, 'temporal_dropout', 0.15)),
            nn.Linear(getattr(config, 'gate_hidden', 128), len(self.expert_names)),
        )
        nn.init.zeros_(self.expert_gate[-1].weight)
        with torch.no_grad():
            self.expert_gate[-1].bias.copy_(torch.tensor(
                [1.0, 0.7, 0.5, 0.3, 0.3, 0.7, 0.2],
                dtype=self.expert_gate[-1].bias.dtype,
            ))
        
        # Quantile head
        self.qhead = QuantileHead(config.pred_len, quantiles=config.quantiles)
        
        # Residual scaling
        self.residual_scale = nn.Parameter(torch.tensor(0.5))
        self.graph_scale = nn.Parameter(torch.tensor(0.1))

    def _asset_features(self, x):
        lookback = min(24, x.shape[1])
        last = x[:, -1, :]
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        recent = x[:, -lookback:, :].mean(dim=1)
        early = x[:, :lookback, :].mean(dim=1)
        recent_vol = x[:, -lookback:, :].std(dim=1, unbiased=False)
        return torch.stack([last, mean, std, recent - early, recent_vol], dim=-1)
    
    def forward(self, x, return_aux=False):
        """
        Forward pass
        
        Args:
            x: [B, L, N] input tensor (batch, seq_len, num_assets)
            return_aux: whether to return auxiliary outputs for explainability
        
        Returns:
            y_point: [B, S, N] point predictions
            q: [B, S, Q, N] quantile predictions (if return_aux)
            aux: dict of interpretable outputs (if return_aux)
        """
        B, L, N = x.shape
        
        # === Context Stream ===
        z, logits, gate, recon = self.regime(x)
        features = x[:, -1, :]
        g_ctx, A_learned = self.graph(features)
        x_fused, bands, freq_weights = self.freq(x)
        x_fused = self.temporal_refiner(x_fused)

        self_mask = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
        A_scores = A_learned.abs().masked_fill(self_mask, -1e4)
        A_norm = torch.softmax(A_scores, dim=-1)
        x_graph = torch.einsum('bij,blj->bli', A_norm, x_fused)
        
        # === Prediction Stream ===
        y_core = self.core(x)
        y_decomp = self.decomp_core(x)
        y_nlinear = self.nlinear(x)
        
        ctx = torch.cat([
            z, g_ctx,
            freq_weights.unsqueeze(0).expand(B, -1)
        ], dim=-1)
        
        y_hyper_list = []
        W_list = []
        asset_ids = torch.arange(N, device=x.device)
        asset_ctx = self.asset_embed(asset_ids)
        for i in range(N):
            ctx_i = torch.cat([
                ctx,
                asset_ctx[i].unsqueeze(0).expand(B, -1)
            ], dim=-1)
            y_i, W_i = self.hyper(x_fused[:, :, i:i+1], ctx_i)
            y_hyper_list.append(y_i)
            W_list.append(W_i)
        
        y_hyper = torch.cat(y_hyper_list, dim=2)
        y_graph = self.graph_to_pred(x_graph)
        y_zero = torch.zeros_like(y_core)
        y_persist = x[:, -1:, :].expand(-1, self.cfg.pred_len, -1)

        global_ctx = ctx.unsqueeze(1).expand(-1, N, -1)
        gate_input = torch.cat([
            global_ctx,
            asset_ctx.unsqueeze(0).expand(B, -1, -1),
            self._asset_features(x),
        ], dim=-1)
        expert_gate = torch.softmax(self.expert_gate(gate_input), dim=-1)

        experts = torch.stack([
            y_core,
            y_decomp,
            y_nlinear,
            y_core + self.residual_scale * y_hyper,
            y_core + self.graph_scale * y_graph,
            y_zero,
            y_persist,
        ], dim=-1)
        y_point = (experts * expert_gate.unsqueeze(1)).sum(dim=-1)
        
        if return_aux:
            q_list = [self.qhead(y_point[:, :, i:i+1]) for i in range(N)]
            q = torch.stack(q_list, dim=3)
            
            aux = {
                'regime_logits': logits,
                'regime_gate': gate,
                'regime_z': z,
                'graph_A': A_learned,
                'graph_ctx': g_ctx,
                'freq_weights': freq_weights,
                'freq_bands': bands,
                'hyper_W': torch.stack(W_list, dim=3),
                'graph_forecast': y_graph,
                'expert_gate': expert_gate,
                'expert_names': self.expert_names,
            }
            
            return y_point, q, aux
        
        return y_point
    
    def get_interpretable_outputs(self, x):
        """Get all interpretable outputs for glass-box explainability"""
        _, q, aux = self.forward(x, return_aux=True)
        
        return {
            'regime_probabilities': F.softmax(aux['regime_logits'], dim=-1),
            'graph_heatmap': aux['graph_A'],
            'frequency_importance': aux['freq_weights'],
            'quantile_predictions': q,
            'expert_gate': aux['expert_gate'],
            'expert_names': aux['expert_names'],
            'context_vector': torch.cat([aux['regime_z'], aux['graph_ctx']], dim=-1)
        }
