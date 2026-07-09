
class HIEUConfig:

    def __init__(self):
        # IO
        self.model_version = 'HIEU-v3-gated-experts'
        self.seq_len = 96
        self.pred_len = 96
        self.enc_in = 1
        self.batch_size = 32
        self.num_nodes = 19  # number of assets

        # Training (optimized for 19 coins)
        self.learning_rate = 5e-4
        self.weight_decay = 2e-4
        self.epochs = 80

        # Regime encoder
        self.num_regimes = 4
        self.regime_dim = 64
        self.regime_temp = 1.0
        self.ssl_weight = 0.1
        self.use_gc_features = True

        # Graph (increased for 19 assets)
        self.graph_hidden = 128  # Increased from 64 for more assets
        self.graph_prior_weight = 1e-2
        self.laplacian_weight = 1e-3
        self.use_gc_prior = True

        # Frequency bank
        self.num_bands = 5
        self.band_kernel = 15
        self.band_stride = 1
        self.use_freq_reweight = True
        self.temporal_hidden = 128
        self.temporal_dropout = 0.15
        self.ma_kernel = 25

        # HyperLinear
        self.linear_rank = 12
        self.asset_dim = 24
        self.gate_hidden = 128
        self.num_experts = 3

        # Probabilistic
        self.quantiles = [0.1, 0.5, 0.9]
        self.pinball_weight = 0.05
        self.crps_weight = 0.2
        self.direction_weight = 0.0

        # TTA (optional)
        self.tta_steps = 2
        self.tta_lr = 1e-4
        self.tta_use_tent = True
