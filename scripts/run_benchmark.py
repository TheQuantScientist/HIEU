"""
HIEU Benchmark Script
=====================
Runs complete benchmark: HIEU + baseline models on 18 cryptocurrency assets.

Usage:
    python scripts/run_benchmark.py

Output:
    - analysis/benchmark_results.csv: Per-asset metrics for all models
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from models.HIEU.multi_asset_loader import create_multiasset_loaders
from models.HIEU.model import HIEUModel, HIEUConfig
from baseline_models.linear_models import Linear, DLinear, NLinear
from baseline_models.rlinear_model import RLinearModel
from baseline_models.patchtst_model import PatchTST
from baseline_models.itransformer_model import iTransformer


# ============================================================================
# Model Configs
# ============================================================================
class LinearConfig:
    seq_len = 96
    pred_len = 96
    enc_in = 19
    individual = True

class DLinearConfig:
    seq_len = 96
    pred_len = 96
    enc_in = 19
    individual = True
    kernel_size = 25

class NLinearConfig:
    seq_len = 96
    pred_len = 96
    enc_in = 19
    individual = True

class RLinearConfig:
    seq_len = 96
    pred_len = 96
    enc_in = 19
    individual = True

class PatchTSTConfig:
    seq_len = 96
    pred_len = 96
    enc_in = 19
    c_out = 19
    d_model = 64
    n_heads = 4
    e_layers = 2
    d_ff = 128
    dropout = 0.1
    fc_dropout = 0.1
    head_dropout = 0.1
    patch_len = 16
    stride = 8
    padding_patch = 'end'
    individual = False
    revin = True
    affine = True
    subtract_last = False
    decomposition = False
    kernel_size = 25

class iTransformerConfig:
    seq_len = 96
    pred_len = 96
    enc_in = 19
    c_out = 19
    d_model = 64
    n_heads = 4
    e_layers = 2
    d_ff = 128
    dropout = 0.1
    factor = 1
    activation = 'gelu'
    output_attention = False
    use_norm = True
    embed = 'timeF'
    freq = 'h'
    class_strategy = 'projection'


# ============================================================================
# Model Wrappers
# ============================================================================
class SimpleMoLE(nn.Module):
    """Simple Mixture of Linear Experts"""
    def __init__(self, seq_len, pred_len, enc_in):
        super().__init__()
        self.num_experts = 4
        self.experts = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(self.num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(seq_len, 32), nn.ReLU(),
            nn.Linear(32, self.num_experts), nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        B, L, N = x.shape
        x_t = x.permute(0, 2, 1)
        weights = self.gate(x_t.mean(dim=1))
        outputs = torch.stack([exp(x_t) for exp in self.experts], dim=-1)
        out = (outputs * weights.unsqueeze(1).unsqueeze(1)).sum(dim=-1)
        return out.permute(0, 2, 1)


class iTransformerWrapper(nn.Module):
    """Wrapper for iTransformer"""
    def __init__(self, config):
        super().__init__()
        self.model = iTransformer(config)
        self.pred_len = config.pred_len
    
    def forward(self, x):
        B, L, N = x.shape
        x_dec = torch.zeros(B, self.pred_len, N, device=x.device)
        return self.model(x, None, x_dec, None)


# ============================================================================
# Model Factory
# ============================================================================
def create_model(model_name, num_assets, seq_len, pred_len):
    """Create model by name"""
    if model_name == 'HIEU':
        config = HIEUConfig()
        config.num_nodes = num_assets
        config.seq_len = seq_len
        config.pred_len = pred_len
        config.graph_hidden = 128
        return HIEUModel(config)
    
    elif model_name == 'SimpleMoLE':
        return SimpleMoLE(seq_len, pred_len, num_assets)
    
    elif model_name == 'Linear':
        config = LinearConfig()
        config.seq_len, config.pred_len, config.enc_in = seq_len, pred_len, num_assets
        return Linear(config)
    
    elif model_name == 'DLinear':
        config = DLinearConfig()
        config.seq_len, config.pred_len, config.enc_in = seq_len, pred_len, num_assets
        return DLinear(config)
    
    elif model_name == 'NLinear':
        config = NLinearConfig()
        config.seq_len, config.pred_len, config.enc_in = seq_len, pred_len, num_assets
        return NLinear(config)
    
    elif model_name == 'RLinear':
        config = RLinearConfig()
        config.seq_len, config.pred_len, config.enc_in = seq_len, pred_len, num_assets
        return RLinearModel(config)
    
    elif model_name == 'PatchTST':
        config = PatchTSTConfig()
        config.seq_len, config.pred_len = seq_len, pred_len
        config.enc_in = config.c_out = num_assets
        return PatchTST(config)
    
    elif model_name == 'iTransformer':
        config = iTransformerConfig()
        config.seq_len, config.pred_len = seq_len, pred_len
        config.enc_in = config.c_out = num_assets
        return iTransformerWrapper(config)
    
    raise ValueError(f"Unknown model: {model_name}")


# ============================================================================
# Training & Evaluation
# ============================================================================
def train_model(model, train_loader, valid_loader, device, epochs=40, lr=1e-3, patience=7):
    """Train with early stopping"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()
    
    best_val, best_state, bad_epochs = float('inf'), None, 0
    
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            yp = model(xb)
            if yp.shape != yb.shape: yp = yp.view_as(yb)
            loss = criterion(yp, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device)
                yp = model(xb)
                if yp.shape != yb.shape: yp = yp.view_as(yb)
                val_loss += criterion(yp, yb).item()
        val_loss /= len(valid_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience: break
    
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def evaluate_model(model, test_loader, device, symbols):
    """Evaluate and return per-asset metrics"""
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            yp = model(xb.to(device))
            if yp.shape != yb.shape: yp = yp.view_as(yb)
            all_preds.append(yp.cpu().numpy())
            all_targets.append(yb.numpy())
    
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    results = []
    for i, sym in enumerate(symbols):
        p, t = preds[:, :, i].flatten(), targets[:, :, i].flatten()
        mae = np.abs(p - t).mean()
        rmse = np.sqrt(((p - t) ** 2).mean())
        results.append({'asset': sym, 'MAE': mae, 'RMSE': rmse})
    
    avg_mae = np.mean([r['MAE'] for r in results])
    avg_rmse = np.mean([r['RMSE'] for r in results])
    results.append({'asset': 'Average', 'MAE': avg_mae, 'RMSE': avg_rmse})
    return results


# ============================================================================
# Main
# ============================================================================
def main():
    print("="*70)
    print("HIEU BENCHMARK - Multi-Asset Cryptocurrency Forecasting")
    print(f"Time: {datetime.now()}")
    print("="*70)
    
    # Config
    seq_len, pred_len, batch_size = 96, 96, 32
    max_samples = 5000
    seeds = [42, 123, 456]
    
    all_symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT',
        'ATOMUSDT', 'XLMUSDT', 'ETCUSDT', 'VETUSDT', 'TRXUSDT',
        'FILUSDT', 'UNIUSDT', 'DOGEUSDT', 'XMRUSDT'
    ]
    
    models_to_test = ['HIEU', 'SimpleMoLE', 'Linear', 'DLinear', 'NLinear', 
                      'RLinear', 'PatchTST', 'iTransformer']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Assets: {len(all_symbols)} cryptocurrencies")
    print(f"Models: {models_to_test}")
    
    # Load data
    print("\nLoading data...")
    train_loader, valid_loader, test_loader, _ = create_multiasset_loaders(
        data_dir='data', symbols=all_symbols, seq_len=seq_len, pred_len=pred_len,
        batch_size=batch_size, max_samples=max_samples, use_returns=True,
        log_returns=True, standardize=True
    )
    num_assets = next(iter(train_loader))[0].shape[2]
    print(f"Loaded {num_assets} assets")
    
    all_results = []
    
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")
        
        seed_results = {sym: {'MAE': [], 'RMSE': []} for sym in all_symbols + ['Average']}
        
        for seed in seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            try:
                model = create_model(model_name, num_assets, seq_len, pred_len).to(device)
                lr = 8e-4 if model_name == 'HIEU' else 1e-3
                epochs = 50 if model_name == 'HIEU' else 40
                patience = 10 if model_name == 'HIEU' else 7
                
                model = train_model(model, train_loader, valid_loader, device, epochs, lr, patience)
                results = evaluate_model(model, test_loader, device, all_symbols)
                
                for r in results:
                    seed_results[r['asset']]['MAE'].append(r['MAE'])
                    seed_results[r['asset']]['RMSE'].append(r['RMSE'])
                
                avg = [r for r in results if r['asset'] == 'Average'][0]
                print(f"MAE={avg['MAE']:.4f}, RMSE={avg['RMSE']:.4f}")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        for asset in all_symbols + ['Average']:
            if seed_results[asset]['MAE']:
                all_results.append({
                    'model': model_name, 'asset': asset,
                    'MAE_mean': np.mean(seed_results[asset]['MAE']),
                    'MAE_std': np.std(seed_results[asset]['MAE']),
                    'RMSE_mean': np.mean(seed_results[asset]['RMSE']),
                    'RMSE_std': np.std(seed_results[asset]['RMSE']),
                })
    
    # Save results
    df = pd.DataFrame(all_results)
    os.makedirs('analysis', exist_ok=True)
    df.to_csv('analysis/benchmark_results.csv', index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY (sorted by RMSE)")
    print("="*70)
    
    avg_df = df[df['asset'] == 'Average'].sort_values('RMSE_mean')
    print(f"\n{'Rank':<6} {'Model':<15} {'MAE':<20} {'RMSE':<20}")
    print("-"*61)
    for rank, (_, row) in enumerate(avg_df.iterrows(), 1):
        mae_str = f"{row['MAE_mean']:.4f}±{row['MAE_std']:.4f}"
        rmse_str = f"{row['RMSE_mean']:.4f}±{row['RMSE_std']:.4f}"
        print(f"{rank:<6} {row['model']:<15} {mae_str:<20} {rmse_str:<20}")
    
    print(f"\nResults saved to: analysis/benchmark_results.csv")


if __name__ == "__main__":
    main()
