"""
FEDformer Benchmark Only
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEDFORMER_PATH = os.path.join(ROOT_DIR, 'baseline_models', 'FEDformer')
sys.path.insert(0, FEDFORMER_PATH)
os.chdir(FEDFORMER_PATH)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from models.FEDformer import Model as FEDformerModel

sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import importlib.util
spec = importlib.util.spec_from_file_location("multi_asset_loader", 
    os.path.join(ROOT_DIR, 'models', 'HIEU', 'multi_asset_loader.py'))
loader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader_module)
create_multiasset_loaders = loader_module.create_multiasset_loaders


class FEDformerConfig:
    def __init__(self, seq_len=96, pred_len=96, enc_in=19):
        self.seq_len = seq_len
        self.label_len = seq_len // 2
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = enc_in
        self.c_out = enc_in
        self.d_model = 64
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 128
        self.moving_avg = 25
        self.factor = 1
        self.dropout = 0.1
        self.embed = 'timeF'
        self.freq = 'h'
        self.activation = 'gelu'
        self.output_attention = False
        self.version = 'Fourier'
        self.mode_select = 'random'
        self.modes = 32
        self.L = 1
        self.base = 'legendre'
        self.cross_activation = 'tanh'


class FEDformerWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = FEDformerModel(config)
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.label_len = config.label_len
    
    def forward(self, x):
        B, L, N = x.shape
        device = x.device
        x_dec = torch.cat([x[:, -self.label_len:, :], torch.zeros(B, self.pred_len, N, device=device)], dim=1)
        x_mark_enc = torch.zeros(B, L, 4, device=device)
        x_mark_dec = torch.zeros(B, self.label_len + self.pred_len, 4, device=device)
        return self.model(x, x_mark_enc, x_dec, x_mark_dec)


def train_model(model, train_loader, valid_loader, device, epochs=20, lr=1e-3, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()
    best_val, best_state, bad_epochs = float('inf'), None, 0
    
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        model.eval()
        val_loss = sum(criterion(model(xb.to(device)), yb.to(device)).item() for xb, yb in valid_loader) / len(valid_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_val:
            best_val, best_state, bad_epochs = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience: break
    
    if best_state: model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def evaluate_model(model, test_loader, device, symbols):
    model.eval()
    preds = np.concatenate([model(xb.to(device)).cpu().numpy() for xb, _ in test_loader])
    targets = np.concatenate([yb.numpy() for _, yb in test_loader])
    
    results = [{'asset': sym, 'MAE': np.abs(preds[:,:,i].flatten() - targets[:,:,i].flatten()).mean(),
                'RMSE': np.sqrt(((preds[:,:,i].flatten() - targets[:,:,i].flatten())**2).mean())} 
               for i, sym in enumerate(symbols)]
    results.append({'asset': 'Average', 'MAE': np.mean([r['MAE'] for r in results]), 'RMSE': np.mean([r['RMSE'] for r in results])})
    return results


def main():
    print("="*70 + f"\nFEDFORMER BENCHMARK\nTime: {datetime.now()}\n" + "="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 
                   'LTCUSDT', 'BCHUSDT', 'ATOMUSDT', 'XLMUSDT', 'ETCUSDT', 'VETUSDT', 'TRXUSDT', 'FILUSDT', 
                   'UNIUSDT', 'DOGEUSDT', 'XMRUSDT']
    seeds = [42, 123, 456]
    
    train_loader, valid_loader, test_loader, _ = create_multiasset_loaders(
        data_dir='data', symbols=all_symbols, seq_len=96, pred_len=96, batch_size=32, max_samples=5000,
        use_returns=True, log_returns=True, standardize=True)
    
    seed_results = {sym: {'MAE': [], 'RMSE': []} for sym in all_symbols + ['Average']}
    
    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        try:
            model = FEDformerWrapper(FEDformerConfig(96, 96, 19)).to(device)
            model = train_model(model, train_loader, valid_loader, device)
            for r in evaluate_model(model, test_loader, device, all_symbols):
                seed_results[r['asset']]['MAE'].append(r['MAE'])
                seed_results[r['asset']]['RMSE'].append(r['RMSE'])
            avg = seed_results['Average']
            print(f"MAE={avg['MAE'][-1]:.4f}, RMSE={avg['RMSE'][-1]:.4f}")
        except Exception as e:
            print(f"Error: {e}")
    
    all_results = [{'model': 'FEDformer', 'asset': asset, 'MAE_mean': np.mean(seed_results[asset]['MAE']),
                    'MAE_std': np.std(seed_results[asset]['MAE']), 'RMSE_mean': np.mean(seed_results[asset]['RMSE']),
                    'RMSE_std': np.std(seed_results[asset]['RMSE'])} for asset in all_symbols + ['Average'] if seed_results[asset]['MAE']]
    
    if all_results:
        df = pd.concat([pd.read_csv('analysis/complete_benchmark_results.csv'), pd.DataFrame(all_results)], ignore_index=True)
        df.to_csv('analysis/complete_benchmark_results.csv', index=False)
        print(f"\nFEDformer appended. Average: MAE={all_results[-1]['MAE_mean']:.4f}±{all_results[-1]['MAE_std']:.4f}")


if __name__ == "__main__":
    main()
