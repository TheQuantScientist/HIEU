"""
HIEU Benchmark Script
=====================
Runs benchmark for HIEU only on 19 cryptocurrency assets.

Usage:
    python scripts/run_hieu_only.py

Output:
    - analysis/hieu_benchmark_results.csv: Per-asset metrics for HIEU
"""

import os
import sys
import re
from datetime import datetime
import warnings

# ============================================================================
# Environment setup
# ============================================================================
# This must be set before importing torch when possible.
# It helps avoid some cuDNN v8 backend issues.
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

# ============================================================================
# Important compatibility fixes
# ============================================================================

# Fix 1:
# Your machine has a CUDA/cuDNN mismatch. HIEU fails inside Conv1d when cuDNN is used.
# This keeps CUDA available but disables cuDNN kernels.
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# Fix 2:
# Newer Pandas versions reject old aliases like "15T".
# The existing loader uses df.resample("15T"), so this patch converts "15T" -> "15min".
def _normalize_pandas_freq(rule):
    if isinstance(rule, str):
        # Convert "T" or "15T" to "min" or "15min"
        if rule.lower().endswith("t"):
            number = rule[:-1]
            if number == "" or number.isdigit():
                return f"{number}min"
    return rule


_original_dataframe_resample = pd.DataFrame.resample
_original_series_resample = pd.Series.resample


def _patched_dataframe_resample(self, rule=None, *args, **kwargs):
    if rule is not None:
        rule = _normalize_pandas_freq(rule)

    if "rule" in kwargs:
        kwargs["rule"] = _normalize_pandas_freq(kwargs["rule"])

    return _original_dataframe_resample(self, rule=rule, *args, **kwargs)


def _patched_series_resample(self, rule=None, *args, **kwargs):
    if rule is not None:
        rule = _normalize_pandas_freq(rule)

    if "rule" in kwargs:
        kwargs["rule"] = _normalize_pandas_freq(kwargs["rule"])

    return _original_series_resample(self, rule=rule, *args, **kwargs)


pd.DataFrame.resample = _patched_dataframe_resample
pd.Series.resample = _patched_series_resample


from models.HIEU.multi_asset_loader import create_multiasset_loaders
from models.HIEU.model import HIEUModel, HIEUConfig


# ============================================================================
# Model Factory
# ============================================================================
def create_hieu_model(num_assets, seq_len, pred_len):
    """Create HIEU model."""
    config = HIEUConfig()

    config.num_nodes = num_assets
    config.seq_len = seq_len
    config.pred_len = pred_len
    config.graph_hidden = 128

    return HIEUModel(config)


# ============================================================================
# Helpers
# ============================================================================
def set_seed(seed):
    """Set random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def is_cuda_backend_error(error):
    """Detect CUDA/cuDNN backend errors that should fall back to CPU."""
    msg = str(error).lower()

    keywords = [
        "cudnn",
        "cuda",
        "sublibrary_version_mismatch",
        "cudnn_status_sublibrary_version_mismatch",
        "cudnn_backend_tensor_descriptor",
    ]

    return any(k in msg for k in keywords)


def normalize_prediction(yp, yb):
    """Make model output match target shape."""
    if isinstance(yp, tuple) or isinstance(yp, list):
        yp = yp[0]

    if yp.shape != yb.shape:
        yp = yp.reshape_as(yb)

    return yp


def pinball_loss(pred_q, y_true, quantiles):
    """Quantile regression loss for [B, S, Q, N] predictions."""
    target = y_true.unsqueeze(2)
    losses = []

    for i, tau in enumerate(quantiles):
        err = target - pred_q[:, :, i:i + 1, :]
        losses.append(torch.maximum(tau * err, (tau - 1.0) * err).mean())

    return torch.stack(losses).mean()


def hieu_training_loss(model, xb, yb):
    """Objective aligned with reported MAE/RMSE metrics."""
    yp, pred_q, _ = model(xb, return_aux=True)
    yp = normalize_prediction(yp, yb)

    huber_beta = getattr(model.cfg, "huber_beta", 0.5)
    mse = F.mse_loss(yp, yb)
    mae = F.l1_loss(yp, yb)
    huber = F.smooth_l1_loss(yp, yb, beta=huber_beta)

    quantile = pinball_loss(
        pred_q,
        yb,
        getattr(model.cfg, "quantiles", [0.1, 0.5, 0.9]),
    )

    direction_weight = getattr(model.cfg, "direction_weight", 0.0)
    if direction_weight > 0:
        pred_cum = yp.sum(dim=1)
        true_cum = yb.sum(dim=1)
        direction = F.softplus(-pred_cum * true_cum).mean()
    else:
        direction = yp.new_tensor(0.0)

    loss = (
        0.65 * mse
        + 0.25 * huber
        + 0.10 * mae
        + getattr(model.cfg, "pinball_weight", 0.05) * quantile
        + direction_weight * direction
    )

    return loss, yp


def get_loaded_symbols(meta, all_symbols, num_assets):
    """Try to recover loaded symbol names from loader metadata."""
    loaded_symbols = None

    if isinstance(meta, dict):
        for key in ["symbols", "loaded_symbols", "asset_symbols", "tickers"]:
            if key in meta and isinstance(meta[key], list):
                loaded_symbols = meta[key]
                break

    elif isinstance(meta, list) and all(isinstance(x, str) for x in meta):
        loaded_symbols = meta

    elif isinstance(meta, tuple) and all(isinstance(x, str) for x in meta):
        loaded_symbols = list(meta)

    if loaded_symbols is None:
        loaded_symbols = all_symbols[:num_assets]

    if len(loaded_symbols) != num_assets:
        print(
            f"Warning: metadata symbol count={len(loaded_symbols)} "
            f"but tensor asset count={num_assets}. Using first {num_assets} symbols."
        )
        loaded_symbols = all_symbols[:num_assets]

    return loaded_symbols


# ============================================================================
# Training & Evaluation
# ============================================================================
def train_model(
    model,
    train_loader,
    valid_loader,
    device,
    epochs=None,
    lr=None,
    patience=15,
):
    """Train HIEU with early stopping."""
    epochs = epochs or getattr(model.cfg, "epochs", 80)
    lr = lr or getattr(model.cfg, "learning_rate", 5e-4)
    weight_decay = getattr(model.cfg, "weight_decay", 2e-4)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        factor=0.5,
    )

    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    use_non_blocking = device.type == "cuda"

    for ep in range(epochs):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=use_non_blocking).float()
            yb = yb.to(device, non_blocking=use_non_blocking).float()

            optimizer.zero_grad(set_to_none=True)

            loss, yp = hieu_training_loss(model, xb, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_mse = 0.0
        val_mae = 0.0

        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device, non_blocking=use_non_blocking).float()
                yb = yb.to(device, non_blocking=use_non_blocking).float()

                yp = model(xb)
                yp = normalize_prediction(yp, yb)

                val_mse += criterion(yp, yb).item()
                val_mae += F.l1_loss(yp, yb).item()

        val_mse /= max(1, len(valid_loader))
        val_mae /= max(1, len(valid_loader))
        val_score = 0.7 * val_mse + 0.3 * val_mae
        scheduler.step(val_score)

        print(
            f"    Epoch {ep + 1:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Val MAE: {val_mae:.6f}"
        )

        if val_score < best_val:
            best_val = val_score
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

            if bad_epochs >= patience:
                print(f"    Early stopping at epoch {ep + 1}")
                break

    if best_state is not None:
        model.load_state_dict({
            k: v.to(device)
            for k, v in best_state.items()
        })

    return model


def evaluate_model(model, test_loader, device, symbols):
    """Evaluate HIEU and return per-asset metrics."""
    model.eval()

    all_preds = []
    all_targets = []

    use_non_blocking = device.type == "cuda"

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=use_non_blocking).float()
            yb = yb.float()

            yp = model(xb)
            yp = normalize_prediction(yp, yb.to(device, non_blocking=use_non_blocking))

            all_preds.append(yp.detach().cpu().numpy())
            all_targets.append(yb.detach().cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    results = []

    num_assets = min(len(symbols), preds.shape[2], targets.shape[2])

    for i in range(num_assets):
        sym = symbols[i]

        p = preds[:, :, i].reshape(-1)
        t = targets[:, :, i].reshape(-1)

        mae = np.abs(p - t).mean()
        rmse = np.sqrt(((p - t) ** 2).mean())

        results.append({
            "asset": sym,
            "MAE": float(mae),
            "RMSE": float(rmse),
        })

    avg_mae = np.mean([r["MAE"] for r in results])
    avg_rmse = np.mean([r["RMSE"] for r in results])

    results.append({
        "asset": "Average",
        "MAE": float(avg_mae),
        "RMSE": float(avg_rmse),
    })

    return results


def run_one_seed(
    seed,
    device,
    train_loader,
    valid_loader,
    test_loader,
    symbols,
    num_assets,
    seq_len,
    pred_len,
):
    """Train and evaluate one seed."""
    print(f"\nSeed {seed} on {device}")

    set_seed(seed)

    model = create_hieu_model(
        num_assets=num_assets,
        seq_len=seq_len,
        pred_len=pred_len,
    ).to(device)

    print(f"    Model version: {getattr(model.cfg, 'model_version', 'HIEU')}")
    if hasattr(model, "expert_names"):
        print(f"    Experts: {', '.join(model.expert_names)}")

    model = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=None,
        lr=None,
        patience=15,
    )

    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        symbols=symbols,
    )

    avg = next(r for r in results if r["asset"] == "Average")

    print(
        f"Seed {seed} result: "
        f"MAE={avg['MAE']:.4f}, RMSE={avg['RMSE']:.4f}"
    )

    return results


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("HIEU ONLY BENCHMARK - Multi-Asset Cryptocurrency Forecasting")
    print(f"Time: {datetime.now()}")
    print("=" * 70)

    # Config
    seq_len = 96
    pred_len = 96
    batch_size = 32
    max_samples = None
    seeds = [42, 123, 456]

    all_symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT",
        "ATOMUSDT", "XLMUSDT", "ETCUSDT", "VETUSDT", "TRXUSDT",
        "FILUSDT", "UNIUSDT", "DOGEUSDT", "XMRUSDT",
    ]

    data_dir = os.path.join(PROJECT_ROOT, "data")
    output_dir = os.path.join(PROJECT_ROOT, "analysis")
    output_path = os.path.join(output_dir, "hieu_benchmark_results.csv")

    preferred_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Preferred device: {preferred_device}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"Assets requested: {len(all_symbols)} cryptocurrencies")
    print("Model: HIEU")

    # Load data
    print("\nLoading data...")

    train_loader, valid_loader, test_loader, meta = create_multiasset_loaders(
        data_dir=data_dir,
        symbols=all_symbols,
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=batch_size,
        max_samples=max_samples,
        use_returns=True,
        log_returns=True,
        standardize=True,
    )

    first_batch = next(iter(train_loader))
    num_assets = first_batch[0].shape[2]

    symbols = get_loaded_symbols(meta, all_symbols, num_assets)

    print(f"Loaded {num_assets} assets")
    print(f"Symbols used: {symbols}")

    seed_results = {
        sym: {
            "MAE": [],
            "RMSE": [],
        }
        for sym in symbols + ["Average"]
    }

    print("\n" + "=" * 60)
    print("Testing: HIEU")
    print("=" * 60)

    force_cpu = False

    for seed in seeds:
        device = torch.device("cpu") if force_cpu else preferred_device

        try:
            results = run_one_seed(
                seed=seed,
                device=device,
                train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                symbols=symbols,
                num_assets=num_assets,
                seq_len=seq_len,
                pred_len=pred_len,
            )

        except RuntimeError as e:
            if device.type == "cuda" and is_cuda_backend_error(e):
                print("\nCUDA/cuDNN backend error detected.")
                print("Retrying this seed on CPU so the benchmark can complete.")

                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                force_cpu = True
                cpu_device = torch.device("cpu")

                results = run_one_seed(
                    seed=seed,
                    device=cpu_device,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    test_loader=test_loader,
                    symbols=symbols,
                    num_assets=num_assets,
                    seq_len=seq_len,
                    pred_len=pred_len,
                )
            else:
                print(f"Error while running seed {seed}: {e}")
                import traceback
                traceback.print_exc()
                continue

        except Exception as e:
            print(f"Error while running seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue

        for r in results:
            asset = r["asset"]

            if asset not in seed_results:
                seed_results[asset] = {
                    "MAE": [],
                    "RMSE": [],
                }

            seed_results[asset]["MAE"].append(r["MAE"])
            seed_results[asset]["RMSE"].append(r["RMSE"])

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    all_results = []

    for asset in symbols + ["Average"]:
        mae_values = seed_results[asset]["MAE"]
        rmse_values = seed_results[asset]["RMSE"]

        if mae_values:
            all_results.append({
                "model": "HIEU",
                "asset": asset,
                "MAE_mean": float(np.mean(mae_values)),
                "MAE_std": float(np.std(mae_values)),
                "RMSE_mean": float(np.mean(rmse_values)),
                "RMSE_std": float(np.std(rmse_values)),
                "num_successful_seeds": len(mae_values),
            })

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(
        all_results,
        columns=[
            "model",
            "asset",
            "MAE_mean",
            "MAE_std",
            "RMSE_mean",
            "RMSE_std",
            "num_successful_seeds",
        ],
    )

    df.to_csv(output_path, index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("HIEU RESULTS SUMMARY")
    print("=" * 70)

    if df.empty:
        print("\nNo successful HIEU runs.")
        print("The CSV was still created, but it contains no metric rows.")
        print(f"Results saved to: {output_path}")
        return

    avg_df = df[df["asset"] == "Average"]

    if not avg_df.empty:
        row = avg_df.iloc[0]

        mae_str = f"{row['MAE_mean']:.4f}±{row['MAE_std']:.4f}"
        rmse_str = f"{row['RMSE_mean']:.4f}±{row['RMSE_std']:.4f}"

        print(f"\n{'Model':<15} {'MAE':<20} {'RMSE':<20} {'Seeds':<10}")
        print("-" * 70)
        print(
            f"{'HIEU':<15} "
            f"{mae_str:<20} "
            f"{rmse_str:<20} "
            f"{int(row['num_successful_seeds']):<10}"
        )
    else:
        print("\nNo Average row found in results.")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
