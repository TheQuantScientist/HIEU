import os
import pandas as pd
import numpy as np
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
import torch

class MultiAssetDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = data  # [T, N]
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]          # [L, N]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]  # [H, N]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def load_align_close_series(data_dir: str, symbols: List[str]) -> Tuple[pd.DataFrame, dict]:
    frames = []
    meta = {}
    for sym in symbols:
        fp = os.path.join(data_dir, f'{sym}.csv')
        df = pd.read_csv(fp)

        if len(df.columns) == 1 and str(df.columns[0]).startswith('version https://git-lfs'):
            raise ValueError(
                f'{fp} is a Git LFS pointer, not a real CSV. '
                'Fetch the LFS data files before training HIEU.'
            )

        time_col = next(
            (c for c in ['Date', 'date', 'Datetime', 'datetime', 'timestamp', 'time', 'Open time']
             if c in df.columns),
            None,
        )
        close_col = next(
            (c for c in ['Close', 'close', 'Adj Close', 'adj_close', 'last', 'Last']
             if c in df.columns),
            None,
        )

        if time_col is None or close_col is None:
            raise ValueError(
                f'{fp} must contain a timestamp column and a close-price column. '
                f'Found columns: {list(df.columns)}'
            )

        timestamp = df[time_col]
        if pd.api.types.is_numeric_dtype(timestamp):
            unit = 'ms' if timestamp.dropna().abs().median() > 10**11 else 's'
            timestamp = pd.to_datetime(timestamp, unit=unit)
        else:
            timestamp = pd.to_datetime(timestamp)

        df = df.assign(Date=timestamp, Close=pd.to_numeric(df[close_col], errors='coerce'))
        df = df.set_index('Date').sort_index()
        df = df.resample('15min').agg({'Close':'last'}).dropna()
        df = df.rename(columns={'Close': sym})
        meta[sym] = {'start': df.index.min(), 'end': df.index.max(), 'len': len(df)}
        frames.append(df[[sym]])
    # inner join on timestamps to align
    aligned = frames[0]
    for f in frames[1:]:
        aligned = aligned.join(f, how='inner')
    return aligned, meta


def split_by_years(df: pd.DataFrame):
    train = df[df.index.year <= 2023]
    valid = df[df.index.year == 2024]
    test = df[df.index.year == 2025]
    return train, valid, test


def _compute_returns(df: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    if log:
        returns = np.log(df / df.shift(1))
    else:
        returns = df.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return returns


def _standardize(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    mu = train.mean(axis=0)
    sigma = train.std(axis=0).replace(0, 1.0)
    train_s = (train - mu) / sigma
    valid_s = (valid - mu) / sigma
    test_s = (test - mu) / sigma
    return train_s, valid_s, test_s, {'mean': mu, 'std': sigma}


def create_multiasset_loaders(data_dir: str, symbols: List[str], seq_len: int, pred_len: int, batch_size: int,
                               max_samples: int = None, use_returns: bool = True, log_returns: bool = True,
                               standardize: bool = True):
    aligned, meta = load_align_close_series(data_dir, symbols)
    if use_returns:
        aligned = _compute_returns(aligned, log=log_returns)
    train_df, valid_df, test_df = split_by_years(aligned)
    if max_samples:
        if len(train_df) > max_samples:
            train_df = train_df.tail(max_samples)
        if len(valid_df) > max_samples // 4:
            valid_df = valid_df.tail(max_samples // 4)
        if len(test_df) > max_samples // 4:
            test_df = test_df.tail(max_samples // 4)
    if standardize:
        train_df, valid_df, test_df, stats = _standardize(train_df, valid_df, test_df)
    else:
        stats = None
    train_np = train_df.values.astype(np.float32)
    valid_np = valid_df.values.astype(np.float32)
    test_np = test_df.values.astype(np.float32)
    train_ds = MultiAssetDataset(train_np, seq_len, pred_len)
    valid_ds = MultiAssetDataset(valid_np, seq_len, pred_len)
    test_ds = MultiAssetDataset(test_np, seq_len, pred_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader, {'train': train_df, 'valid': valid_df, 'test': test_df, 'stats': stats}
