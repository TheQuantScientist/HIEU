"""
Linear models for time series forecasting
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Ridge
from typing import Optional

class Linear(nn.Module):
    """Simple Linear model"""
    def __init__(self, config):
        super(Linear, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.channels = config.enc_in
        self.individual = config.individual
        
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
    
    def forward(self, x):
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x

class DLinear(nn.Module):
    """Decomposition-Linear model"""
    def __init__(self, config):
        super(DLinear, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        kernel_size = config.kernel_size
        self.decomposition = series_decomp(kernel_size)
        self.individual = config.individual
        self.channels = config.enc_in
        
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
    
    def forward(self, x):
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len], 
                                        dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len], 
                                     dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

class NLinear(nn.Module):
    """Normalization-Linear model"""
    def __init__(self, config):
        super(NLinear, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.channels = config.enc_in
        self.individual = config.individual
        
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
    
    def forward(self, x):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        x = x + seq_last
        return x

class RLinearOLS:
    """RLinear model with OLS (Ridge regression)"""
    def __init__(self, dataset_train, context_length, horizon, instance_norm=True, 
                 individual=False, alpha=0.000001, seed=42, verbose=False, max_train_N=None):
        self.dataset = dataset_train
        self.context_length = context_length
        self.horizon = horizon
        self.instance_norm = instance_norm
        self.individual = individual
        self.alpha = alpha
        self.seed = seed
        self.verbose = verbose
        self.max_train_N = max_train_N
        
        # Initialize RevIN
        from ..models.revin import RevIN
        self.revin = RevIN(num_features=dataset_train.shape[1], affine=True)
        
        self.fit_ols_solutions()
    
    def fit_ols_solutions(self):
        """Fit OLS solutions"""
        # Normalize data with RevIN
        dataset_tensor = torch.FloatTensor(self.dataset).unsqueeze(0)
        dataset_normalized = self.revin(dataset_tensor, mode='norm')
        dataset_normalized = dataset_normalized.squeeze(0).detach().numpy()
        
        # Create sliding windows
        instances = np.lib.stride_tricks.sliding_window_view(
            dataset_normalized, (self.context_length + self.horizon), axis=0
        )
        
        if self.max_train_N is not None:
            instances = instances[-self.max_train_N:]
        
        X = instances[:, :self.context_length, :]
        y = instances[:, self.context_length:, :]
        
        if self.individual:
            self.models = []
            for i in range(X.shape[2]):
                model = Ridge(alpha=self.alpha, random_state=self.seed)
                model.fit(X[:, :, i], y[:, :, i])
                self.models.append(model)
        else:
            X_flat = X.reshape(X.shape[0], -1)
            y_flat = y.reshape(y.shape[0], -1)
            self.model = Ridge(alpha=self.alpha, random_state=self.seed)
            self.model.fit(X_flat, y_flat)
    
    def predict(self, X):
        """Make predictions"""
        # Normalize input with RevIN
        X_tensor = torch.FloatTensor(X).unsqueeze(0)
        X_normalized = self.revin(X_tensor, mode='norm')
        X_normalized = X_normalized.squeeze(0).detach().numpy()
        
        if self.individual:
            preds = np.zeros((X_normalized.shape[0], self.horizon, X_normalized.shape[2]))
            for i in range(X_normalized.shape[2]):
                preds[:, :, i] = self.models[i].predict(X_normalized[:, :, i])
        else:
            X_flat = X_normalized.reshape(X_normalized.shape[0], -1)
            preds_flat = self.model.predict(X_flat)
            preds = preds_flat.reshape(X_normalized.shape[0], self.horizon, X_normalized.shape[2])
        
        # Denormalize predictions
        X_original = X
        context_means = np.mean(X_original, axis=1, keepdims=True)
        context_stds = np.sqrt(np.var(X_original, axis=1, keepdims=True) + 1e-5)
        
        with torch.no_grad():
            dummy_tensor = torch.FloatTensor(X_original).unsqueeze(0)
            self.revin(dummy_tensor, mode='norm')
            if self.revin.affine:
                affine_weight = self.revin.affine_weight.detach().numpy()
                affine_bias = self.revin.affine_bias.detach().numpy()
            else:
                affine_weight = np.ones(X_original.shape[2])
                affine_bias = np.zeros(X_original.shape[2])
        
        preds_denormalized = preds - affine_bias
        preds_denormalized = preds_denormalized / (affine_weight + 1e-5)
        preds_denormalized = preds_denormalized * context_stds
        preds_denormalized = preds_denormalized + context_means
        
        return preds_denormalized

# Helper function for series decomposition
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
