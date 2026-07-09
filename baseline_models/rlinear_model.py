import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Ridge

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        Reversible Instance Normalization
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: 
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class RLinearModel(nn.Module):
    """
    RLinear: RevIN + Linear model
    Combines Reversible Instance Normalization with Linear regression
    """
    def __init__(self, configs):
        super(RLinearModel, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual
        
        # RevIN layer
        self.revin = RevIN(num_features=self.channels, eps=1e-5, affine=True)
        
        # Linear layers
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # Apply RevIN normalization
        x = self.revin(x, mode='norm')
        
        # Apply linear transformation
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Apply RevIN denormalization
        x = self.revin(x, mode='denorm')
        
        return x  # [Batch, Output length, Channel]

class RLinearOLS:
    """
    RLinear with OLS (Ordinary Least Squares) approach
    Combines RevIN with Ridge regression similar to linear-forecasting library
    """
    def __init__(self, 
                 dataset_train, 
                 context_length, 
                 horizon, 
                 instance_norm=True, 
                 individual=False,
                 alpha=0.000001,
                 seed=42,
                 verbose=False,
                 max_train_N=None):
        """
        RLinear OLS wrapper
        args:
            dataset_train: training dataset
            context_length: features that the linear model will see
            horizon: up until where the linear model will forecast
            instance_norm: switch on or off instance normalisation
            individual: determines if separate model should be learned per channel
            alpha: regularization parameter for Ridge regression
            seed: for repeatability when using SVD solver
            max_train_N: set this if your dataset is very large and you want to subsample
        """
        self.context_length = context_length
        self.horizon = horizon
        self.dataset = dataset_train
        self.individual = individual
        self.instance_norm = instance_norm
        self.max_train_N = max_train_N
        self.verbose = verbose
        
        # Disable 'fit_intercept' in Ridge regression when instance normalization is used
        fit_intercept = False if instance_norm else True
        
        # Initialize RevIN
        self.revin = RevIN(num_features=dataset_train.shape[1], eps=1e-5, affine=True)
        
        if self.individual:
            self.models = []
            for _ in range(dataset_train.shape[1]):
                self.models.append(Ridge(alpha=alpha,
                                       fit_intercept=fit_intercept, 
                                       tol=0.00001, 
                                       copy_X=True, 
                                       max_iter=None, 
                                       solver='svd', 
                                       random_state=seed))
        else:
            self.model = Ridge(alpha=alpha,
                             fit_intercept=fit_intercept, 
                             tol=0.00001, 
                             copy_X=True, 
                             max_iter=None, 
                             solver='svd', 
                             random_state=seed)
        
        self.fit_ols_solutions()
    
    def fit_ols_solutions(self):
        """
        Fit the OLS solutions for each series or in a global mode.
        """
        # Convert to torch tensor for RevIN
        dataset_tensor = torch.FloatTensor(self.dataset).unsqueeze(0)  # [1, D, V]
        
        # Apply RevIN normalization
        dataset_normalized = self.revin(dataset_tensor, mode='norm')
        dataset_normalized = dataset_normalized.squeeze(0).detach().numpy()  # [D, V]
        
        # Create sliding window instances
        instances = np.lib.stride_tricks.sliding_window_view(dataset_normalized, (self.context_length+self.horizon), axis=0)
        
        if self.instance_norm:
            if self.verbose:
                print('Subtracting means (additional instance normalization)')
            context_means = np.mean(instances[:,:,:self.context_length], axis=2, keepdims=True)
            instances = instances - context_means
        
        X = instances[:,:,:self.context_length]  # (D_trimmed, V, context)
        y = instances[:,:,self.context_length:]  # (D_trimmed, V, horizon)
        
        if self.instance_norm:
            # Concatenate the standard deviation when doing instance norm
            context_stds = np.sqrt(np.var(instances[:,:,:self.context_length], axis=2, keepdims=True) + 1e-5)
            X = np.concatenate((X, context_stds), axis=2)  # (D_trimmed, V, context+1)
        
        if self.verbose:
            print('Fitting RLinear models')

        if self.individual:
            for series_idx in range(X.shape[1]):
                if self.verbose:
                    print(f'\t Fitting in individual mode, series idx {series_idx}')

                X_data = X[:,series_idx,:]
                y_data = y[:,series_idx,:]
                if self.max_train_N is not None and X_data.shape[0]>self.max_train_N:
                    idxs = np.arange(X_data.shape[0])
                    idxs = np.random.choice(idxs, size=self.max_train_N, replace=False)
                    self.models[series_idx].fit(X_data[idxs], y_data[idxs])
                else:
                    self.models[series_idx].fit(X_data, y_data)
        else:
            # Flatten 3D data into 2D data: training instances across all variables for 'global' mode
            X_data = np.reshape(X, (X.shape[0]*X.shape[1], -1))
            y_data = np.reshape(y, (y.shape[0]*y.shape[1], -1))
            
            if self.max_train_N is not None and X_data.shape[0]>self.max_train_N:
                idxs = np.arange(X_data.shape[0])
                idxs = np.random.choice(idxs, size=self.max_train_N, replace=False)
                self.model.fit(X_data[idxs], y_data[idxs])
            else:
                self.model.fit(X_data, y_data)
            self.weight_matrix = self.model.coef_   
            self.bias = self.model.intercept_  
    
    def predict(self, X):
        """
        Using the pre-fitted models and context, x, predict to horizon
        """
        D, V = X.shape[0], X.shape[1]
        
        # Convert to torch tensor for RevIN
        X_tensor = torch.FloatTensor(X).unsqueeze(0)  # [1, D, V, context]
        
        # Apply RevIN normalization
        X_normalized = self.revin(X_tensor, mode='norm')
        X_normalized = X_normalized.squeeze(0).detach().numpy()  # [D, V, context]
        
        if self.instance_norm:
            x_mean = np.mean(X_normalized, axis=2, keepdims=True)
            X_normalized = X_normalized - x_mean
            x_std = np.sqrt(np.var(X_normalized, axis=2, keepdims=True) + 1e-5)
            X_normalized = np.concatenate((X_normalized, x_std), axis=2)
        
        if self.individual:
            preds = []
            for series_idx in range(X_normalized.shape[1]):
                pred_i = self.models[series_idx].predict(X_normalized[:,series_idx])
                preds.append(pred_i[:,np.newaxis])
            preds = np.concatenate(preds, axis=1)
        else:
            pred = self.model.predict(X_normalized.reshape(D*V, -1))
            preds = pred.reshape(D, V, -1)
        
        # For denormalization, we need to manually apply the inverse RevIN transformation
        # since the prediction horizon is different from the input context length
        
        # Get the statistics from the original input X (context)
        X_original = X  # [D, V, context]
        
        # Calculate mean and std for each sample in the context
        context_means = np.mean(X_original, axis=2, keepdims=True)  # [D, V, 1]
        context_stds = np.sqrt(np.var(X_original, axis=2, keepdims=True) + 1e-5)  # [D, V, 1]
        
        # Apply inverse RevIN transformation to predictions
        # First, we need to get the affine parameters from RevIN
        with torch.no_grad():
            # Create a dummy tensor to get the affine parameters
            dummy_tensor = torch.FloatTensor(X_original).unsqueeze(0)
            self.revin(dummy_tensor, mode='norm')  # This sets the statistics
            
            # Get affine parameters
            if self.revin.affine:
                affine_weight = self.revin.affine_weight.detach().numpy()  # [V]
                affine_bias = self.revin.affine_bias.detach().numpy()  # [V]
            else:
                affine_weight = np.ones(V)
                affine_bias = np.zeros(V)
        
        # Apply inverse affine transformation
        preds_denormalized = preds - affine_bias  # [D, V, horizon]
        preds_denormalized = preds_denormalized / (affine_weight + 1e-5)  # [D, V, horizon]
        
        # Apply inverse normalization
        preds_denormalized = preds_denormalized * context_stds  # [D, V, horizon]
        preds_denormalized = preds_denormalized + context_means  # [D, V, horizon]
        
        if self.instance_norm:
            return preds_denormalized + x_mean  # Undo instance norm
        else:
            return preds_denormalized
