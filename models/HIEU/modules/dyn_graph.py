import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGraph(nn.Module):
    """
    Dynamic Graph Module - generates time-varying adjacency matrix
    conditioned on input features (context-dependent).
    
    Key improvements over static version:
    - A is generated dynamically based on input features (truly time-varying)
    - Supports both positive and negative edge weights (signed graph)
    - Can model asymmetric relationships (lead-lag structure)
    """
    def __init__(self, num_nodes: int, hidden_dim: int = 64, 
                 use_signed: bool = True, use_asymmetric: bool = True):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.use_signed = use_signed
        self.use_asymmetric = use_asymmetric
        
        # Base adjacency (learnable prior)
        self.raw_A_base = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.1)
        
        # Dynamic adjacency generator: features -> adjacency perturbation
        # Input: [B, N] node features -> Output: [B, N, N] dynamic adjacency
        self.adj_generator = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes * num_nodes)
        )
        
        # Context MLP (for graph-level context vector)
        self.ctx_mlp = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Scaling factor for dynamic component
        self.dynamic_scale = nn.Parameter(torch.tensor(0.1))
    
    def _make_adjacency(self, raw_A: torch.Tensor, batch: bool = False) -> torch.Tensor:
        """
        Convert raw adjacency to valid adjacency matrix.
        
        Args:
            raw_A: Raw adjacency values [N, N] or [B, N, N]
            batch: Whether input is batched
        
        Returns:
            Processed adjacency matrix
        """
        if self.use_signed:
            # Allow negative edges (tanh for bounded range [-1, 1])
            A = torch.tanh(raw_A)
        else:
            # Non-negative edges only
            A = F.softplus(raw_A)
        
        if not self.use_asymmetric:
            # Make symmetric
            if batch:
                A = (A + A.transpose(-2, -1)) / 2.0
            else:
                A = (A + A.t()) / 2.0
        
        # Zero diagonal (no self-loops)
        if batch:
            # [B, N, N]
            mask = 1.0 - torch.eye(self.num_nodes, device=A.device).unsqueeze(0)
            A = A * mask
        else:
            A = A - torch.diag_embed(torch.diag(A))
        
        return A
    
    def get_base_adjacency(self) -> torch.Tensor:
        """Get the base (static) adjacency matrix."""
        return self._make_adjacency(self.raw_A_base, batch=False)
    
    def get_dynamic_adjacency(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generate dynamic adjacency matrix conditioned on input features.
        
        Args:
            features: Node features [B, N]
        
        Returns:
            Dynamic adjacency matrix [B, N, N]
        """
        B = features.shape[0]
        
        # Generate dynamic perturbation
        delta_A = self.adj_generator(features)  # [B, N*N]
        delta_A = delta_A.view(B, self.num_nodes, self.num_nodes)  # [B, N, N]
        
        # Combine base + dynamic (scaled)
        base_A = self.get_base_adjacency().unsqueeze(0)  # [1, N, N]
        raw_A = base_A + self.dynamic_scale * delta_A  # [B, N, N]
        
        # Process to valid adjacency
        A = self._make_adjacency(raw_A, batch=True)
        
        return A
    
    def adjacency(self, features: torch.Tensor = None) -> torch.Tensor:
        """
        Get adjacency matrix (dynamic if features provided, else base).
        
        Args:
            features: Optional node features [B, N] for dynamic adjacency
        
        Returns:
            Adjacency matrix [N, N] or [B, N, N]
        """
        if features is None:
            return self.get_base_adjacency()
        else:
            return self.get_dynamic_adjacency(features)
    
    def laplacian_smoothness(self, W: torch.Tensor, A: torch.Tensor = None, 
                             weight: float = 1e-3) -> torch.Tensor:
        """
        Compute Laplacian smoothness regularization.
        
        Args:
            W: Per-node parameter vectors [N, d] or [B, N, d]
            A: Adjacency matrix (uses base if None)
            weight: Regularization weight
        
        Returns:
            Smoothness loss scalar
        """
        if A is None:
            A = self.get_base_adjacency().detach()
        
        # Handle batched case
        if len(A.shape) == 3:
            # Average over batch
            A = A.mean(dim=0).detach()
        
        # For signed graphs, use absolute values for Laplacian
        A_abs = torch.abs(A)
        D = torch.diag(A_abs.sum(dim=-1))
        L = D - A_abs
        
        if len(W.shape) == 3:
            # Batched W: [B, N, d] -> average
            W = W.mean(dim=0)
        
        return weight * torch.trace(W.t() @ L @ W)
    
    def prior_loss(self, A_prior: torch.Tensor, A_learned: torch.Tensor = None,
                   weight: float = 1e-3) -> torch.Tensor:
        """
        Compute prior alignment loss.
        
        Args:
            A_prior: Prior adjacency matrix [N, N]
            A_learned: Learned adjacency (uses base if None)
            weight: Loss weight
        
        Returns:
            Prior loss scalar
        """
        if A_learned is None:
            A_learned = self.get_base_adjacency()
        
        # Handle batched A_learned
        if len(A_learned.shape) == 3:
            A_learned = A_learned.mean(dim=0)
        
        return weight * F.mse_loss(A_learned, A_prior)
    
    def get_graph_statistics(self, A: torch.Tensor) -> dict:
        """
        Compute interpretable graph statistics.
        
        Args:
            A: Adjacency matrix [N, N] or [B, N, N]
        
        Returns:
            Dictionary of statistics
        """
        if len(A.shape) == 3:
            A = A.mean(dim=0)  # Average over batch
        
        A_abs = torch.abs(A)
        
        stats = {
            'in_degree': A_abs.sum(dim=0),  # [N]
            'out_degree': A_abs.sum(dim=1),  # [N]
            'asymmetry': (A - A.t()).abs().mean(),  # Scalar
            'sparsity': (A_abs < 0.1).float().mean(),  # Scalar
            'positive_ratio': (A > 0).float().mean(),  # Scalar (for signed)
            'negative_ratio': (A < 0).float().mean(),  # Scalar (for signed)
        }
        return stats
    
    @staticmethod
    def gc_features_from_adj(A: torch.Tensor) -> torch.Tensor:
        """
        Extract Granger-causality-like features from adjacency.
        
        Args:
            A: Adjacency matrix [N, N] (j -> i as A[i, j])
        
        Returns:
            Node features [N, 3] (in_degree, out_degree, asymmetry)
        """
        A_abs = torch.abs(A)
        deg_in = A_abs.sum(dim=1)
        deg_out = A_abs.sum(dim=0)
        asym = deg_in - deg_out
        stats = torch.stack([deg_in, deg_out, asym], dim=-1)
        return stats
    
    def forward(self, features: torch.Tensor) -> tuple:
        """
        Forward pass: generate context and dynamic adjacency.
        
        Args:
            features: Node features [B, N] (e.g., last timestep values)
        
        Returns:
            ctx: Graph context vector [B, hidden_dim]
            A: Dynamic adjacency matrix [B, N, N]
        """
        # Generate graph context
        ctx = self.ctx_mlp(features)  # [B, hidden_dim]
        
        # Generate dynamic adjacency
        A = self.get_dynamic_adjacency(features)  # [B, N, N]
        
        return ctx, A
