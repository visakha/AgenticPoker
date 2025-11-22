"""
Transformer-based LLM for poker decision making.

This module implements a small transformer model that:
- Takes encoded poker game states as input
- Outputs action probabilities (policy head)
- Outputs state value estimation (value head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Single transformer block with multi-head attention and FFN."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through transformer block."""
        # Multi-head attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class PokerTransformer(nn.Module):
    """
    Transformer-based model for poker decision making.
    
    Architecture:
    - Input projection layer
    - Positional encoding
    - N transformer blocks
    - Policy head (action probabilities)
    - Value head (state value estimation)
    """
    
    def __init__(
        self,
        state_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        num_actions: int = 5,
        dropout: float = 0.1,
        max_seq_len: int = 128
    ):
        """
        Initialize poker transformer.
        
        Args:
            state_dim: Dimension of input state encoding
            d_model: Model embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            num_actions: Number of possible actions
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.d_model = d_model
        self.num_actions = num_actions
        
        # Input projection
        self.input_proj = nn.Linear(state_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output heads
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        state: torch.Tensor, 
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            state: Input state tensor of shape (batch_size, state_dim) or
                   (batch_size, seq_len, state_dim) for sequences
            mask: Optional attention mask
        
        Returns:
            Tuple of (action_logits, state_value)
            - action_logits: (batch_size, num_actions)
            - state_value: (batch_size, 1)
        """
        # Handle both single state and sequence inputs
        if state.dim() == 2:
            # Single state: (batch_size, state_dim) -> (batch_size, 1, state_dim)
            state = state.unsqueeze(1)
        
        batch_size, seq_len, _ = state.shape
        
        # Project input to model dimension
        x = self.input_proj(state)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Use last token for prediction (or mean pooling)
        # Here we use the last token
        x = x[:, -1, :]  # (batch_size, d_model)
        
        # Compute policy and value
        action_logits = self.policy_head(x)  # (batch_size, num_actions)
        state_value = self.value_head(x)  # (batch_size, 1)
        
        return action_logits, state_value
    
    def get_action_probs(
        self, 
        state: torch.Tensor, 
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Get action probabilities from state.
        
        Args:
            state: Input state
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            Action probabilities (batch_size, num_actions)
        """
        action_logits, _ = self.forward(state)
        return F.softmax(action_logits / temperature, dim=-1)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get state value estimation.
        
        Args:
            state: Input state
        
        Returns:
            State value (batch_size, 1)
        """
        _, value = self.forward(state)
        return value
    
    def sample_action(
        self, 
        state: torch.Tensor, 
        temperature: float = 1.0,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        
        Args:
            state: Input state
            temperature: Sampling temperature
            deterministic: If True, return argmax action
        
        Returns:
            Tuple of (action, log_prob)
        """
        action_probs = self.get_action_probs(state, temperature)
        
        if deterministic:
            action = action_probs.argmax(dim=-1)
            log_prob = torch.log(action_probs.gather(-1, action.unsqueeze(-1)))
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def evaluate_actions(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO training.
        
        Args:
            state: Input states
            action: Actions taken
        
        Returns:
            Tuple of (log_probs, state_values, entropy)
        """
        action_logits, state_values = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_probs, state_values.squeeze(-1), entropy
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    state_dim = 200  # Example state dimension
    batch_size = 4
    
    model = PokerTransformer(
        state_dim=state_dim,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        num_actions=5
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    dummy_state = torch.randn(batch_size, state_dim)
    action_logits, value = model(dummy_state)
    
    print(f"Action logits shape: {action_logits.shape}")
    print(f"Value shape: {value.shape}")
    
    # Test action sampling
    action, log_prob = model.sample_action(dummy_state)
    print(f"Sampled action: {action}")
    print(f"Log prob: {log_prob}")
