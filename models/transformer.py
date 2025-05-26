import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class TyreDegradationTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 100
    ):
        super().__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.uniform_(-initrange, initrange)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(
        self, 
        src: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the transformer model.
        
        Args:
            src: Input sequences of shape (batch_size, seq_len, input_dim)
            src_mask: Mask for source sequences
            return_attention: Whether to return attention weights
        
        Returns:
            Predicted pace delta of shape (batch_size, 1)
        """
        # Input projection and positional encoding
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)
        
        # Transformer encoder
        if return_attention:
            # Store attention weights for explainability
            attention_weights = []
            x = src
            for layer in self.transformer_encoder.layers:
                x = layer(x, src_mask)
                # Note: Getting attention weights from transformer layers is complex
                # This is a simplified version for demonstration
            encoded = x
        else:
            encoded = self.transformer_encoder(src, src_mask)
        
        # Global average pooling across sequence dimension
        pooled = encoded.mean(dim=1)
        
        # Output prediction
        output = self.output_layers(pooled)
        
        if return_attention:
            return output, attention_weights
        return output

class AttentionTransformer(nn.Module):
    """
    Enhanced transformer with explicit attention extraction for explainability.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 100
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(
        self, 
        src: torch.Tensor, 
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass with optional attention weight extraction.
        
        Args:
            src: Input sequences of shape (batch_size, seq_len, input_dim)
            return_attention: Whether to return attention weights
        
        Returns:
            Tuple of (predicted_pace_delta, attention_weights)
        """
        # Input projection and positional encoding
        x = self.input_projection(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        attention_weights = [] if return_attention else None
        
        # Pass through attention layers
        for i, (attn_layer, ffn_layer, ln1, ln2) in enumerate(
            zip(self.attention_layers, self.ffn_layers, self.layer_norms1, self.layer_norms2)
        ):
            # Multi-head attention
            if return_attention:
                attn_output, attn_weights = attn_layer(x, x, x)
                attention_weights.append(attn_weights.detach())
            else:
                attn_output, _ = attn_layer(x, x, x)
            
            # Residual connection and layer norm
            x = ln1(x + attn_output)
            
            # Feed-forward network
            ffn_output = ffn_layer(x)
            x = ln2(x + ffn_output)
        
        # Global average pooling
        pooled = x.mean(dim=1)
        
        # Output prediction
        output = self.output_layers(pooled)
        
        return output, attention_weights

def create_model(
    input_dim: int,
    model_type: str = "transformer",
    **kwargs
) -> nn.Module:
    """
    Factory function to create different model variants.
    
    Args:
        input_dim: Number of input features
        model_type: Type of model to create
        **kwargs: Additional model parameters
    
    Returns:
        Initialized model
    """
    if model_type == "transformer":
        return TyreDegradationTransformer(input_dim, **kwargs)
    elif model_type == "attention":
        return AttentionTransformer(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Model configuration presets
MODEL_CONFIGS = {
    "small": {
        "d_model": 256,
        "nhead": 4,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "dim_feedforward": 1024,
        "dropout": 0.1
    },
    "medium": {
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1
    },
    "large": {
        "d_model": 768,
        "nhead": 12,
        "num_encoder_layers": 12,
        "num_decoder_layers": 12,
        "dim_feedforward": 3072,
        "dropout": 0.1
    }
}
