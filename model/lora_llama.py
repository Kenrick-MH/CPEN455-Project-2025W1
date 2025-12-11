import torch
from torch import nn
from .llama import LlamaModel
from .cache import DynamicCache, Cache
from typing import Iterable, List, Optional, Callable
from .attention import LlamaAttention, apply_rotary_pos_emb, eager_attention_forward

class LoraAttention(nn.Module):
    def __init__(self, config, base_attention:LlamaAttention, lora_dim, sigma):
        super().__init__()
        self.base_attention = base_attention
        
        self.head_dim = self.base_attention.head_dim
            
        # Down projections        
        self.A_v_proj = nn.Linear(config.hidden_size, lora_dim, bias=False)
        self.A_k_proj = nn.Linear(config.hidden_size, lora_dim, bias=False)
        self.A_o_proj = nn.Linear(config.num_attention_heads * self.head_dim, lora_dim, bias=False)

        # Up projections
        self.B_v_proj = nn.Linear(lora_dim, config.num_key_value_heads * self.head_dim, bias=False)
        self.B_k_proj = nn.Linear(lora_dim, config.num_key_value_heads * self.head_dim, bias=False)
        self.B_o_proj = nn.Linear(lora_dim, config.hidden_size, bias=False)
        
        nn.init.normal_(self.A_v_proj.weight, sigma)
        nn.init.normal_(self.A_k_proj.weight, sigma)
        nn.init.normal_(self.A_o_proj.weight, sigma)
    
        nn.init.zeros_(self.B_v_proj.weight)
        nn.init.zeros_(self.B_k_proj.weight)
        nn.init.zeros_(self.B_o_proj.weight)
        
        # Freeze base layer parameters
        for param in self.base_attention.parameters():
            param.requires_grad = False
        self.base_attention.eval()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the attention layer.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            position_embeddings: Tuple of (cos, sin) from RoPE [batch, seq_len, head_dim]
            attention_mask: Causal attention mask [batch, 1, seq_len, total_len]
            past_key_values: Cache for storing previous key-values
            cache_position: Position indices for current tokens
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (attention_output, attention_weights):
                - attention_output: [batch, seq_len, hidden_size]
                - attention_weights: [batch, num_heads, seq_len, total_len]
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project to queries, keys, values and reshape to [batch, num_heads, seq_len, head_dim]
        query_states = self.base_attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.base_attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.base_attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        value_lora = self.B_v_proj(self.A_v_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
        key_lora =  self.B_k_proj(self.A_k_proj(hidden_states)).view(hidden_shape).transpose(1, 2)

        K = key_states + key_lora
        V = value_states + value_lora

        # Apply Rotary Position Embedding to queries and keys
        cos, sin = position_embeddings
        query_states, K = apply_rotary_pos_emb(query_states, K, cos, sin)

        # Update cache with new key-value pairs
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            K, V = past_key_values.update(K, V, self.base_attention.layer_idx, cache_kwargs)

        # Compute attention using the selected implementation
        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self.base_attention,
            query_states,
            K,
            V,
            attention_mask,
            dropout=0.0 if not self.base_attention.training else self.base_attention.attention_dropout,
            scaling=self.base_attention.scaling,
            **kwargs,
        )

        # Reshape back to [batch, seq_len, hidden_size] and apply output projection
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.base_attention.o_proj(attn_output)
        lora_out_proj = self.B_o_proj(self.A_o_proj(attn_output))
        
        O = attn_output + lora_out_proj
        
        return O, attn_weights
    
class LoraLlamaModel(nn.Module):
    def __init__(self, base_model: LlamaModel, lora_dim, sigma = 1) -> None:
        super().__init__()
        self.model = base_model
        self.config = base_model.config        
        self.lora_dim = lora_dim
        
        
        # Freeze model params first
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        # Replace all layers with LoRa layers
        for i, layer in enumerate(self.model.layers):
            self.model.layers[i].self_attn = LoraAttention(
                self.config, layer.self_attn, self.lora_dim, sigma
            )

    def forward(self, *args, **kwargs):
        self.model.eval()
        return self.model(*args, **kwargs)
        
