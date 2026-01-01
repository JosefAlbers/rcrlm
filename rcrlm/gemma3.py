import mlx.core as mx
import mlx.nn as nn
from functools import partial
import math

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)

@partial(mx.compile, shapeless=True)
def clip_residual(x, y):
    if x.dtype != mx.float16:
        return x + y
    return mx.clip(x.astype(mx.float32) + y.astype(mx.float32), -65504, 65504).astype(mx.float16)

def make_sliding_mask_bool(global_mask, window_size, offset=None):
    L_q = global_mask.shape[-2]
    L_k = global_mask.shape[-1]
    if offset is None:
        offset = L_k - L_q
    q_idx = mx.arange(offset, offset + L_q)[:, None]
    k_idx = mx.arange(L_k)[None, :] 
    is_within_window = (q_idx - k_idx) < window_size
    return global_mask & is_within_window[None, None, :, :]

class Attention(nn.Module):
    def __init__(self, config, *, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = getattr(config, "query_pre_attn_scalar", 256)**-0.5
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        pattern = getattr(config, "_sliding_window_pattern", 6)
        self.is_global = (layer_idx + 1) % pattern == 0
        self.window_size = getattr(config, "sliding_window", 512)
        if self.is_global:
            self.rope = nn.RoPE(dims=self.head_dim, traditional=False, base=config.rope_theta)
        else:
            self.rope = nn.RoPE(dims=self.head_dim, traditional=False, base=config.rope_local_base_freq)

    def __call__(self, x, attention_mask, rope, cache):
        B, L, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        offset = cache.offset
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        k, v = cache(k, v)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=attention_mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o)

class MLP(nn.Module):
    def __init__(self, config, *, layer_idx):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, config, *, layer_idx=None):
        super().__init__()
        self.self_attn = Attention(config, layer_idx=layer_idx)
        self.mlp = MLP(config, layer_idx=layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, attention_mask, rope, cache):
        r = self.self_attn(self.input_layernorm(x), attention_mask, rope, cache)
        h = clip_residual(x, self.post_attention_layernorm(r))
        r = self.mlp(self.pre_feedforward_layernorm(h))
        out = clip_residual(h, self.post_feedforward_layernorm(r))
        return out

class Gemma3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = [
            TransformerBlock(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ]
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.sliding_window = getattr(config, "sliding_window", 512)
        self.sliding_window_pattern = getattr(config, "_sliding_window_pattern", 6)

    def __call__(self, input_ids, attention_mask, rope, cache, hiddens=None):
        captures = []
        x = self.embed_tokens(input_ids)
        x *= mx.array(self.config.hidden_size**0.5, x.dtype)
        global_mask = attention_mask
        sliding_mask = global_mask
        if self.sliding_window_pattern > 1:
            sliding_mask = make_sliding_mask_bool(attention_mask, self.sliding_window)
        for _idx, (c, layer) in enumerate(zip(cache, self.layers)):
            is_global = (_idx % self.sliding_window_pattern == self.sliding_window_pattern - 1)
            mask = global_mask if is_global else sliding_mask
            x = layer(x, mask, rope, c)
            if hiddens is not None and _idx in hiddens:
                captures.append(x)
        return self.norm(x), captures

class Gemma3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tie = tie = getattr(config, "tie_word_embeddings", None) is not False
        self.model = Gemma3Model(config)
        self._config = config
        if not tie:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids, attention_mask, rope, cache, hiddens=None):
        x, captures = self.model(input_ids, attention_mask, rope, cache, hiddens=hiddens)
        if self.tie:
            x = self.model.embed_tokens.as_linear(x)
        else:
            x = self.lm_head(x)
        if hiddens is None:
            return x
        return x, captures
