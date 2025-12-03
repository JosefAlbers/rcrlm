from .utils import apply_rope, create_rope_applier

import mlx.core as mx
import mlx.nn as nn

class Attention(nn.Module):
    def __init__(self, config, *, rcr_idx):
        super().__init__()
        self.n_q_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.n_repeat = self.n_q_heads // self.n_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, self.n_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_q_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.rot_dims = rot_dims = None if getattr(config, "partial_rotary_factor", 1.0)>=1.0 else int(self.head_dim * getattr(config, "partial_rotary_factor", 1.0))
        self.apply_rope = mx.compile(create_rope_applier(rot_dims, config.rope_traditional))

    def __call__(self, x, attention_mask, rope, cache):
        B, L, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q_norm(q.reshape(B, L, self.n_q_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(k.reshape(B, L, self.n_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = self.apply_rope(q, k, rope[0], rope[1], rot_dims=self.rot_dims)
        k, v = cache(k, v)
        o = mx.fast.scaled_dot_product_attention(q,k,v,scale=self.scale,mask=attention_mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o)

class MLP(nn.Module):
    def __init__(self, config, *, rcr_idx):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def __call__(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(nn.silu(gate) * up)

class TransformerBlock(nn.Module):
    def __init__(self, config, *, rcr_idx=None):
        super().__init__()
        self.self_attn = Attention(config, rcr_idx=rcr_idx)
        self.mlp = MLP(config, rcr_idx=rcr_idx)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, attention_mask, rope, cache):
        h = self.self_attn(self.input_layernorm(x), attention_mask=attention_mask, rope=rope, cache=cache)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x))

class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, input_ids, attention_mask, rope, cache, hiddens=None):
        captures = []
        x = self.embed_tokens(input_ids)
        for _idx, (c, layer) in enumerate(zip(cache, self.layers)):
            x = layer(x, attention_mask=attention_mask, rope=rope, cache=c)
            if hiddens is not None and _idx in hiddens:
                captures.append(x)
        return self.norm(x), captures

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tie = tie = config.tie_word_embeddings
        self.model = Qwen3Model(config)
        if not tie:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # else:
        #     self.lm_head = self.model.embed_tokens.as_linear

    def __call__(self, input_ids, attention_mask, rope, cache, hiddens=None):
        x, captures = self.model(input_ids, attention_mask=attention_mask, rope=rope, cache=cache, hiddens=hiddens)
        if self.tie:
            x = self.model.embed_tokens.as_linear(x)
        else:
            x = self.lm_head(x)
        # x = self.lm_head(x)
        if hiddens is None:
            return x
        return x, captures

