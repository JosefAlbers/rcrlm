from .utils import apply_rope, create_rope_applier

import mlx.core as mx
import mlx.nn as nn
import math

class Retention(nn.Module):
    def __init__(self, config, *, rcr_idx=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = config.head_dim
        self.scale = head_dim ** -0.5
        self.n_repeat = n_heads // n_kv_heads
        val_dim = n_heads * head_dim
        self.q_proj = nn.Linear(config.hidden_size, val_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, n_kv_heads * head_dim, bias=False)
        self.g_proj_new = nn.Linear(config.hidden_size, val_dim, bias=False) # The RetNet Gate
        self.o_proj = nn.Linear(val_dim, config.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        xmin, xmax = math.log(1 / 32), math.log(1 / 512)
        x = mx.linspace(xmin, xmax, num=n_heads)
        self._gamma = mx.stop_gradient(1 - x.exp())
        self.gn_new = nn.GroupNorm(num_groups=n_heads, dims=val_dim, affine=False)
        self.rot_dims = rot_dims = None if getattr(config, "partial_rotary_factor", 1.0) >= 1.0 else int(self.head_dim * getattr(config, "partial_rotary_factor", 1.0))
        self.apply_rope = mx.compile(create_rope_applier(rot_dims, config.rope_traditional))

    def __call__(self, x, attention_mask, rope, cache):
        B, L, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q_norm(q.reshape(B, L, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(k.reshape(B, L, self.n_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = self.apply_rope(q, k, rope[0], rope[1], rot_dims=self.rot_dims)
        if self.n_repeat > 1:
            k = mx.repeat(k, self.n_repeat, axis=1)
            v = mx.repeat(v, self.n_repeat, axis=1)
        if L > 1:
            prev_s, n_prev = cache.get_sn() if getattr(cache, 'set_sn', None) else (None, 0)
            out, final_state = self.parallel_retention(q, k, v, prev_s)
            if getattr(cache, 'set_sn', None):
                cache.set_sn(final_state, n_prev + L)
        else:
            out = self.recurrent_retention(q, k, v, cache)
        out = out.transpose(0, 2, 1, 3).reshape(B*L, -1)
        out = self.gn_new(out)
        g = self.g_proj_new(x)
        out = nn.silu(g) * out
        out = out.reshape(B, L, -1)
        return self.o_proj(out)

    def parallel_retention(self, q, k, v, prev_state=None):
        B, H, L, D = q.shape
        q_scaled = q * self.scale
        attn = q_scaled @ k.transpose(0, 1, 3, 2)
        attn = attn * self._decay(L)
        h = attn @ v
        decay_vec = self._gamma[:, None] ** (L - 1 - mx.arange(L)[None, :])
        k_decayed = k * decay_vec[None, :, :, None]
        current_chunk_state = k_decayed.transpose(0, 1, 3, 2) @ v
        
        if prev_state is not None:
            chunk_decay = self._gamma[:, None, None] ** L 
            final_state = (prev_state * chunk_decay) + current_chunk_state
        else:
            final_state = current_chunk_state
        return h, final_state

    def recurrent_retention(self, q, k, v, cache):
        k = k * self.scale
        gamma = self._gamma[None, :, None, None]
        ktv = k.transpose(0, 1, 3, 2) @ v
        s, n = cache.get_sn()
        s = gamma * s + ktv
        cache.set_sn(s, n+1)
        h = q @ s
        return h

    def _decay(self, sequence_length):
        n = mx.arange(sequence_length)[:, None]
        m = mx.arange(sequence_length)[None]
        mask = (n >= m)
        diff = n - m
        gamma = self._gamma[:, None, None]
        D = (gamma ** diff) * mask
        return D

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

    def to_retention(self, config):
        ret_net = Retention(config)
        ret_net.q_proj.weight = self.q_proj.weight
        ret_net.k_proj.weight = self.k_proj.weight
        ret_net.v_proj.weight = self.v_proj.weight
        ret_net.o_proj.weight = self.o_proj.weight
        ret_net.q_norm.weight = self.q_norm.weight
        ret_net.k_norm.weight = self.k_norm.weight
        # {{ init
        gain = 2**-2.5
        fan_out, fan_in = ret_net.g_proj_new.weight.shape
        limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
        ret_net.g_proj_new.weight = mx.random.uniform(low=-limit, high=limit, shape=ret_net.g_proj_new.weight.shape)
        # }} init
        return ret_net

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
        self.self_attn = Retention(config, rcr_idx=rcr_idx) if config.extra_config.get('rectify', False) else Attention(config, rcr_idx=rcr_idx)
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
        self._config = config
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

