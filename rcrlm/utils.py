# {{{ === PREP ===
from datetime import datetime
import os
import functools
import json
import time
import math
from urllib.request import urlretrieve
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from typing import List, Tuple, Optional, Dict, Any, Union, Type, Callable
import glob
from tokenizerz import Tokenizer
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

PRETTY_HW = '─'*30

def strftime_now(format="%Y-%m-%d %H:%M:%S"):
    return datetime.now().strftime(format)

def tqdm_hook(t):
    last_b = [0]
    def update_to(block_num=1, block_size=1, total_size=None):
        if total_size is not None:
            t.total = total_size
        downloaded = block_num * block_size
        t.update(downloaded - last_b[0])
        last_b[0] = downloaded
    return update_to

def download_file(url, path, desc, verbose=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        if verbose:
            print(f"File '{path}' already exists. Skipping.")
        return
    with tqdm(unit='B', unit_scale=True, desc=desc, leave=False) as t:
        urlretrieve(url, path, reporthook=tqdm_hook(t))

def get_model_files(repo, model, dest=None):
    base_url = f"https://huggingface.co/{repo}/{model}/resolve/main"
    model_dir = model if dest is None else os.path.join(dest, model)
    os.makedirs(model_dir, exist_ok=True)
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    try:
        if not os.path.exists(index_path):
            download_file(f"{base_url}/model.safetensors.index.json", index_path, "model index")
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        pattern = next(iter(weight_map.values()))
        if "-of-" in pattern:
            base = pattern[:pattern.find("-00")]
            count = int(pattern.split("-of-")[1].split("-")[0].split(".")[0])
            ext = pattern[pattern.rfind("."):]
            files += [f"{base}-{i:05d}-of-{count:05d}{ext}" for i in range(1, count + 1)]
        else:
            files.append(pattern)
    except Exception:
        files.append("model.safetensors")
    return model_dir, [(f"{base_url}/{file}", os.path.join(model_dir, file), file) for file in files]

def download_repo(repo, model, dest='models', max_workers=4):
    model_dir, tasks = get_model_files(repo, model, dest)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_file, url, path, desc) for url, path, desc in tasks]
        for future in futures:
            future.result()
    return model_dir

@dataclass
class Config:
    architectures: List[str]
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    eos_token_id: str
    rms_norm_eps: float = 1e-6
    vocab_size: int = 0
    num_key_value_heads: int = None
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    torch_dtype: str = "float32"
    head_dim: int = None
    attention_bias: bool = True
    mlp_bias: bool = False
    rope_traditional: bool = True
    partial_rotary_factor: float = 1.0
    max_position_embeddings: Optional[int] = None
    original_max_position_embeddings: Optional[int] = None
    logits_scaling: float = 1.0
    attention_multiplier: float = 1.0
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    extra_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @property
    def dtype(self):
        return eval(f'mx.{self.torch_dtype}')

def get_nested(d, keys, default=None):
    if not isinstance(d, dict):
        return default
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d

def load_config(model_name, cls=Config):
    with open(Path(model_name) / 'config.json', 'r') as f:
        config_dict = json.load(f)
    cls_fields = {f.name for f in fields(cls)}
    init_args = {k: v for k, v in config_dict.items() if k in cls_fields}
    extra_args = {}
    for k, v in config_dict.items():
        if k not in cls_fields:
            extra_args[k] = v
    return cls(**init_args, extra_config=extra_args)


def load_model(model_cls, model_dir, model_cfg):
    def _get_wt(model_dir, model_cfg):
        if getattr(model_cfg, 'sanitized', False):
            return [(k, v) for wf in glob.glob(f"{model_dir}/*.safetensors") for k, v in mx.load(wf).items()]
        return [(k, v.transpose(0, 2, 3, 1) if "patch_embedding.weight" in k else v) for wf in glob.glob(f"{model_dir}/*.safetensors") for k, v in mx.load(wf).items()]

    model = model_cls(model_cfg)
    model.load_weights(_get_wt(model_dir, model_cfg), strict=False)
    # # {{
    # if lora_cfg:
    #     model.freeze()
    #     linear_to_lora_layers(model, lora_layers=lora_cfg['layers'], lora_targets=lora_cfg['targets'], lora_rank=lora_cfg['rank'], lora_scale=lora_cfg['scale'], lora_dropout=lora_cfg['dropout'])
    #     if lora_cfg['wt_from'] and os.path.exists(lora_cfg['wt_from']):
    #         model.load_weights(lora_cfg['wt_from'], strict=False)
    #     model.apply_to_modules(lambda k, v: v.unfreeze() if any(k.endswith(t) for t in lora_cfg['thaws']) else None)
    # # }}
    mx.eval(model)
    # model.eval()
    return model

def measure_performance(start_time, prompt_time, end_time, batch_size, seq_length, gen_length, verbose=True):
    prompt_duration = prompt_time - start_time
    generation_duration = end_time - prompt_time
    tokens_processed = batch_size * seq_length
    tokens_generated = gen_length * batch_size
    prompt_throughput = tokens_processed / prompt_duration if prompt_duration > 0 else 0
    generation_throughput = tokens_generated / generation_duration if generation_duration > 0 else 0
    metrics = {
        "prompt_throughput": prompt_throughput,
        "generation_throughput": generation_throughput,
        "prompt_tokens": tokens_processed,
        "prompt_time": prompt_duration,
        "generation_tokens": tokens_generated,
        "generation_time": generation_duration
    }
    if verbose:
        print(f'┌{PRETTY_HW} Benchmark {PRETTY_HW}┐')
        print(f"Prompt processing: {prompt_throughput:8.1f} tokens/sec ({tokens_processed} tokens in {prompt_duration:.1f}s)")
        print(f"Tokens generation: {generation_throughput:8.1f} tokens/sec ({tokens_generated} tokens in {generation_duration:.1f}s)")
        print(f'└{PRETTY_HW*2}───────────┘')
    return metrics
# }}} === PREP ===
# {{{ === ROPE/CACHE ===
@mx.compile
def update_cache(max_len, cache_k, cache_v, new_k, new_v):
    seq_len = new_k.shape[2]
    shifted_k = mx.concatenate([cache_k[:, :, seq_len:, :], new_k], axis=2)
    shifted_v = mx.concatenate([cache_v[:, :, seq_len:, :], new_v], axis=2)
    return shifted_k, shifted_v

class RollCacher(nn.Module):
    def __init__(self, dtype, batch_size, num_heads, max_len, head_dim, k=None, v=None):
        super().__init__()
        self.max_len = max_len
        self.k = mx.zeros((batch_size, num_heads, max_len, head_dim), dtype=dtype) if k is None else k
        self.v = mx.zeros((batch_size, num_heads, max_len, head_dim), dtype=dtype) if v is None else v

    def __call__(self, k, v):
        # print(f'{k.shape=}')
        # print(f'{v.shape=}')
        # self.k, self.v = self_k, self_v = update_cache(self.max_len, self.k, self.v, k, v)
        self.k = self_k = mx.concatenate([self.k[:, :, k.shape[2]:, :], k], axis=2)
        self.v = self_v = mx.concatenate([self.v[:, :, v.shape[2]:, :], v], axis=2)
        return self_k, self_v

class CatCacher(nn.Module):
    def __init__(self, dtype, batch_size, num_heads, max_len, head_dim, k=None, v=None):
        super().__init__()
        self.k = mx.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype) if k is None else k
        self.v = mx.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype) if v is None else v

    def __call__(self, k, v):
        self.k = mx.concat([self.k, k], axis=2)
        self.v = mx.concat([self.v, v], axis=2)
        return self.k, self.v

class Roper:#(nn.Module):
    def __init__(self, config, su_len=None):
        # super().__init__()
        self.su_scale = 1.0
        if get_nested(config.extra_config, ["rope_scaling", "rope_type"])=='llama3':
            self._llama3(config)
        elif get_nested(config.extra_config, ["rope_scaling", "type"])=='longrope':
            self._su(config, su_len)
        else:
            dim = int(config.head_dim*getattr(config, "partial_rotary_factor", 1.0)/2)
            self.freq = 1.0 / (config.rope_theta ** (mx.arange(0, dim, dtype=mx.float32) / dim))
        self.dtype=config.dtype

    def __call__(self, positions):
        positions = positions[:, None, :, None]
        angles = positions * self.freq
        # angles = mx.concatenate([angles, angles], axis=-1) # for rotate_half
        cos = mx.cos(angles) * self.su_scale
        sin = mx.sin(angles) * self.su_scale
        # return mx.stop_gradient(cos), mx.stop_gradient(sin)
        return cos.astype(self.dtype), sin.astype(self.dtype)

    def _llama3(self, config):
        rot_dims = int(config.head_dim * config.partial_rotary_factor)
        scaling_config = get_nested(config.extra_config, ["rope_scaling"])
        factor = scaling_config.get("factor", 1.0)
        low_freq_factor = scaling_config.get("low_freq_factor", 1.0)
        high_freq_factor = scaling_config.get("high_freq_factor", 4.0)
        old_max = scaling_config.get("original_max_position_embeddings", 8192)
        idx = mx.arange(0, rot_dims, 2, dtype=mx.float32)
        freqs = config.rope_theta ** (idx / rot_dims)
        wavelens = 2 * mx.pi * freqs
        low_wl = old_max / low_freq_factor
        high_wl = old_max / high_freq_factor
        freqs_adj = mx.where(wavelens > low_wl, freqs * factor, freqs)
        is_med = (wavelens > high_wl) & (wavelens < low_wl)
        smooth = (old_max / wavelens - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_freqs = freqs_adj / ((1 - smooth) / factor + smooth)
        freqs_final = mx.where(is_med, smooth_freqs, freqs_adj)
        self.freq = nnx.Variable(1.0 / freqs_final)

    def _su(self, config, su_len):
        factor = 'long' if su_len > config.original_max_position_embeddings else 'short'
        self.su_scale = math.sqrt(1.0 + math.log(config.max_position_embeddings / config.original_max_position_embeddings) / math.log(config.original_max_position_embeddings))
        rot_dims = int(config.head_dim * config.partial_rotary_factor)
        scaling_config = get_nested(config.extra_config, ["rope_scaling"])
        freqs = config.rope_theta ** (mx.arange(0, rot_dims, 2, dtype=mx.float32) / rot_dims)
        factor = scaling_config.get(f'{factor}_factor')
        factor = mx.array(factor, dtype=mx.float32)
        self.freq = nnx.Variable(1.0 / (freqs * factor))

def apply_rope(dtype, q, k, cos, sin, rot_dims=None, traditional=False):
    if rot_dims is None:
        q_rot, k_rot = q, k
    else:
        q_rot, q_pass = q[..., :rot_dims], q[..., rot_dims:]
        k_rot, k_pass = k[..., :rot_dims], k[..., rot_dims:]
    if traditional:
        q_even = q_rot[..., 0::2]
        q_odd  = q_rot[..., 1::2]
        k_even = k_rot[..., 0::2]
        k_odd  = k_rot[..., 1::2]
        q_rotated = mx.stack([(q_even * cos - q_odd * sin), (q_even * sin + q_odd * cos)], axis=-1).reshape(q_rot.shape).astype(dtype)
        k_rotated = mx.stack([(k_even * cos - k_odd * sin), (k_even * sin + k_odd * cos)], axis=-1).reshape(k_rot.shape).astype(dtype)
    else:
        q_split = q_rot.reshape(*q.shape[:-1], 2, -1)
        k_split = k_rot.reshape(*k.shape[:-1], 2, -1)
        q_rotated = mx.concatenate([
            q_split[..., 0, :] * cos - q_split[..., 1, :] * sin,
            q_split[..., 1, :] * cos + q_split[..., 0, :] * sin,
        ], axis=-1).astype(dtype)
        k_rotated = mx.concatenate([
            k_split[..., 0, :] * cos - k_split[..., 1, :] * sin,
            k_split[..., 1, :] * cos + k_split[..., 0, :] * sin,
        ], axis=-1).astype(dtype)
    if rot_dims is None:
        return q_rotated, k_rotated
    else:
        q_out = mx.concatenate([q_rotated, q_pass], axis=-1)
        k_out = mx.concatenate([k_rotated, k_pass], axis=-1)
        return q_out, k_out

def rotate_half(dtype, _x, cos, sin, rot_dims):
    x, x_pass = _x[..., :rot_dims], _x[..., rot_dims:]
    midpoint = x.shape[-1] // 2
    x1, x2 = x[..., :midpoint], x[..., midpoint:]
    result = (x * cos) + (mx.concatenate([-x2, x1], axis = -1) * sin)
    return mx.concatenate([result, x_pass], axis=-1).astype(dtype)

def create_causal_mask(padding_mask):
    padding_mask = mx.array(padding_mask)
    seq_length = padding_mask.shape[1]
    causal_matrix = mx.tril(mx.ones((seq_length, seq_length), dtype=mx.bool_))
    causal_mask = causal_matrix & padding_mask[:, None, :]
    # causal_mask = mx.where(causal_matrix & padding_mask[:, None, :], 0.0, -1e9) # additive
    return causal_mask[:, None, :, :]
# }}} === ROPE/CACHE ===
# {{{ === INFER ===
def infer(
    prompts,
    model,
    tokenizer,
    config,
    lora_path = None,
    max_new_tokens = 100,
    use_chat_template = True,
    custom_tokenizer_fn: Callable = None,
    model_creator: Callable = None,
    stream = True,
    use_scan = False,
    use_jit = True,
    chat_template_kwargs = None,
    verbose = True,
):

    if lora_path and os.path.exists(lora_path):
        def decode_metadata(meta):
            out = {}
            for k, v in meta.items():
                try:
                    vv = json.loads(v)
                    out[k] = vv
                    continue
                except Exception:
                    pass
                out[k] = v
            return out
        lora_wts, lora_cfg = mx.load(lora_path, return_metadata=True)
        lora_cfg = decode_metadata(lora_cfg)
        model.freeze()
        linear_to_lora_layers(model, lora_layers=lora_cfg['layers'], lora_targets=lora_cfg['targets'], lora_rank=lora_cfg['rank'], lora_scale=lora_cfg['scale'], lora_dropout=lora_cfg['dropout'])
        model.load_weights(lora_wts, strict=False)
        model.apply_to_modules(lambda k, v: v.unfreeze() if any(k.endswith(t) for t in lora_cfg['thaws']) else None)
        mx.eval(model)
    model.eval()
    if isinstance(prompts, str):
        prompts = [prompts]
    if use_chat_template:
        try:
            if chat_template_kwargs is None:
                chat_template_kwargs = {}
            if 'add_generation_prompt' not in chat_template_kwargs:
                chat_template_kwargs['add_generation_prompt'] = True
            prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], strftime_now=strftime_now, **chat_template_kwargs) for prompt in prompts]
        except Exception as e:
            print(e)
    input_str, input_ids, position_ids, padding_mask = tokenizer(prompts)
    input_ids = mx.array(input_ids)
    B, L = input_ids.shape
    position_ids = mx.array(position_ids)
    total_len = max_new_tokens + L
    roper = Roper(config, total_len)
    causal_mask = create_causal_mask(padding_mask) # for boolean masking
    causal_mask = mx.pad(causal_mask, ((0,0), (0,0), (0,0), (max_new_tokens,0)), 'constant', constant_values=False) # for RollCacher
    cache = [RollCacher(config.dtype, B, config.num_key_value_heads, total_len, config.head_dim) for _ in range(config.num_hidden_layers)] # for RollCacher
    zeropad = mx.ones((B, 1, 1, 1), dtype=mx.bool_) # for boolean masking
    goon = mx.ones((B, 1), dtype=mx.bool_)
    eos_id = config.eos_token_id if isinstance(config.eos_token_id, int) else config.eos_token_id[0] # ad hoc
    carry = (input_ids, position_ids, causal_mask, mx.ones((B, 1), dtype=mx.bool_))
    mx.eval(model, roper, cache, carry)
    def scan_step(carry):
        input_ids, position_ids, causal_mask, goon = carry
        rope = roper(position_ids)
        logits = model(input_ids, causal_mask, rope, cache)
        next_input_ids = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        next_input_ids = mx.where(goon, next_input_ids, eos_id)
        new_mask = mx.concat([causal_mask[:, :, -1:, 1:], zeropad], axis=-1) # for RollCacher
        goon = goon & (next_input_ids != eos_id)
        next_position_ids = position_ids[:, -1:] + 1
        new_carry = (next_input_ids, next_position_ids, new_mask, goon)
        return new_carry, next_input_ids
    if use_jit:
        scan_fn = mx.compile(scan_step, inputs=cache, outputs=cache)
    else:
        scan_fn = scan_step
    eval_every = 30
    if stream:
        print(f'┌{PRETTY_HW} Streaming {PRETTY_HW}┐')
    start_tic = time.perf_counter()
    carry, _output_ids = scan_step(carry)
    output_ids = [_output_ids]
    mx.eval(carry)
    prompt_tic = time.perf_counter()
    for i in range(max_new_tokens-1):
        carry, _output_ids = scan_fn(carry)
        mx.async_eval(carry, cache)
        if i % eval_every == eval_every-1:
            if stream:
                print(tokenizer.decode(mx.concat(output_ids[-eval_every:], axis=1)[-1].tolist()), end='', flush=True)
            if not mx.any(carry[-1]):
                break
        output_ids.append(_output_ids)
    end_tic = time.perf_counter()
    output_ids = mx.concat(output_ids, axis=1).tolist()
    if stream:
        print(tokenizer.decode(output_ids[-1][-(i%eval_every):]), end='', flush=True)
        print(f'\n└{PRETTY_HW*2}───────────┘')
    mx.clear_cache()
    output_str = []
    for i, (i_str, o_ids) in enumerate(zip(input_str, output_ids)):
        o_ids = o_ids[:o_ids.index(eos_id)] if eos_id in o_ids else o_ids
        o_str = tokenizer.decode(o_ids)
        output_str.append(o_str)
        if verbose:
            print(f'┌{PRETTY_HW} Inp {i:05} {PRETTY_HW}┐\n{i_str.strip()}\n└{PRETTY_HW*2}───────────┘\n┌{PRETTY_HW} Out {i:05} {PRETTY_HW}┐\n{o_str.strip()}\n└{PRETTY_HW*2}───────────┘')
    _ = measure_performance(start_tic, prompt_tic, end_tic, B, L, max_new_tokens, verbose=verbose)
    return dict(inp_str=input_str, inp_ids=input_ids,out_str=output_str, out_ids=output_ids)
# }}} === INFER ===
# {{{ === DORA ===
class DoRALinear(nn.Module):
    @staticmethod
    def from_linear(linear, r, alpha, scale, dropout):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = DoRALinear(input_dims=input_dims, output_dims=output_dims, r=r, alpha=alpha, scale=scale, dropout=dropout)
        lora_lin.linear = linear
        return lora_lin

    def __init__(self, input_dims, output_dims, r, alpha, scale, dropout, bias=False):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale * (alpha / r)
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(low=-scale, high=scale, shape=(input_dims, r))
        self.lora_b = mx.zeros(shape=(r, output_dims))
        self.m = mx.linalg.norm(self._dequantized_weight(), axis=1).astype(mx.float32)

    def _dequantized_weight(self):
        weight = self.linear.weight
        if isinstance(self.linear, nn.QuantizedLinear):
            weight = mx.dequantize(weight, self.linear.scales, self.linear.biases, self.linear.group_size, self.linear.bits)
        return weight

    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        z = y + (self.scale * z)
        adapted = self._dequantized_weight() + (self.scale * self.lora_b.T) @ self.lora_a.T
        denom = mx.stop_gradient(mx.linalg.norm(adapted, axis=1))
        z = (self.m / denom) * z
        return z.astype(x.dtype)

def linear_to_lora_layers(model, lora_layers, lora_targets, lora_rank, lora_scale, lora_dropout):
    _model = model.model
    from mlx.utils import tree_unflatten
    if lora_layers == 'all':
        lora_layers = _model.layers
    elif isinstance(lora_layers, int):
        lora_layers = _model.layers[-lora_layers:]
    elif isinstance(lora_layers, list):
        lora_layers = [_model.layers[i] for i in lora_layers]
    else:
        raise ValueError("Invalid type for lora_layers. Expected int (number of layers) or list (layer indices or names).")
    def to_lora(layer):
        return DoRALinear.from_linear(layer, r=lora_rank, alpha=lora_rank, scale=lora_scale, dropout=lora_dropout)
    for l in lora_layers:
        lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k in lora_targets]
        l.update_modules(tree_unflatten(lora_layers))
# }}} === DORA ===
# {{{ === TRAIN ===
LORA_CFG = dict(layers='all', targets=['self_attn.o_proj'], rank=32, scale=0.1, dropout=0.0,
                thaws=['norm'], wt_from=None, wt_to='saved_lora.safetensors',
)

def train(ds_id, model, tokenizer, config, lora_cfg=None, n_epochs=2, lr=1e-5, bs=1):
    def loss_fn(model, X, causal_mask, rope, cache, y):
        logits = model(X, causal_mask, rope, cache)
        return nn.losses.cross_entropy(logits, y, reduction='mean') # [] for bs>1 need masking padding&query

    # {{
    if lora_cfg is None:
        lora_cfg = {}
    if 'wt_to' not in lora_cfg:
        lora_cfg = lora_cfg|dict(wt_to=strftime_now("%Y%m%d_%H%M%S.safetensors"))
    lora_cfg = LORA_CFG|lora_cfg
    model.freeze()
    linear_to_lora_layers(model, lora_layers=lora_cfg['layers'], lora_targets=lora_cfg['targets'], lora_rank=lora_cfg['rank'], lora_scale=lora_cfg['scale'], lora_dropout=lora_cfg['dropout'])
    if lora_cfg['wt_from'] and os.path.exists(lora_cfg['wt_from']):
        model.load_weights(lora_cfg['wt_from'], strict=False)
    model.apply_to_modules(lambda k, v: v.unfreeze() if any(k.endswith(t) for t in lora_cfg['thaws']) else None)
    mx.eval(model)
    # }}
    model.train()
    from datasets import load_dataset
    ds = load_dataset(ds_id, split="train").to_list()
    ds = ds[:100] # debug
    n_steps = n_epochs * len(ds) # [] bs
    import mlx.optimizers as optim
    lr_schedule = optim.cosine_decay(lr, n_steps, 1e-5)
    optimizer = optim.Adam(learning_rate=lr_schedule)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    roper = Roper(config)
    mx.eval(roper, model, optimizer)
    cache = [lambda x,y: (x,y)]*config.num_hidden_layers
    for epoch in range(n_epochs): # [] shuffle
        tic_train = time.perf_counter()
        model.train()
        total_loss = num_batches = 0
        for row in ds: # [] allows only bs=1 d/t masking
            prompt = f"<|im_start|>user\n{row['description'].strip()}\n<|im_end|>\n<|im_start|>assistant\n{row['value'].strip()}\n<|im_end|>"
            input_ids = tokenizer.encode(prompt) # [] mask query
            input_ids = input_ids[:128] # [] ad hoc for crash
            input_ids += [config.eos_token_id]
            total_len = len(input_ids) - 1
            rope = roper(mx.arange(total_len)[None,:])
            causal_mask = create_causal_mask(mx.ones((1, total_len), dtype=mx.bool_))
            X = mx.array([input_ids[:-1]])
            y = mx.array([input_ids[1:]])
            loss, grads = loss_and_grad_fn(model, X, causal_mask, rope, cache, y)
            optimizer.update(model, grads)
            mx.eval(loss, model, optimizer)
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / len(ds)
        elp_train = time.perf_counter() - tic_train
        print(f'{epoch=:5d} {avg_loss=:8.2f} {elp_train=:8.2f}')
        # {{
        model.eval()
        _dict_eval = infer('medium red circle\n', model, tokenizer, config, max_new_tokens=20, stream=False, verbose=False)
        print('└ test output:', _dict_eval['out_str'])
        # }}

    from mlx.utils import tree_flatten
    metadata = lora_cfg|dict(wt_from=lora_cfg['wt_to'], wt_to=None)
    metadata = {str(k): v if isinstance(v, str) else json.dumps(v) for k, v in metadata.items() if v is not None}
    mx.save_safetensors(lora_cfg['wt_to'], dict(tree_flatten(model.trainable_parameters())), metadata=metadata)
    mx.clear_cache()
# }}} === TRAIN ===

import numpy as np




def collapse(model, tokenizer, config):
    collapse_targets = ['self_attn.o_proj']
    k=8
    rank=4
    # ---
    layers = model.model.layers
    weights = []
    for l in layers:
        weights += [m.weight for k, m in l.named_modules() if k in collapse_targets]
    n = len(weights)
    d1, d2 = weights[0].shape
    W_cat = mx.concatenate(weights, axis=1)
    # U, S, Vt = mx.linalg.svd(W_cat.astype(mx.float32), stream=mx.cpu) # mlx one as of v0.30.0 doesn't support gpu nor thin svd for svd..
    U, S, Vt = np.linalg.svd(W_cat.astype(mx.float32), full_matrices=False) # can thin svd but slow af
    Ws = [np.array(_w.astype(mx.float32)) for _w in weights]
    print(f'{U.shape=}')
    print(f'{S.shape=}')
    print(f'{Vt.shape=}')
    k_eff = min(k, U.shape[1])
    U_k = U[:, :k_eff]
    S_k = S[:k_eff]
    Vt_k = Vt[:k_eff, :]
    print(f'{U_k=}')
    print(f'{S_k=}')
    print(f'{Vt_k=}')
    B = U_k * S_k[None,:]
    print(f'{B=}')
    Cs = [Vt_k[:, i*d2:(i+1)*d2] for i in range(n)]
    Rs = []
    for w, c in zip(Ws, Cs):
        Rs.append(w - B@c)
    LoRAs = []
    for R in Rs:
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        # r_eff = min(rank, U.shape[1])
        r_eff = max(rank, U.shape[1]) # sanity check
        if r_eff>0:
            LoRAs.append([U[:, :r_eff]*S[None, :r_eff], Vt[:r_eff, :]])
        else:
            LoRAs.append([0,0])
    w0_recon = B@Cs[0] + LoRAs[0][0]@LoRAs[0][1]
    print(f'{Ws[0]=}')
    print(f'{w0_recon=}')












