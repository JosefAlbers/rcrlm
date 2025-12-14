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

def print_trainable_parameters(model):
    from mlx.utils import tree_flatten
    def get_total_parameters(model):
        leaf_modules = tree_flatten(
            model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
        )

        def nparams(m):
            if hasattr(m, "bits"):
                n = 0 if not hasattr(m, "bias") else m.bias.size
                return n + m.weight.size * 32 // m.bits
            return sum(v.size for _, v in tree_flatten(m.parameters()))

        return sum(nparams(m) for _, m in leaf_modules)
    total_p = get_total_parameters(model) / 1e6
    trainable_p = (
        sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 1e6
    )
    print(
        f"Trainable parameters: {(trainable_p * 100 / total_p):.3f}% "
        f"({trainable_p:.3f}M/{total_p:.3f}M)"
    )

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
    eos_token_id: int
    rms_norm_eps: float = 1e-6
    vocab_size: int = 0
    num_key_value_heads: int = None
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    torch_dtype: str = "float32"
    head_dim: int = None
    attention_bias: bool = True
    mlp_bias: bool = False
    rope_traditional: bool = False
    partial_rotary_factor: float = 1.0
    bos_token_id: Optional[int] = None
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
    mx.eval(model)
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
        print(f"Prompt processing: {prompt_throughput:8.1f} tokens/sec ({tokens_processed:>3d} tokens in {prompt_duration:3.1f}s)")
        print(f"Tokens generation: {generation_throughput:8.1f} tokens/sec ({tokens_generated:>3d} tokens in {generation_duration:3.1f}s)")
        print(f'└{PRETTY_HW*2}───────────┘')
    return metrics
# }}} === PREP ===
# {{{ === I_UTILS ===
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
        self.dtype=dtype

    def __call__(self, k, v):
        # self.k = self_k = mx.concatenate([self.k, k], axis=2)[:, :, -self.max_len:, :]
        # self.v = self_v = mx.concatenate([self.v, v], axis=2)[:, :, -self.max_len:, :]
        # self.k = self_k = mx.concatenate([self.k[:, :, k.shape[2]:, :], k], axis=2)
        # self.v = self_v = mx.concatenate([self.v[:, :, v.shape[2]:, :], v], axis=2)
        self.k, self.v = self_k, self_v = update_cache(self.max_len, self.k, self.v, k, v)
        return self_k, self_v

    def rollback(self, len_back):
        self.k = mx.roll(self.k, shift=len_back, axis=2)
        self.v = mx.roll(self.v, shift=len_back, axis=2)

class CatCacher(nn.Module):
    def __init__(self, dtype, batch_size, num_heads, max_len, head_dim, k=None, v=None):
        super().__init__()
        self.k = mx.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype) if k is None else k
        self.v = mx.zeros((batch_size, num_heads, 0, head_dim), dtype=dtype) if v is None else v

    def __call__(self, k, v):
        self.k = mx.concat([self.k, k], axis=2)
        self.v = mx.concat([self.v, v], axis=2)
        return self.k, self.v

class RetCacher(nn.Module):
    def __init__(self, dtype, batch_size, num_heads, max_len, head_dim, k=None, v=None):
        super().__init__()
        self.s = mx.zeros((batch_size, num_heads, head_dim, head_dim), dtype=dtype)
        self.n = 0

    def get_sn(self):
        return self.s, self.n

    def set_sn(self, s, n):
        self.s = s
        self.n = n

@mx.compile
def get_rope(positions, freq, su_scale):
    angles = positions[:, None, :, None] * freq
    return mx.cos(angles) * su_scale, mx.sin(angles) * su_scale

class Roper(nn.Module):
    def __init__(self, config, su_len=None):
        super().__init__()
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
        # positions = positions[:, None, :, None]
        # angles = positions * self.freq
        # cos = mx.cos(angles) * self.su_scale
        # sin = mx.sin(angles) * self.su_scale
        cos, sin = get_rope(positions, self.freq, self.su_scale)
        # return cos.astype(self.dtype), sin.astype(self.dtype)
        return mx.stop_gradient(cos.astype(self.dtype)), mx.stop_gradient(sin.astype(self.dtype))

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

@mx.compile
def apply_rope(q, k, cos, sin, rot_dims=None, traditional=False):
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
        q_rotated = mx.stack([(q_even * cos - q_odd * sin), (q_even * sin + q_odd * cos)], axis=-1).reshape(q_rot.shape)
        k_rotated = mx.stack([(k_even * cos - k_odd * sin), (k_even * sin + k_odd * cos)], axis=-1).reshape(k_rot.shape)
    else:
        q_split = q_rot.reshape(*q.shape[:-1], 2, -1)
        k_split = k_rot.reshape(*k.shape[:-1], 2, -1)
        q_rotated = mx.concatenate([
            q_split[..., 0, :] * cos - q_split[..., 1, :] * sin,
            q_split[..., 1, :] * cos + q_split[..., 0, :] * sin,
        ], axis=-1)
        k_rotated = mx.concatenate([
            k_split[..., 0, :] * cos - k_split[..., 1, :] * sin,
            k_split[..., 1, :] * cos + k_split[..., 0, :] * sin,
        ], axis=-1)
    if rot_dims is None:
        return q_rotated.astype(q.dtype), k_rotated.astype(k.dtype)
    else:
        q_out = mx.concatenate([q_rotated.astype(q.dtype), q_pass], axis=-1)
        k_out = mx.concatenate([k_rotated.astype(k.dtype), k_pass], axis=-1)
        return q_out, k_out

def create_rope_applier(rot_dims=None, traditional=False):
    def _apply_rope_None_True(q, k, cos, sin, rot_dims):
        q_rot, k_rot = q, k
        q_even = q_rot[..., 0::2]
        q_odd  = q_rot[..., 1::2]
        k_even = k_rot[..., 0::2]
        k_odd  = k_rot[..., 1::2]
        q_rotated = mx.stack([(q_even * cos - q_odd * sin), (q_even * sin + q_odd * cos)], axis=-1).reshape(q_rot.shape)
        k_rotated = mx.stack([(k_even * cos - k_odd * sin), (k_even * sin + k_odd * cos)], axis=-1).reshape(k_rot.shape)
        return q_rotated, k_rotated
    def _apply_rope_dim_True(q, k, cos, sin, rot_dims):
        q_rot, q_pass = q[..., :rot_dims], q[..., rot_dims:]
        k_rot, k_pass = k[..., :rot_dims], k[..., rot_dims:]
        q_even = q_rot[..., 0::2]
        q_odd  = q_rot[..., 1::2]
        k_even = k_rot[..., 0::2]
        k_odd  = k_rot[..., 1::2]
        q_rotated = mx.stack([(q_even * cos - q_odd * sin), (q_even * sin + q_odd * cos)], axis=-1).reshape(q_rot.shape)
        k_rotated = mx.stack([(k_even * cos - k_odd * sin), (k_even * sin + k_odd * cos)], axis=-1).reshape(k_rot.shape)
        q_out = mx.concatenate([q_rotated.astype(q.dtype), q_pass], axis=-1)
        k_out = mx.concatenate([k_rotated.astype(k.dtype), k_pass], axis=-1)
        return q_out, k_out
    def _apply_rope_dim_False(q, k, cos, sin, rot_dims):
        q_rot, q_pass = q[..., :rot_dims], q[..., rot_dims:]
        k_rot, k_pass = k[..., :rot_dims], k[..., rot_dims:]
        q_split = q_rot.reshape(*q.shape[:-1], 2, -1)
        k_split = k_rot.reshape(*k.shape[:-1], 2, -1)
        q_rotated = mx.concatenate([
            q_split[..., 0, :] * cos - q_split[..., 1, :] * sin,
            q_split[..., 1, :] * cos + q_split[..., 0, :] * sin,
        ], axis=-1)
        k_rotated = mx.concatenate([
            k_split[..., 0, :] * cos - k_split[..., 1, :] * sin,
            k_split[..., 1, :] * cos + k_split[..., 0, :] * sin,
        ], axis=-1)
        q_out = mx.concatenate([q_rotated.astype(q.dtype), q_pass], axis=-1)
        k_out = mx.concatenate([k_rotated.astype(k.dtype), k_pass], axis=-1)
        return q_out, k_out
    def _apply_rope_None_False(q, k, cos, sin, rot_dims):
        q_rot, k_rot = q, k
        q_split = q_rot.reshape(*q.shape[:-1], 2, -1)
        k_split = k_rot.reshape(*k.shape[:-1], 2, -1)
        q_rotated = mx.concatenate([
            q_split[..., 0, :] * cos - q_split[..., 1, :] * sin,
            q_split[..., 1, :] * cos + q_split[..., 0, :] * sin,
        ], axis=-1)
        k_rotated = mx.concatenate([
            k_split[..., 0, :] * cos - k_split[..., 1, :] * sin,
            k_split[..., 1, :] * cos + k_split[..., 0, :] * sin,
        ], axis=-1)
        return q_rotated, k_rotated
    if traditional:
        if rot_dims is None:
            return _apply_rope_None_True
        else:
            return _apply_rope_dim_True
    if not traditional:
        if rot_dims is None:
            return _apply_rope_None_False
        else:
            return _apply_rope_dim_False

def create_causal_mask(padding_mask):
    padding_mask = mx.array(padding_mask, dtype=mx.bool_)
    seq_length = padding_mask.shape[1]
    causal_matrix = mx.tril(mx.ones((seq_length, seq_length), dtype=mx.bool_))
    causal_mask = causal_matrix & padding_mask[:, None, :]
    # causal_mask = mx.where(causal_matrix & padding_mask[:, None, :], 0.0, -1e9) # additive
    return causal_mask[:, None, :, :]
# }}} === I_UTILS ===
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
    limit_thinking=False,
):
    if limit_thinking is True:
        end_think_id = tokenizer.encode('</think>')
        end_think_id = end_think_id[-1]
    else:
        end_think_id = None

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
        linear_to_lora_layers(model, lora_layers=lora_cfg['layers'], lora_targets=lora_cfg['targets'], lora_rank=lora_cfg['rank'], lora_scale=lora_cfg['scale'], lora_dropout=lora_cfg['dropout'], lora_quantize=lora_cfg['quantize'],lora_class=eval(lora_cfg['kind']))
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
            input_str, input_ids, position_ids, padding_mask = tokenizer(prompts)
        except Exception as e:
            print(e)
    else:
        tokens = [[config.bos_token_id]+tokenizer.encode(_p) for _p in prompts]
        input_ids, position_ids, padding_mask = tokenizer.pad_token_sequences(tokens, 1, 1)
        input_str = [tokenizer.decode(_t) for _t in tokens]
    input_ids = mx.array(input_ids)
    B, L = input_ids.shape
    position_ids = mx.array(position_ids)
    total_len = max_new_tokens + L
    roper = Roper(config, total_len)
    causal_mask = create_causal_mask(padding_mask)
    causal_mask = mx.pad(causal_mask, ((0,0), (0,0), (0,0), (max_new_tokens,0)), 'constant', constant_values=False)
    cache = []
    for _layer_elm in model.model.layers:
        _n_rcr = getattr(_layer_elm, 'n_rcr', None)
        if _n_rcr:
            _Cacher, c_heads = (RollCacher, config.num_key_value_heads) if getattr(_layer_elm.layer.self_attn, 'to_retention', None) else (RetCacher, config.num_attention_heads)
            _layer_cch = [_Cacher(config.dtype, B, c_heads, total_len, config.head_dim) for _ in range(_n_rcr)]
        else:
            _Cacher, c_heads = (RollCacher, config.num_key_value_heads) if getattr(_layer_elm.self_attn, 'to_retention', None) else (RetCacher, config.num_attention_heads)
            _layer_cch = _Cacher(config.dtype, B, c_heads, total_len, config.head_dim)
        cache.append(_layer_cch)
    zeropad = mx.ones((B, 1, 1, 1), dtype=mx.bool_)
    goon = mx.ones((B, 1), dtype=mx.bool_)
    eos_ids = [config.eos_token_id] if isinstance(config.eos_token_id, int) else config.eos_token_id # ad hoc
    eos_ids = mx.array(eos_ids)
    carry = (input_ids, position_ids, causal_mask, mx.ones((B, 1), dtype=mx.bool_))
    mx.eval(model, roper, cache, carry, zeropad, eos_ids)
    def scan_step(prev_carry):
        prev_input_ids, prev_position_ids, prev_mask, prev_goon = prev_carry
        rope = roper(prev_position_ids)
        logits = model(prev_input_ids, prev_mask, rope, cache)
        next_input_ids = mx.where(prev_goon, mx.argmax(logits[:, -1, :], axis=-1, keepdims=True), eos_ids[0])
        new_mask = mx.concat([prev_mask[:, :, -1:, 1:], zeropad], axis=-1)
        is_eos = mx.any(next_input_ids == eos_ids, axis=-1, keepdims=True)
        new_goon = prev_goon & (~is_eos)
        next_position_ids = prev_position_ids[:, -1:] + 1
        new_carry = (next_input_ids, next_position_ids, new_mask, new_goon)
        return new_carry#, next_input_ids
    if use_jit:
        scan_fn = mx.compile(scan_step, inputs=cache, outputs=cache)
    else:
        scan_fn = scan_step
    if stream:
        print(f'┌{PRETTY_HW} Streaming {PRETTY_HW}┐')
    eval_every = 64
    nbuf=eval_every//2
    ntok=1
    start_tic = time.perf_counter()
    # carry, _oid = scan_step(carry)
    carry = scan_step(carry)
    mx.eval(carry)
    output_ids = [carry[0]]
    prompt_tic = time.perf_counter()
    for i in range(max_new_tokens-1):
        carry = scan_fn(carry)
        # carry, _oid = scan_fn(carry)
        # {{ limit thinking
        if end_think_id is not None and i == int(0.75*max_new_tokens):
            cat_oids = mx.concat(output_ids, axis=1)
            is_done = mx.any(cat_oids == end_think_id, axis=-1, keepdims=True)
            # print(f'{tokenizer.decode(cat_oids[-1].tolist())=}')
            # print(f'{is_done=}')
            _carry_0 = carry[0]
            _carry_0 = mx.where(is_done, _carry_0, end_think_id)
            # print(f'{_carry_0=}')
            carry = tuple([_carry_0, *carry[1:]])
            # print(f'{carry=}')
        # }} limit thinking
        output_ids.append(carry[0])
        ntok+=1
        if i % eval_every == eval_every-1:
            if stream:
                print(tokenizer.decode(mx.concat(output_ids[-ntok-nbuf:-nbuf], axis=1)[-1].tolist()), end='', flush=True)
                ntok=0
            else:
                mx.async_eval(carry)
            if not mx.any(carry[-1]):
                break
    end_tic = time.perf_counter()
    output_ids = mx.concat(output_ids, axis=1).tolist()
    if stream:
        print(tokenizer.decode(output_ids[-1][-ntok-nbuf:]), end='', flush=True)
        print(f'\n└{PRETTY_HW*2}───────────┘')
    mx.clear_cache()
    output_str = []
    for i, (i_str, o_ids) in enumerate(zip(input_str, output_ids)):
        o_ids = o_ids[:o_ids.index(eos_ids[0])+1] if eos_ids[0] in o_ids else o_ids # [] ad hoc, should instead slice till any of the eos_id"s"
        o_str = tokenizer.decode(o_ids)
        output_str.append(o_str)
        if verbose:
            print(f'┌{PRETTY_HW} Inp {i:05} {PRETTY_HW}┐\n{i_str.strip()}\n└{PRETTY_HW*2}───────────┘\n┌{PRETTY_HW} Out {i:05} {PRETTY_HW}┐\n{o_str.strip()}\n└{PRETTY_HW*2}───────────┘')
    if verbose:
        _ = measure_performance(start_tic, prompt_tic, end_tic, B, L, max_new_tokens, verbose=verbose)
    return dict(inp_str=input_str, inp_ids=input_ids, out_str=output_str, out_ids=output_ids)
# }}} === INFER ===
# {{{ === DoRA ===
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
        init_scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(low=-init_scale, high=init_scale, shape=(input_dims, r))
        self.lora_b = mx.zeros(shape=(r, output_dims))
        self.m = mx.linalg.norm(self._dequantized_weight().astype(mx.float32), axis=1)

    def _dequantized_weight(self):
        weight = self.linear.weight
        if isinstance(self.linear, nn.QuantizedLinear):
            weight = mx.dequantize(weight, self.linear.scales, self.linear.biases, self.linear.group_size, self.linear.bits)
        return weight

    def __call__(self, x):
        bias = self.linear.bias if "bias" in self.linear else 0
        y = self.linear(x)
        y = y - bias
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        z = y + (self.scale * z)
        w = self._dequantized_weight()
        adapted = w + (self.scale * self.lora_b.T) @ self.lora_a.T
        denom = mx.linalg.norm(adapted, axis=1)
        z = (self.m / denom) * z
        z = z + bias
        return z#.astype(x.dtype)

def linear_to_lora_layers(model, lora_layers, lora_targets, lora_rank, lora_scale, lora_dropout, lora_quantize, lora_class):
    if lora_quantize:
        nn.quantize(model, 32, 8, class_predicate=lambda p, m: (isinstance(m, nn.Linear) or isinstance(m, nn.Embedding)) and not p.endswith('_new'))
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
        return lora_class.from_linear(layer, r=lora_rank, alpha=lora_rank, scale=lora_scale, dropout=lora_dropout)
    for l in lora_layers:
        loralized = [(k, to_lora(m)) for k, m in l.named_modules() if k in lora_targets]
        l.update_modules(tree_unflatten(loralized))
# }}} === DoRA ===
# {{{ === LoRA-XS ===
class LoRAXSLinear(nn.Module):
    @staticmethod
    def from_linear(linear, r, dropout=0.0, verbose=False, **kwargs):
        if isinstance(linear, nn.QuantizedLinear):
            w = mx.dequantize(linear.weight, linear.scales, linear.biases, linear.group_size, linear.bits)
        else:
            w = linear.weight
        import numpy as np
        w_np = np.array(w.astype(mx.float32)) # shape (out, in)
        U, S, Vt = np.linalg.svd(w_np, full_matrices=False)
        U_r = mx.array(U[:, :r])
        Vt_r = mx.array(Vt[:r, :])
        if verbose:
            print(f"   -> Init LoRA-XS: {w.shape} -> r={r} (Params: {r*r})")
        return LoRAXSLinear(linear, U_r, Vt_r, r, dropout)

    def __init__(self, linear, U, Vt, r, dropout=0.0):
        super().__init__()
        self.linear = linear
        self.dropout = nn.Dropout(p=dropout)
        self.U = mx.stop_gradient(U) 
        self.Vt = mx.stop_gradient(Vt)
        self.R = mx.zeros((r, r))

    def __call__(self, x):
        y = self.linear(x)
        h = self.dropout(x) @ self.Vt.T
        h = h @ self.R.T
        h = h @ self.U.T
        return y + h
# }}} === LoRA-XS ===
# {{{ === DRINK ===
class LowRankLinear(nn.Module):
    @staticmethod
    def from_linear(linear, U, S, Vt, rank):
        U_k = U[:, :rank]
        S_k = S[:rank]
        Vt_k = Vt[:rank, :]
        # {{ v0
        # A_np = U_k * S_k[None, :]
        # B_np = Vt_k
        # }} v0
        # {{ v1
        import numpy as np
        sqrt_S = np.sqrt(S_k)
        A_np = U_k * sqrt_S[None, :]  
        B_np = sqrt_S[:, None] * Vt_k 
        # }} v1
        out_dims, in_dims = linear.weight.shape
        bias = "bias" in linear
        lrl = LowRankLinear(in_dims, out_dims, rank, bias)
        lrl.lora_b.weight = mx.array(B_np, dtype=linear.weight.dtype) 
        lrl.lora_a.weight = mx.array(A_np, dtype=linear.weight.dtype)
        if bias:
            lrl.lora_a.bias = linear.bias
        return lrl

    def __init__(self, in_dims, out_dims, rank, bias=False):
        super().__init__()
        self.lora_b = nn.Linear(in_dims, rank, bias=False)
        self.lora_a = nn.Linear(rank, out_dims, bias=bias)

    def __call__(self, x):
        return self.lora_a(self.lora_b(x))

def drink(model, compression_ratio=0.2, dry_run=False):
    import numpy as np
    targets = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    layers = model.model.layers
    stats = []
    pbar = tqdm(total=len(layers) * len(targets), desc="Analyzing Effective Ranks")
    total_params_original = 0
    for i, layer in enumerate(layers):
        for name, module in layer.named_modules():
            if any(t in name for t in targets) and isinstance(module, nn.Linear):
                if isinstance(module, nn.QuantizedLinear):
                    W = mx.dequantize(module.weight, module.scales, module.biases, module.group_size, module.bits)
                else:
                    W = module.weight
                W_np = np.array(W.astype(mx.float32))
                out_d, in_d = W_np.shape
                total_params_original += (out_d * in_d)
                try:
                    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
                except np.linalg.LinAlgError:
                    print(f"   ! SVD failed for layer {i} {name}, skipping.")
                    continue
                p = S / np.sum(S)
                entropy = -np.sum(p * np.log(p + 1e-12))
                eff_rank = np.exp(entropy)
                stats.append({
                    'layer_idx': i,
                    'name': name,
                    'module': module,
                    'U': U,
                    'S': S,
                    'Vt': Vt,
                    'eff_rank': eff_rank,
                    'shape': (out_d, in_d),
                    'full_path': f"{i}.{name}"
                })
                pbar.update(1)
    pbar.close()
    if not stats:
        print("No eligible layers found.")
        return model
    target_params = total_params_original * (1.0 - compression_ratio)
    denom = sum((s['shape'][0] + s['shape'][1]) * s['eff_rank'] for s in stats)
    alpha = target_params / denom
    for s in stats:
        k_target = int(alpha * s['eff_rank'])
        max_rank = min(s['shape'])
        k_target = max(32, min(k_target, max_rank))
        if "v_proj" in s['name']:
            k_target = int(k_target * 1.1)
        elif "q_proj" in s['name'] or "k_proj" in s['name']:
            k_target = int(k_target * 0.9)
        k_target = min(k_target, max_rank)
        current_params = s['shape'][0] * s['shape'][1]
        new_params = (s['shape'][0] + s['shape'][1]) * k_target
        if new_params >= current_params:
            if dry_run: print(f"   Skipping {s['full_path']}: Compress ({new_params}) > Org ({current_params})")
            continue
        if dry_run:
            print(f"   {s['full_path']:<40} | EffR: {s['eff_rank']:6.1f} | k: {max_rank} -> {k_target}")
        else:
            lrl = LowRankLinear.from_linear(s['module'], s['U'], s['S'], s['Vt'], k_target)
            layer = layers[s['layer_idx']]
            parts = s['name'].split('.')
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], lrl)
    if dry_run:
        print("[Dry Run] No changes applied.")
    else:
        mx.eval(model)
    model.teacher_to_student = [(_l, _l) for _l in range(len(model.model.layers))]
    return model
# }}} === DRINK ===
# {{{ === TRAIN ===
def train(ds_id, model, tokenizer, config, n_epochs=2, lr=1e-4, bs=2, sl=1024, lora_cfg=None):
    LORA_CFG = dict(layers='all', targets=['self_attn.o_proj'], rank=32, scale=0.1, dropout=0.0,
                    thaws=['norm'], wt_from=None, wt_to='saved_lora.safetensors',
                    # kind='LoRAXSLinear',
                    kind='DoRALinear',
                    quantize=True,
    )
    if lora_cfg is None:
        lora_cfg = {}
    if 'wt_to' not in lora_cfg:
        lora_cfg = lora_cfg|dict(wt_to=strftime_now("%Y%m%d_%H%M%S.safetensors"))
    lora_cfg = LORA_CFG|lora_cfg
    model.freeze()
    linear_to_lora_layers(model, lora_layers=lora_cfg['layers'], lora_targets=lora_cfg['targets'], lora_rank=lora_cfg['rank'], lora_scale=lora_cfg['scale'], lora_dropout=lora_cfg['dropout'], lora_quantize=lora_cfg['quantize'], lora_class=eval(lora_cfg['kind']))
    if lora_cfg['wt_from'] and os.path.exists(lora_cfg['wt_from']):
        model.load_weights(lora_cfg['wt_from'], strict=False)
    model.apply_to_modules(lambda k, v: v.unfreeze() if k.endswith(tuple(lora_cfg['thaws'])) else None)
    from datasets import load_dataset
    ds = load_dataset(ds_id, split="train").to_list()
    ds = ds[:100]
    ds = [dict(str_i=_r['description'], str_o=_r['value']) for _r in ds]
    model = _train(ds, model, tokenizer, config, n_epochs=n_epochs, lr=lr, bs=bs, sl=sl)
    from mlx.utils import tree_flatten
    metadata = lora_cfg|dict(wt_from=lora_cfg['wt_to'], wt_to=None)
    metadata = {str(k): v if isinstance(v, str) else json.dumps(v) for k, v in metadata.items() if v is not None}
    mx.save_safetensors(lora_cfg['wt_to'], dict(tree_flatten(model.trainable_parameters())), metadata=metadata)
    mx.clear_cache()
    return model

class AutoBalancingStudent(nn.Module): # https://arxiv.org/pdf/1705.07115 kinda but not rly
    def __init__(self, student):
        super().__init__()
        self.student = student
        self.log_var_logits=mx.array(-0.0821954, dtype=mx.float32)
        if hasattr(student, "teacher_to_student"):
            self.teacher_to_student = student.teacher_to_student
            self.log_var_hiddens = [mx.array(0.494292, dtype=mx.float32) for _ in range(len(student.teacher_to_student))]
        else:
            self.log_var_hiddens = None
    def __call__(self, *args, **kwargs):
        return self.student(*args, **kwargs)

def _train(ds, model, tokenizer, config, n_epochs=2, lr=1e-4, bs=2, sl=1024, 
           teacher=None, teacher_to_student=None, 
           hidden_wt=2.0):
    cache = [lambda x,y: (x,y)] * config.num_hidden_layers
    t_hiddens = [i[0] for i in teacher_to_student] if teacher_to_student else []
    s_hiddens = [i[1] for i in teacher_to_student] if teacher_to_student else []
    l_hiddens = len(t_hiddens)

    if teacher is not None:
        teacher.freeze()
        model = AutoBalancingStudent(model)

    def loss_fn(model, X, causal_mask, rope, y, label_mask):
        logits, student_captures = model(X, causal_mask, rope, cache, hiddens=s_hiddens)

        if teacher is not None:
            teacher_logits, teacher_captures = teacher(X, causal_mask, rope, cache, hiddens=t_hiddens)
            log_p_student = nn.log_softmax(logits.astype(mx.float32), axis=-1)
            log_p_teacher = mx.stop_gradient(nn.log_softmax(teacher_logits.astype(mx.float32), axis=-1))
            # # {{ --- forward kld ---
            # p_teacher = mx.stop_gradient(mx.exp(log_p_teacher))
            # kld_loss = mx.sum(p_teacher * (log_p_teacher - log_p_student), axis=-1)
            # # }} --- forward kld ---
            # # {{ --- reverse kld ---
            # p_student = mx.exp(log_p_student)
            # kld_loss = mx.sum(p_student * (log_p_student - log_p_teacher), axis=-1)
            # # }} --- reverse kld ---
            # {{ --- both kld ---
            p_teacher = mx.stop_gradient(mx.exp(log_p_teacher))
            fkld_loss = mx.sum(p_teacher * (log_p_teacher - log_p_student), axis=-1)
            p_student = mx.exp(log_p_student)
            rkld_loss = mx.sum(p_student * (log_p_student - log_p_teacher), axis=-1)
            kld_loss = 0.5*fkld_loss + 0.5*rkld_loss
            # }} --- both kld ---
            # # {{ --- jsd ---
            # p_teacher = mx.stop_gradient(mx.exp(log_p_teacher))
            # p_student = mx.exp(log_p_student)
            # p_mix = 0.5 * p_teacher + 0.5 * p_student
            # log_p_mix = mx.log(p_mix + 1e-10)
            # kld_loss = 0.5 * mx.sum(p_teacher * (log_p_teacher - log_p_mix), axis=-1) + \
            #            0.5 * mx.sum(p_student * (log_p_student - log_p_mix), axis=-1)
            # # }} --- jsd ---
            prec_logits = mx.exp(-model.log_var_logits)
            loss_logits = prec_logits * kld_loss + 0.5 * model.log_var_logits
            hidden_loss_accum = 0.0
            for idx_cap, (s_cap, t_cap) in enumerate(zip(student_captures, teacher_captures)):
                t_cap_detached = mx.stop_gradient(t_cap.astype(mx.float32))
                # {{ --- mse ---
                mse_raw = (s_cap - t_cap_detached) ** 2
                loss_per_token = mx.mean(mse_raw, axis=-1) # Mean over hidden dim
                # }} --- mse ---
                # # {{ --- cos ---
                # dot_prod = mx.sum(s_cap * t_cap_detached, axis=-1)
                # norm_s = mx.sqrt(mx.sum(mx.square(s_cap), axis=-1) + 1e-9)
                # norm_t = mx.sqrt(mx.sum(mx.square(t_cap_detached), axis=-1) + 1e-9)
                # cos_sim = dot_prod / (norm_s * norm_t)
                # loss_per_token = 1.0 - cos_sim
                # # }} --- cos ---
                log_var = model.log_var_hiddens[idx_cap] 
                prec = mx.exp(-log_var)
                layer_loss = prec * loss_per_token + 0.5 * log_var
                hidden_loss_accum = hidden_loss_accum + layer_loss
            return ((loss_logits + hidden_wt*hidden_loss_accum)*label_mask).astype(mx.float32).sum() / label_mask.sum()
        else:
            ce = nn.losses.cross_entropy(logits, y, reduction='none') * label_mask
            return ce.astype(mx.float32).sum() / label_mask.sum()

    mx.eval(model)
    n_steps = n_epochs * len(ds) // bs
    import mlx.optimizers as optim
    lr_schedule = optim.cosine_decay(lr, n_steps, 0.0)
    optimizer = optim.Adam(learning_rate=lr_schedule)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    roper = Roper(config)
    mx.eval(roper, model, optimizer, teacher)
    test_prompt = tokenizer.apply_chat_template([{"role": "user", "content": "medium red circle"}], strftime_now=strftime_now, **{'add_generation_prompt':True, 'enable_thinking':False})
    eos_id = config.eos_token_id
    import random
    for epoch in range(n_epochs):
        tic_train = time.perf_counter()
        model.train()
        total_loss = num_batches = 0
        ds = random.sample(ds, len(ds))
        pbar = tqdm(range(0, len(ds), bs), desc=f"Epoch {epoch+1}/{n_epochs}")
        for i in pbar:
            batch_rows = ds[i:i+bs]
            _Xs = []
            _ys = []
            _lms = []
            _ams = []
            for row in batch_rows:
                str_a = tokenizer.apply_chat_template([{"role": "user", "content": row['str_i'].strip()}, {"role": "assistant", "content": row['str_o'].strip()}], strftime_now=strftime_now, **{'add_generation_prompt':False, 'enable_thinking':False})
                str_a = str_a.strip()
                iid_a = tokenizer.encode(str_a)
                if teacher is None:
                    str_i = tokenizer.apply_chat_template([{"role": "user", "content": row['str_i'].strip()}], strftime_now=strftime_now, **{'add_generation_prompt':True, 'enable_thinking':False})
                    iid_i = tokenizer.encode(str_i)
                    iid_o = iid_a[len(iid_i):]
                    input_ids = iid_i + iid_o
                    label_mask = [0]*len(iid_i) + [1]*len(iid_o)
                else:
                    input_ids = iid_a
                    label_mask = [1]*len(iid_a)
                input_ids = input_ids[:sl]
                label_mask = label_mask[:sl]
                _Xs.append(input_ids[:-1])
                _ys.append(input_ids[1:])
                _lms.append(label_mask[1:])
                _ams.append([True]*(len(label_mask)-1))
            _seq_len = max(len(_m) for _m in _lms)
            X = []
            y = []
            label_mask = []
            attention_mask = []
            for e in range(len(_lms)):
                _pad_len = _seq_len - len(_ams[e])
                X.append(_Xs[e]+[eos_id]*_pad_len)
                y.append(_ys[e]+[eos_id]*_pad_len)
                label_mask.append(_lms[e]+[0]*_pad_len)
                attention_mask.append(_ams[e]+[False]*_pad_len)
            rope = roper(mx.array([list(range(_seq_len))]*len(_ams)))
            causal_mask = create_causal_mask(attention_mask)
            loss, grads = loss_and_grad_fn(
                model, 
                mx.array(X), 
                causal_mask, 
                rope, 
                mx.array(y), 
                mx.array(label_mask),
            )
            optimizer.update(model, grads)
            mx.eval(loss, model, optimizer)
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        avg_loss = total_loss / len(ds)
        elp_train = time.perf_counter() - tic_train
        print(f'{epoch=:5d} {avg_loss=:8.2f} {elp_train=:8.2f}')
        model.eval()
        mx.eval(model)
        _dict_eval = infer(test_prompt, model.student if teacher is not None else model, tokenizer, config, max_new_tokens=20, stream=False, verbose=False, use_chat_template=False)
        print('└ test output:', _dict_eval['out_str'])
        
    if teacher is not None:
        # print(model.log_var_logits, model.log_var_hiddens)
        return model.student
    return model

# }}} === TRAIN ===
# {{{ === SVD ===
def svdlora(model, tokenizer, config, collapse_cfg=None):
    import numpy as np
    COLLAPSE_CFG = dict(k=256, rank=32, targets=['self_attn.v_proj'])
    if collapse_cfg is None:
        collapse_cfg = {}
    collapse_cfg = COLLAPSE_CFG|collapse_cfg
    targets = collapse_cfg['targets']
    k=collapse_cfg['k']
    rank=collapse_cfg['rank']
    # ---
    tic_clp = time.perf_counter()
    layers = model.model.layers
    Ws = []
    for l in layers:
        Ws += [m.weight for k, m in l.named_modules() if k in targets]
    n = len(Ws)
    d1, d2 = Ws[0].shape
    print(f'{d1=}')
    print(f'{d2=}')
    W_cat = mx.concatenate(Ws, axis=1)
    # U, S, Vt = mx.linalg.svd(W_cat.astype(mx.float32), stream=mx.cpu) # mlx svd as of v0.30.0 doesn't support gpu nor thinning; crashes too..
    # mx.eval(U, S, Vt)
    U, S, Vt = np.linalg.svd(W_cat.astype(mx.float32), full_matrices=False) # can thin svd but slow af
    Ws = [np.array(_w.astype(mx.float32)) for _w in Ws]
    k_eff = min(k, d1)
    U_k = U[:, :k_eff]
    S_k = S[:k_eff]
    Vt_k = Vt[:k_eff, :]
    B = U_k * S_k[None,:]
    Cs = [Vt_k[:, i*d2:(i+1)*d2] for i in range(n)]
    Rs = []
    for w, c in zip(Ws, Cs):
        Rs.append(w - B@c)
    LoRAs = []
    for R in Rs:
        # U, S, Vt = mx.linalg.svd(R, stream=mx.cpu)
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        # r_eff = min(rank, d1)
        r_eff = d1 # sanity check
        if r_eff>0:
            LoRAs.append([U[:, :r_eff]*S[None, :r_eff], Vt[:r_eff, :]])
        else:
            LoRAs.append([0,0])
    w0_recon = B@Cs[0] + LoRAs[0][0]@LoRAs[0][1]
    print(f'{Ws[0]=}')
    print(f'{w0_recon=}')
    elp_clp = time.perf_counter() - tic_clp
    print(f'{elp_clp=:.1f}')

def prune(calib_ds_id, model, tokenizer, config, n_samples=64, bs=4, lambda_val=1.0, prune_ratio=0.3, verbose=True):
    class UniQL_MLP_Collector(nn.Module):
        def __init__(self, original_mlp, layer_idx, stats_dict, counts_dict):
            super().__init__()
            self.target = original_mlp 
            self.layer_idx = layer_idx
            self.stats = stats_dict
            self.counts = counts_dict

        def __call__(self, x):
            gate = self.target.gate_proj(x)
            up = self.target.up_proj(x)
            h = nn.silu(gate) * up
            h_flat = h.reshape(-1, h.shape[-1])
            h_f32 = mx.stop_gradient(h_flat).astype(mx.float32)
            c_batch = h_f32.T @ h_f32
            if self.layer_idx not in self.stats:
                self.stats[self.layer_idx] = c_batch
                self.counts[self.layer_idx] = h_flat.shape[0]
            else:
                self.stats[self.layer_idx] = self.stats[self.layer_idx] + c_batch
                self.counts[self.layer_idx] = self.counts[self.layer_idx] + h_flat.shape[0]
            return self.target.down_proj(h)

    stats = {} 
    counts = {}
    original_mlps = {}
    
    for i, layer in enumerate(model.model.layers):
        original_mlps[i] = layer.mlp
        layer.mlp = UniQL_MLP_Collector(layer.mlp, i, stats, counts)

    from datasets import load_dataset
    ds = load_dataset(calib_ds_id, split="test", streaming=True)
    data_iter = iter(ds)
    
    model.eval()
    processed = 0
    pbar = tqdm(total=n_samples, desc="Calibrating MLP")
    cache = [lambda x, y: (x, y)] * len(model.model.layers)

    while processed < n_samples:
        batch_prompts = []
        for _ in range(bs):
            try:
                row = next(data_iter)
                txt = row.get('text', row.get('prompt', row.get('description', ''))) # ad hoc
                if txt: batch_prompts.append(txt)
            except StopIteration:
                break
        
        if not batch_prompts: break

        for txt in batch_prompts:
            if processed >= n_samples: break
            ids = tokenizer.encode(txt)
            if len(ids) == 0: continue
            ids = mx.array([ids])
            B, L = ids.shape
            roper = Roper(config, L)
            rope = roper(mx.arange(L)[None, :])
            mask = create_causal_mask(mx.array([[True]*L]))
            try:
                model(ids, mask, rope, cache)
                mx.eval(stats)
            except Exception as e:
                print(f"Warning: Batch failed {e}")
            processed += 1
            pbar.update(1)
    pbar.close()

    for i, layer in enumerate(model.model.layers):
        layer.mlp = original_mlps[i]

    if not stats:
        print("Error: No stats collected.")
        return model

    for i in range(len(model.model.layers)):
        if i not in stats: continue
        
        C = stats[i] / counts[i]
        D_int = C.shape[0]
        
        reg = C + (lambda_val * mx.eye(D_int, dtype=mx.float32))
        reg_inv = mx.linalg.inv(reg, stream=mx.cpu)
        s = mx.diag(C @ reg_inv)

        perm = mx.argsort(s)[::-1]
        
        mlp = model.model.layers[i].mlp
        
        mlp.gate_proj.weight = mlp.gate_proj.weight[perm]
        mlp.up_proj.weight   = mlp.up_proj.weight[perm]
        mlp.down_proj.weight = mlp.down_proj.weight[:, perm]

        if prune_ratio > 0.0:
            keep = int(D_int * (1.0 - prune_ratio))
            mlp.gate_proj.weight = mlp.gate_proj.weight[:keep]
            mlp.up_proj.weight   = mlp.up_proj.weight[:keep]
            mlp.down_proj.weight = mlp.down_proj.weight[:, :keep]
            if i == 0 and verbose: print(f"   Layer 0 pruned to {keep}/{D_int}")

    mx.eval(model)
    return model

def cascade(ds_id, model, tokenizer, config, teacher, to='distilled.safetensors', collapse_ranges=None):
    if collapse_ranges is None:
        collapse_ranges = [(9,13), (13,17)]
    for range_tuple in collapse_ranges:
        model = collapse(model, collapse_ranges=[range_tuple])
        model = distill(ds_id, model, tokenizer, config, teacher, to=to)
    return model

def dampen(model, lambda_val=0.3, sim_threshold=0.5, ft_check_rank=1024, verbose=True):
    import numpy as np
    from mlx.utils import tree_unflatten

    def to_np(x): 
        return np.array(x.astype(mx.float32))

    replacements = []
    
    for name, module in model.named_modules():
        if isinstance(module, DoRALinear):
            if verbose:
                print(f"[{PRETTY_HW}] Processing {name}...")
            W_0_mx = module._dequantized_weight()
            dtype = W_0_mx.dtype
            delta = (module.scale * module.lora_b.T) @ module.lora_a.T
            W_prime = W_0_mx + delta
            
            norm_W_prime = mx.linalg.norm(W_prime, axis=1, keepdims=True)
            scale_vec = module.m[:, None] / (norm_W_prime) 
            W_ft_mx = scale_vec * W_prime
            
            W_0 = to_np(W_0_mx)
            W_ft = to_np(W_ft_mx)
            
            U_ft, S_ft, Vt_ft = np.linalg.svd(W_ft, full_matrices=False)
            U_0, _, _ = np.linalg.svd(W_0, full_matrices=False)
            U_ref = U_0
            
            k_check = min(ft_check_rank, U_ft.shape[1])
            U_check = U_ft[:, :k_check]
            
            # projections = np.dot(U_check.T, U_ref)
            # similarities = np.sqrt(np.sum(projections**2, axis=1))
            similarity_matrix = np.abs(np.dot(U_check.T, U_0))
            similarities = np.max(similarity_matrix, axis=1)
            
            is_intruder = similarities < sim_threshold
            intruder_indices = np.where(is_intruder)[0]
            
            if len(intruder_indices) > 0:
                # # {{{ --- v0 ---
                # worst_idx_local = np.argmin(similarities[intruder_indices])
                # worst_idx_global = intruder_indices[worst_idx_local]
                # worst_sim = similarities[worst_idx_global]
                # S_new = S_ft.copy()
                # S_new[worst_idx_global] *= lambda_val
                # if verbose:
                #     print(f"   -> Found {len(intruder_indices)} candidates. Dampening rank {worst_idx_global} (Sim: {worst_sim:.4f})")
                # W_new = (U_ft * S_new[None, :]) @ Vt_ft
                # W_final_mx = mx.array(W_new)
                # # }}} --- v0 ---
                # {{ --- v1 ---
                worst_idx_local = np.argmin(similarities[intruder_indices])
                worst_idx_global = intruder_indices[worst_idx_local]
                worst_sim = similarities[worst_idx_global]
                remove_factor = (1.0 - lambda_val)
                intruder_vec = np.outer(U_ft[:, worst_idx_global], Vt_ft[worst_idx_global, :])
                intruder_component = intruder_vec * S_ft[worst_idx_global]
                subtraction_matrix = intruder_component * remove_factor
                if verbose:
                    print(f"   -> Found {len(intruder_indices)} candidates.")
                    print(f"   -> Subtracting {(remove_factor*100):.0f}% of rank {worst_idx_global} (Sim: {worst_sim:.4f})")
                W_final_mx = W_ft_mx - mx.array(subtraction_matrix)
                # }} --- v1 ---
            else:
                if verbose:
                    print(f"   -> No strong intruders found. Keeping exact weights.")
                W_final_mx = W_ft_mx
            out_d, in_d = W_final_mx.shape
            has_bias = hasattr(module.linear, 'bias') and (module.linear.bias is not None)
            new_linear = nn.Linear(in_d, out_d, bias=has_bias)
            new_linear.weight = W_final_mx.astype(dtype)
            if has_bias:
                new_linear.bias = module.linear.bias
            replacements.append((name, new_linear))
    if replacements:
        model.update_modules(tree_unflatten(replacements))
        print(f"[{PRETTY_HW}] Successfully healed and merged {len(replacements)} layers.")
    else:
        print("No DoRALinear layers found.")
    mx.eval(model)
    return model
# }}} === SVD ===
class RecurrentBlock(nn.Module):
    def __init__(self, layer, n_rcr=3):
        super().__init__()
        self.layer = layer
        self.n_rcr = n_rcr

    def __call__(self, x, attention_mask=None, rope=None, cache=None):
        B, L, D = x.shape
        for i in range(self.n_rcr):
        # for i in range(1): # https://openreview.net/forum?id=ngmEcEer8a
            # if getattr(cache, "rollback", None):
            #     cache.rollback(L*(i>0))
            # x = self.layer(x, attention_mask=attention_mask, rope=rope, cache=cache[i])
            c = cache[i] if isinstance(cache, list) else cache
            x = self.layer(x, attention_mask=attention_mask, rope=rope, cache=c)
        return x

def collapse(model, collapse_ranges=None, do_rectify=False, do_quantize=False):
    if collapse_ranges is None:
        collapse_ranges = [(13,17)]
    layers = model.model.layers
    new_layers = []
    teacher_to_student = [] 
    start_to_range = {start: (start, end) for start, end in collapse_ranges}
    indices_to_skip = set()
    for start, end in collapse_ranges:
        for i in range(start + 1, end):
            indices_to_skip.add(i)
    teacher_offset = 0
    for i, layer in enumerate(layers):
        student_idx = len(new_layers)
        if i in start_to_range:
            start, end = start_to_range[i]
            n_rcr = end - start
            teacher_to_student.append((teacher_offset+end - 1, student_idx))
            ref_layer = layer
            if hasattr(ref_layer.self_attn, 'to_retention') and do_rectify:
                ref_layer.self_attn = ref_layer.self_attn.to_retention(model._config)
            rec_block = RecurrentBlock(ref_layer, n_rcr=n_rcr)
            new_layers.append(rec_block)
        elif i in indices_to_skip:
            continue
        else:
            if getattr(layer, 'n_rcr', None):
                teacher_offset += (layer.n_rcr-1)
            new_layers.append(layer)
    model.model.layers = new_layers
    model.teacher_to_student = teacher_to_student
    if do_quantize:
        nn.quantize(model, 32, 8, class_predicate=lambda p, m: (isinstance(m, nn.Linear) or isinstance(m, nn.Embedding)) and not p.endswith('_new'))
    return model

def distill(ds_id, model, tokenizer, config, teacher=None, n_epochs=1, lr=1e-4, bs=1, sl=4096, to='distilled.safetensors', unfreeze_all=False):
    teacher_to_student = getattr(model, "teacher_to_student", None)
    student_indices = [s[1] for s in teacher_to_student] if teacher_to_student else []
    print(f'{teacher_to_student=}')
    from mlx.utils import tree_unflatten
    model.freeze()
    def to_dora_(layer):
        _rank = 64
        return DoRALinear.from_linear(layer, r=_rank, alpha=_rank, scale=1.0, dropout=0.0)
    for layer_idx, l in enumerate(model.model.layers):
        if getattr(l, 'n_rcr', None) and (layer_idx in student_indices):
            print(f'{layer_idx=}')
            loralized = [(k, to_dora_(m)) for k, m in l.named_modules() if k.endswith('proj')]
            l.update_modules(tree_unflatten(loralized))
            l.apply_to_modules(lambda k, v: v.unfreeze() if k.endswith('_new') else None)
    if unfreeze_all:
        for l in model.model.layers:
            l.apply_to_modules(lambda k, v: v.unfreeze() if k.endswith(('lora_a', 'lora_b')) else None)
    print_trainable_parameters(model)
    from datasets import load_dataset
    ds = load_dataset(ds_id, split="test").to_list()
    ds = ds#[:200]
    ds = [dict(str_i=_r['prompt'], str_o=_r['completion']) for _r in ds]
    model = _train(ds, model, tokenizer, config, teacher=teacher, n_epochs=n_epochs, lr=lr, bs=bs, sl=sl, teacher_to_student=teacher_to_student)
    from mlx.utils import tree_flatten
    mx.save_safetensors(to, dict(tree_flatten(model.parameters())), metadata=None) # just to see how small; [] need quant params to load though
    mx.clear_cache()
    return model
