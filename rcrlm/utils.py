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
import numpy as np

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
# {{{ === UTIL ===
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
        cos, sin = get_rope(positions, self.freq, self.su_scale)
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
    return causal_mask[:, None, :, :]
# }}} === UTIL ===
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
        return new_carry
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
    carry = scan_step(carry)
    mx.eval(carry)
    output_ids = [carry[0]]
    prompt_tic = time.perf_counter()
    for i in range(max_new_tokens-1):
        carry = scan_fn(carry)
        if end_think_id is not None and i == int(0.75*max_new_tokens):
            cat_oids = mx.concat(output_ids, axis=1)
            is_done = mx.any(cat_oids == end_think_id, axis=-1, keepdims=True)
            _carry_0 = carry[0]
            _carry_0 = mx.where(is_done, _carry_0, end_think_id)
            carry = tuple([_carry_0, *carry[1:]])
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
# {{{ === MISC ===
def show_tie_1(model, collapse_cfg=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    targets, k_rank = collapse_cfg['targets'], collapse_cfg['rank']
    layers = model.model.layers
    Ws = []
    layer_names = []
    layer_boundaries = [0]
    for i, l in enumerate(layers):
        for name, m in l.named_modules():
            if name in targets:
                if isinstance(m, nn.QuantizedLinear):
                    w = mx.dequantize(m.weight, m.scales, m.biases, m.group_size, m.bits)
                else:
                    w = m.weight
                Ws.append(w)
                layer_names.append(f"L{i}")
                layer_boundaries.append(layer_boundaries[-1] + w.shape[0])
    W_tall = mx.concatenate(Ws, axis=0)
    d_out_layer = Ws[0].shape[0]
    W_np = np.array(W_tall.astype(mx.float32))
    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
    k_calc = min(k_rank, len(S))
    U_k = U[:, :k_calc]
    S_k = S[:k_calc]
    E_matrix = (U_k * S_k[None, :]) ** 2
    heatmap_rows = []
    stats = []
    for idx in range(len(layer_names)):
        start = layer_boundaries[idx]
        end = layer_boundaries[idx+1]
        layer_energy = np.sum(E_matrix[start:end, :], axis=0) # Shape (k,)
        total_e = np.sum(layer_energy)
        pdf = layer_energy / (total_e + 1e-9)
        cdf = np.cumsum(pdf)
        heatmap_rows.append(pdf)
        peak_idx = np.argmax(pdf)
        ranks = np.arange(k_calc)
        mean_idx = np.sum(ranks * pdf)
        p05 = np.argmax(cdf >= 0.05)
        p25 = np.argmax(cdf >= 0.25)
        p50 = np.argmax(cdf >= 0.50) # Median
        p75 = np.argmax(cdf >= 0.75)
        p95 = np.argmax(cdf >= 0.95)
        stats.append({
            'mean': mean_idx,
            'peak': peak_idx,
            'p05': p05,
            'p25': p25,
            'p50': p50,
            'p75': p75,
            'p95': p95
        })
    heatmap_data = np.array(heatmap_rows)
    heatmap_vis = np.sqrt(heatmap_data) 
    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(heatmap_vis, cmap="viridis", yticklabels=layer_names, cbar=False)
    y_coords = np.arange(len(layer_names)) + 0.5
    for i, s in enumerate(stats):
        y = y_coords[i]
        plt.plot([s['p05'], s['p95']], [y, y], color='white', linewidth=1, alpha=0.5)
        plt.plot([s['p25'], s['p75']], [y, y], color='white', linewidth=4, alpha=0.6)
        plt.plot(s['p50'], y, '|', color='cyan', markersize=10, markeredgewidth=2, label='Median' if i==0 else "")
        plt.plot(s['peak'], y, '.', color='yellow', markersize=5, label='Peak' if i==0 else "")
        plt.plot(s['p95'], y, '|', color='red', markersize=10, markeredgewidth=2, label='95% Cumul.' if i==0 else "")
    plt.title(f"Energy Distribution per Layer\n(Target: {targets[0]}, Top {k_calc} Ranks)")
    plt.xlabel("Principal Component (Rank Index)")
    plt.ylabel("Layer Depth")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='white', lw=4, label='IQR (25-75%)'),
        Line2D([0], [0], color='cyan', marker='|', linestyle='None', label='Median (50%)'),
        Line2D([0], [0], color='red', marker='|', linestyle='None', label='95% Cutoff'),
        Line2D([0], [0], color='yellow', marker='o', linestyle='None', label='Peak Mode'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    print(f"\n[{PRETTY_HW}] Recommendation Report")
    max_95 = max(s['p95'] for s in stats)
    min_95 = min(s['p95'] for s in stats)
    print(f"1. Conservative Rank (Max 95%): {max_95} (Safest)")
    print(f"2. Aggressive Rank (Min 95%): {min_95} (Likely degradation)")
    print(f"3. Average Median: {np.mean([s['p50'] for s in stats]):.1f}")

def show_tie_2(model, collapse_cfg=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    targets, check_rank = collapse_cfg['targets'], collapse_cfg['rank']
    layers = model.model.layers
    layer_bases = []
    layer_names = []
    for i, l in enumerate(layers):
        for name, m in l.named_modules():
            if name in targets:
                if isinstance(m, nn.QuantizedLinear):
                    w = mx.dequantize(m.weight, m.scales, m.biases, m.group_size, m.bits)
                else:
                    w = m.weight
                w_np = np.array(w.astype(mx.float32))
                _, _, Vt = np.linalg.svd(w_np, full_matrices=False)
                basis = Vt[:check_rank, :]
                basis = basis / (np.linalg.norm(basis, axis=1, keepdims=True) + 1e-9)
                layer_bases.append(basis)
                layer_names.append(f"L{i}")
    num_layers = len(layer_bases)
    similarity_matrix = np.zeros((num_layers, num_layers))
    for i in range(num_layers):
        for j in range(i, num_layers):
            corr = np.abs(np.dot(layer_bases[i], layer_bases[j].T))
            score_i_j = np.mean(np.max(corr, axis=1))
            score_j_i = np.mean(np.max(corr, axis=0))
            avg_score = (score_i_j + score_j_i) / 2.0
            similarity_matrix[i, j] = avg_score
            similarity_matrix[j, i] = avg_score
    np.fill_diagonal(similarity_matrix, 0.0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap="magma", xticklabels=layer_names, yticklabels=layer_names)
    plt.title(f"Similarity (Target: {targets[0]})")
    plt.savefig('_'.join(targets)+'.png')
    print(f"\n[{PRETTY_HW}] Suggested Bundles (Threshold > 0.6)")
    bundles = []
    current_bundle = [0]
    for i in range(1, num_layers):
        ref_idx = current_bundle[0]
        sim = similarity_matrix[ref_idx, i]
        if sim > 0.6:
            current_bundle.append(i)
        else:
            bundles.append(current_bundle)
            current_bundle = [i]
    bundles.append(current_bundle)
    for b in bundles:
        print(f"   -> Bundle: {b} (Size: {len(b)})")
    return bundles

def show_tie_3(model, collapse_cfg=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    targets = collapse_cfg.get('targets', ['self_attn.v_proj'])
    check_rank = collapse_cfg.get('rank', 128)
    layers = model.model.layers
    layer_grams = []
    layer_names = []
    layer_traces = []
    for i, l in enumerate(layers):
        for name, m in l.named_modules():
            if name in targets:
                if isinstance(m, nn.QuantizedLinear):
                    w = mx.dequantize(m.weight, m.scales, m.biases, m.group_size, m.bits)
                else:
                    w = m.weight
                w_np = np.array(w.astype(mx.float32))
                g = np.dot(w_np.T, w_np)
                layer_grams.append(g)
                layer_traces.append(np.trace(g))
                layer_names.append(f"L{i}")
    num_layers = len(layer_grams)
    loss_matrix = np.zeros((num_layers, num_layers))
    for i in range(num_layers):
        for j in range(i, num_layers):
            if i == j:
                G_combined = layer_grams[i]
                total_energy = layer_traces[i]
            else:
                G_combined = layer_grams[i] + layer_grams[j]
                total_energy = layer_traces[i] + layer_traces[j]
            eigvals = np.linalg.eigvalsh(G_combined) 
            eigvals = np.flip(eigvals)
            if check_rank < len(eigvals):
                residual_energy = np.sum(eigvals[check_rank:])
            else:
                residual_energy = 0.0
            loss = residual_energy / (total_energy + 1e-9)
            loss_matrix[i, j] = loss
            loss_matrix[j, i] = loss
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(loss_matrix, cmap="viridis_r", xticklabels=layer_names, yticklabels=layer_names)
    plt.title(f"Pairwise Frobenius Loss (Basis Sharing Simulation)\nTarget: {targets[0]} | Shared Rank: {check_rank}\n(Yellow/Low is better)")
    print(f"\n[{PRETTY_HW}] Suggested Bundles (Loss Threshold < 0.05)") 
    bundles = []
    current_bundle = [0]
    threshold = 0.05
    for i in range(1, num_layers):
        ref_idx = current_bundle[0]
        loss = loss_matrix[ref_idx, i]
        if loss < threshold:
            current_bundle.append(i)
        else:
            bundles.append(current_bundle)
            current_bundle = [i]
    bundles.append(current_bundle)
    for b in bundles:
        print(f"   -> Bundle: {b} (Size: {len(b)})")
    return bundles

def show_tie_4(model, collapse_cfg=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if collapse_cfg is None: collapse_cfg = {}
    targets = collapse_cfg.get('targets', ['self_attn.v_proj'])
    rank_mode = 'dynamic' 
    fixed_rank = collapse_cfg.get('rank', 128)
    layers = model.model.layers
    Ws = []
    layer_names = []
    for i, l in enumerate(layers):
        for name, m in l.named_modules():
            if name in targets:
                if isinstance(m, nn.QuantizedLinear):
                    w = mx.dequantize(m.weight, m.scales, m.biases, m.group_size, m.bits)
                else:
                    w = m.weight
                Ws.append(np.array(w.astype(mx.float32)))
                layer_names.append(f"L{i}")
    num_layers = len(Ws)
    loss_matrix = np.zeros((num_layers, num_layers))
    rank_matrix = np.zeros((num_layers, num_layers))
    for i in range(num_layers):
        for j in range(i, num_layers):
            if i == j:
                W_comb = Ws[i]
            else:
                W_comb = np.concatenate([Ws[i], Ws[j]], axis=0)
            G = np.dot(W_comb.T, W_comb)
            S_sq = np.linalg.eigvalsh(G)
            S_sq = np.flip(S_sq)
            S_sq = np.maximum(S_sq, 0)
            S = np.sqrt(S_sq)
            if rank_mode == 'dynamic':
                p = S / np.sum(S)
                entropy = -np.sum(p * np.log(p + 1e-12))
                eff_rank_val = np.exp(entropy)
                k = int(eff_rank_val)
            else:
                k = fixed_rank
            rank_matrix[i, j] = k
            rank_matrix[j, i] = k
            total_energy = np.sum(S_sq)
            kept_energy = np.sum(S_sq[:k])
            loss = 1.0 - (kept_energy / total_energy)
            loss_matrix[i, j] = loss
            loss_matrix[j, i] = loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(loss_matrix, cmap="viridis_r", xticklabels=layer_names, yticklabels=layer_names, ax=ax1)
    ax1.set_title(f"Frobenius Loss (Dynamic Rank)\n(Yellow/Low is better)")
    sns.heatmap(rank_matrix, cmap="magma", xticklabels=layer_names, yticklabels=layer_names, ax=ax2)
    ax2.set_title(f"Auto-Selected Effective Rank\n(Lighter = Higher Rank required)")
    print(f"\n[{PRETTY_HW}] Diagonal Analysis (Self-Compressibility)")
    for i in range(num_layers):
        print(f"   L{i}: Rank {int(rank_matrix[i,i])} | Loss {loss_matrix[i,i]:.3f}")
    return loss_matrix

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
# }}} === MISC ===
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
        return z

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
# {{{ === LoRAXS ===
class LoRAXSLinear(nn.Module):
    @staticmethod
    def from_linear(linear, r, dropout=0.0, verbose=False, **kwargs):
        if isinstance(linear, nn.QuantizedLinear):
            w = mx.dequantize(linear.weight, linear.scales, linear.biases, linear.group_size, linear.bits)
        else:
            w = linear.weight
        w_np = np.array(w.astype(mx.float32)) 
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
# }}} === LoRAXS ===
# {{{ === DRINK ===
def _trunc_SVD(U, S, Vt, rank):
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vt_k = Vt[:rank, :]
    return U_k, S_k, Vt_k

class LoRAONLynear(nn.Module):
    @staticmethod
    def from_weights(A, B, linear=None, bias=None, indices=None):
        has_bias = (bias is not None)
        lrl = LoRAONLynear(A.shape[1], B.shape[0], A.shape[0], bias=has_bias)
        if has_bias:
            lrl.lora_b.bias = bias
        lrl.lora_a.weight = mx.array(A)
        lrl.lora_b.weight = mx.array(B)
        lrl.linear = linear
        lrl.indices = mx.array(indices) if indices is not None else None
        return lrl

    def __init__(self, in_dims, out_dims, rank, bias=False):
        super().__init__()
        self.lora_a = nn.Linear(in_dims, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dims, bias=bias)
        self.linear = None
        self.indices = None

    def __call__(self, x):
        out = self.lora_b(self.lora_a(x))
        if self.linear is not None:
            out = out + self.linear(x)
        if self.indices is not None:
            out = out[..., self.indices]
        return out

def drink(model, compression_ratio=0.2, dry_run=False):
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
            print(f"   Skipping {s['full_path']}: Compress ({new_params}) > Org ({current_params})")
            continue
        print(f"   {s['full_path']:<40} | EffR: {s['eff_rank']:6.1f} | k: {max_rank} -> {k_target}")
        # {{
        _U, _S, _Vt = _trunc_SVD(s['U'], s['S'], s['Vt'], k_target)
        sqrt_S = np.sqrt(_S)
        B_np = _U * sqrt_S[None, :]  
        A_np = sqrt_S[:, None] * _Vt
        lrl_bias = s['module'].bias if "bias" in s['module'] else None
        # }}
        lrl = LoRAONLynear.from_weights(A_np, B_np, linear=None, bias=lrl_bias)
        layer = layers[s['layer_idx']]
        parts = s['name'].split('.')
        parent = layer
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], lrl)
    mx.eval(model)
    model.teacher_to_student = [(_l, _l) for _l in range(len(model.model.layers))]
    return model
# }}} === DRINK ===
# {{{ === RCR
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
# }}} === RCR
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

def distill(ds_id, model, tokenizer, config, teacher=None, n_epochs=1, lr=1e-4, bs=1, sl=4096, to='distilled.safetensors', add_dora=True, unfreeze_all=False):
    teacher_to_student = getattr(model, "teacher_to_student", None)
    student_indices = [s[1] for s in teacher_to_student] if teacher_to_student else []
    print(f'{teacher_to_student=}')
    from mlx.utils import tree_unflatten
    model.freeze()
    def to_dora_(layer):
        _rank = 64
        return DoRALinear.from_linear(layer, r=_rank, alpha=_rank, scale=1.0, dropout=0.0)
    if add_dora:
        for layer_idx, l in enumerate(model.model.layers):
            if getattr(l, 'n_rcr', None) and (layer_idx in student_indices):
                print(f'{layer_idx=}')
                loralized = [(k, to_dora_(m)) for k, m in l.named_modules() if k.endswith('proj')]
                l.update_modules(tree_unflatten(loralized))
                l.apply_to_modules(lambda k, v: v.unfreeze() if k.endswith('_new') else None)
    if unfreeze_all or not add_dora:
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

class AutoBalancingStudent(nn.Module): # https://arxiv.org/pdf/1705.07115
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
        return model.student
    return model

# }}} === TRAIN ===
# {{{ === TIE ===

def tie(model, tokenizer, config, collapse_cfg=None, do_calib=True, len_calib=256, n_calib_samples=64, bs=4, show=False):
    from mlx.utils import tree_unflatten

    if collapse_cfg is None:
        collapse_cfg = [
            {'rank': 64, 'targets': ['self_attn.v_proj'], 'layers': [10,11,12,13,14]},
        ]
    
    if show:
        # show_cfg = [dict(targets=[_t], rank=32) for _t in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'mlp.gate_proj', 'mlp.up_proj']]
        show_cfg = [dict(targets=[_t], rank=32) for _t in ['self_attn.v_proj']]
        try:
            import matplotlib.pyplot as plt
            for _c_cfg in show_cfg:
                # show_tie_1(model, _c_cfg)
                # show_tie_2(model, _c_cfg)
                # show_tie_3(model, _c_cfg)
                show_tie_4(model, _c_cfg)
            plt.show()
        except Exception as e:
            print(f"[{PRETTY_HW}] Visualization skipped: {e}")

    # {{ --- Calib ---
    layer_stats = {}
    if do_calib:
        class ActivationCollector(nn.Module):
            def __init__(self, original_module, stats_ref, layer_key):
                super().__init__()
                self.target = original_module
                self.stats = stats_ref
                self.key = layer_key

            def __call__(self, x, *args, **kwargs):
                x_flat = x.reshape(-1, x.shape[-1])
                x_f32 = mx.stop_gradient(x_flat).astype(mx.float32)
                
                c_batch = x_f32.T @ x_f32
                
                if self.key not in self.stats:
                    self.stats[self.key] = c_batch
                else:
                    self.stats[self.key] = self.stats[self.key] + c_batch
                
                return self.target(x, *args, **kwargs)

        original_modules = {}
        target_layer_indices = set()
        target_module_names = set()
        
        for cfg in collapse_cfg:
            layers_indices = cfg.get('layers', [])
            if layers_indices:
                target_layer_indices.update(layers_indices)
            if 'targets' in cfg:
                target_module_names.update(cfg['targets'])

        for i, l in enumerate(model.model.layers):
            if i in target_layer_indices:
                for name, m in l.named_modules():
                    if name in target_module_names:
                        full_name = f"{i}.{name}"
                        original_modules[full_name] = m
                        collector = ActivationCollector(m, layer_stats, full_name)
                        parts = name.split('.')
                        parent = l
                        for p in parts[:-1]:
                            parent = getattr(parent, p)
                        setattr(parent, parts[-1], collector)

        roper = Roper(config, len_calib)

        from datasets import load_dataset
        ds = load_dataset('Salesforce/wikitext', 'wikitext-103-v1', split="train", streaming=True)
        # ds = ds.filter(lambda example: len(example.get('text', '')) > (len_calib * 2))
        data_iter = iter(ds)
        
        model.eval()
        processed = 0
        pbar = tqdm(total=n_calib_samples, desc="Collecting Activations")
        cache = [lambda x, y: (x, y)] * len(model.model.layers)

        while processed < n_calib_samples:
            batch_iids = []
            while len(batch_iids) < bs:
                try:
                    row = next(data_iter)
                    txt = row.get('text', row.get('prompt', row.get('description', '')))
                    if txt: 
                        iid = tokenizer.encode(txt)
                        # if len(iid) > len_calib:
                        #     batch_iids.append(iid[:len_calib])
                        batch_iids.append(iid)
                except StopIteration:
                    break

            if not batch_iids: break

            max_len = max(len(t) for t in batch_iids) 
            if max_len == 0: continue
            input_ids = np.zeros((len(batch_iids), max_len), dtype=int)
            for i, t in enumerate(batch_iids):
                input_ids[i, :len(t)] = t
            
            input_ids_mx = mx.array(input_ids)
            B, L = input_ids_mx.shape

            rope = roper(mx.arange(L)[None, :])
            mask = create_causal_mask(mx.array([[True]*L]*B))
                
            model(input_ids_mx, mask, rope, cache) 
            mx.eval(layer_stats)
            
            processed += len(batch_iids)
            pbar.update(len(batch_iids))
        
        pbar.close()

        for i, l in enumerate(model.model.layers):
            if i in target_layer_indices:
                for name, m in l.named_modules():
                    if name in target_module_names:
                        full_name = f"{i}.{name}"
                        if full_name in original_modules:
                            parts = name.split('.')
                            parent = l
                            for p in parts[:-1]:
                                parent = getattr(parent, p)
                            setattr(parent, parts[-1], original_modules[full_name])
        mx.eval(model)
    # }} --- Calib ---

    print(f"[{PRETTY_HW}] Compressing using Vertical Stacking (Input Basis Sharing)...")

    all_replacements = []

    for cfg_idx, cfg in enumerate(collapse_cfg):
        targets = cfg.get('targets', ['self_attn.v_proj'])
        target_rank = cfg.get('rank', 16)
        target_layers = cfg.get('layers', [])
        
        if not target_layers: continue

        print(f"   -> Group {cfg_idx}: Rank={target_rank}, Targets={targets}, Layers={target_layers}")

        Ws = []
        layer_indices = []
        bundle_cov = None
        
        for i, l in enumerate(model.model.layers):
            if i not in target_layers: continue 
            
            for name, m in l.named_modules():
                if name in targets:
                    if isinstance(m, nn.QuantizedLinear):
                        w = mx.dequantize(m.weight, m.scales, m.biases, m.group_size, m.bits)
                    else:
                        w = m.weight
                    
                    Ws.append(w)
                    layer_indices.append((i, name, m))
                    
                    if do_calib:
                        full_name = f"{i}.{name}"
                        if full_name in layer_stats:
                            if bundle_cov is None:
                                bundle_cov = layer_stats[full_name]
                            else:
                                bundle_cov = bundle_cov + layer_stats[full_name]

        if not Ws: continue

        W_tall_mx = mx.concatenate(Ws, axis=0)
        W_tall_np = np.array(W_tall_mx.astype(mx.float32))
        d_out_total, d_in = W_tall_np.shape
        
        S_inv_np = None
        if do_calib and bundle_cov is not None:
            cov_np = np.array(bundle_cov.astype(mx.float32))
            ridge = 0.01 * np.mean(np.diag(cov_np))
            cov_reg = cov_np + ridge * np.eye(d_in, dtype=cov_np.dtype)
            try:
                S_cov_np = np.linalg.cholesky(cov_reg)
                S_inv_np = np.linalg.inv(S_cov_np)
                W_tall_np = W_tall_np @ S_cov_np
                print(f"      -> Group {cfg_idx}: ASVD scaling applied.")
            except Exception as e:
                print(f"      ! Cholesky failed for Group {cfg_idx}: {e}. Using standard SVD.")
                S_inv_np = None

        U, S, Vt = np.linalg.svd(W_tall_np, full_matrices=False)

        k_eff = min(target_rank, len(S))
        S_sqrt = np.sqrt(S[:k_eff])
        Vt_trunc = Vt[:k_eff, :] 
        
        if S_inv_np is not None:
            Vt_trunc = Vt_trunc @ S_inv_np
        
        A_shared = S_sqrt[:, None] * Vt_trunc
        A_shared_mx = mx.array(A_shared)
        
        shared_lora_a = nn.Linear(d_in, k_eff, bias=False)
        shared_lora_a.weight = A_shared_mx

        U_trunc = U[:, :k_eff]
        B_total = U_trunc * S_sqrt[None, :] 
        
        current_row_offset = 0
        for (layer_idx, name, orig_module) in layer_indices:
            h = orig_module.weight.shape[0]
            B_specific = B_total[current_row_offset : current_row_offset + h, :]
            current_row_offset += h

            has_bias = "bias" in orig_module
            lrl = LoRAONLynear(d_in, h, k_eff, bias=has_bias)
            
            lrl.lora_a = shared_lora_a
            
            lrl.lora_b.weight = mx.array(B_specific)
            if has_bias:
                lrl.lora_a.bias = orig_module.bias

            all_replacements.append((f"layers.{layer_idx}.{name}", lrl))
        
        print(f"      -> Compressed {len(Ws)} layers in Group {cfg_idx} to rank {k_eff}.")

    if all_replacements:
        model.model.update_modules(tree_unflatten(all_replacements))
        mx.eval(model)
        print(f"[{PRETTY_HW}] Tie Complete. Replaced {len(all_replacements)} modules.")
    
    return model

# }}} === TIE ===
# {{{ === DEACON ===
def deacon(model, tokenizer, config, groups=None, rank=64, verbose=True, do_align=True):
    from mlx.utils import tree_unflatten
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        do_align = False
    
    if groups is None: groups = [(0, len(model.model.layers))]
    targets = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

    for start, end in groups:
        layers = list(range(start, end))
        if len(layers) < 2: continue
        for name in targets:
            modules = []
            for i in layers:
                m = model.model.layers[i]
                for p in name.split('.'): m = getattr(m, p)
                modules.append((i, m))
            Ws = [np.array(m.weight.astype(mx.float32)) for _, m in modules]
            if not Ws: continue
            W_ref = Ws[0]
            aligned, perms = [W_ref], [None]
            for W in Ws[1:]:
                if do_align:
                    cost = -np.dot(W / np.linalg.norm(W, axis=1, keepdims=True), 
                                   (W_ref / np.linalg.norm(W_ref, axis=1, keepdims=True)).T)
                    r_ind, c_ind = linear_sum_assignment(cost)
                    P = np.argsort(c_ind)
                    W_aligned = W[c_ind, :] 
                    aligned.append(W_aligned)
                    perms.append(np.argsort(c_ind)) 
                else:
                    aligned.append(W)
                    perms.append(None)
            W_base = np.mean(np.stack(aligned), axis=0)
            base_mod = nn.Linear(W_base.shape[1], W_base.shape[0], bias=False)
            base_mod.weight = mx.array(W_base)
            replacements = []
            for i, (l_idx, orig) in enumerate(modules):
                R = aligned[i] - W_base
                U, S, Vt = np.linalg.svd(R, full_matrices=False)
                sqrt_S = np.sqrt(S[:rank])
                A = sqrt_S[:, None] * Vt[:rank, :]
                B = U[:, :rank] * sqrt_S[None, :]
                bias = orig.bias if "bias" in orig and orig.bias is not None else None
                new_mod = LoRAONLynear.from_weights(A, B, base_mod, bias=bias, indices=perms[i])
                replacements.append((f"layers.{l_idx}.{name}", new_mod))
            model.model.update_modules(tree_unflatten(replacements))
    mx.eval(model)
    return model
# }}} === DEACON ===
# {{{ === FISH ===
def fish(model, tokenizer, config, calib_ds_id, n_samples=64, bs=4, rank=32, targets=None, verbose=True):
    from mlx.utils import tree_unflatten, tree_flatten
    if targets is None:
        targets = ["self_attn.v_proj", "self_attn.o_proj", "mlp.down_proj"]
    stats_registry = {}

    def update_sym_matrix(old_mat, new_vecs):
        flat = new_vecs.reshape(-1, new_vecs.shape[-1])
        prod = flat.T @ flat 
        if old_mat is None:
            return prod
        return old_mat + prod

    class GFWWrapper(nn.Module):
        def __init__(self, linear, layer_id):
            super().__init__()
            self.linear = linear
            self.layer_id = layer_id
            self.probe = mx.zeros((1,)) 
            stats_registry[layer_id] = {'n': 0, 'S_in': None, 'S_out': None}

        def __call__(self, x):
            if not self.training:
                return self.linear(x)
            y = self.linear(x)
            x_detached = mx.stop_gradient(x)
            stats = stats_registry[self.layer_id]
            stats['S_in'] = update_sym_matrix(stats['S_in'], x_detached)
            stats['n'] += (x.shape[0] * x.shape[1])
            return y + self.probe
    original_modules = {}
    wrappers = []
    model.freeze() 
    for i, l in enumerate(model.model.layers):
        for name, m in l.named_modules():
            if name in targets and isinstance(m, nn.Linear):
                full_path = f"{i}.{name}"
                original_modules[full_path] = m
                wrapper = GFWWrapper(m, full_path)
                wrappers.append(wrapper)
                parts = name.split('.')
                parent = l
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], wrapper)
    
    print(f"   -> Instrumented {len(wrappers)} layers. Starting Calibration...")

    from datasets import load_dataset
    ds = load_dataset(calib_ds_id, split="test", streaming=True)
    data_iter = iter(ds)
    cache = [(lambda x,y:(x,y)) for _ in range(len(model.model.layers))]
    
    def step_fn(model, x, mask, rope):
        logits = model(x, mask, rope, cache)
        logits = logits[:, :-1, :]
        targets = x[:, 1:]
        loss = nn.losses.cross_entropy(logits, targets)
        return mx.mean(loss)
    
    loss_and_grad = nn.value_and_grad(model, step_fn)
    
    model.train()
    processed = 0
    pbar = tqdm(total=n_samples, desc="Fisher Calibration")
    
    while processed < n_samples:
        batch_iids = []
        while len(batch_iids) < bs:
            try:
                row = next(data_iter)
                txt = row.get('text', row.get('prompt', row.get('description', '')))
                if txt: batch_iids.append(tokenizer.encode(txt))
            except StopIteration:
                break
        if not batch_iids: break
        
        max_len = max(len(t) for t in batch_iids)
        input_ids = np.zeros((len(batch_iids), max_len), dtype=int)
        for i, t in enumerate(batch_iids): input_ids[i, :len(t)] = t
        input_ids = mx.array(input_ids)
        B, L = input_ids.shape
        
        for w in wrappers:
            out_dim = w.linear.weight.shape[0]
            w.probe = mx.zeros((B, L, out_dim))
        
        roper = Roper(config, L)
        rope = roper(mx.arange(L)[None, :])
        mask = create_causal_mask(mx.array([[True]*L]*B))
        
        loss, grads = loss_and_grad(model, input_ids, mask, rope)
        
        grad_leaves = tree_flatten(grads)
        for path, g_val in grad_leaves:
            if path.endswith(".probe"):
                for layer_id in stats_registry:
                    if path.endswith(f"layers.{layer_id}.probe"):
                        stats_registry[layer_id]['S_out'] = update_sym_matrix(stats_registry[layer_id]['S_out'], g_val)
                        break

        mx.eval(loss)
        processed += B
        pbar.update(B)
    pbar.close()

    replacements = []
    
    def compute_sqrt_and_inv(M, epsilon=1e-5):
        d, V = np.linalg.eigh(M)
        d = np.maximum(d, epsilon)
        sqrt_d = np.sqrt(d)
        inv_sqrt_d = 1.0 / sqrt_d
        M_sqrt = (V * sqrt_d[None, :]) @ V.T
        M_inv_sqrt = (V * inv_sqrt_d[None, :]) @ V.T
        return M_sqrt, M_inv_sqrt

    for layer_id, stats in stats_registry.items():
        if stats['S_in'] is None or stats['S_out'] is None:
            if verbose: print(f"      ! No stats for {layer_id}, skipping.")
            continue
        orig_module = original_modules[layer_id]
        W_orig = orig_module.weight
        if isinstance(orig_module, nn.QuantizedLinear):
            W_orig = mx.dequantize(W_orig, orig_module.scales, orig_module.biases, orig_module.group_size, orig_module.bits)
        W_np = np.array(W_orig.astype(mx.float32)) # (Out, In)
        n = stats['n']
        C_in = np.array(stats['S_in'].astype(mx.float32)) / n
        C_out = np.array(stats['S_out'].astype(mx.float32)) / n
        
        try:
            S_in_sqrt, S_in_inv = compute_sqrt_and_inv(C_in)
            S_out_sqrt, S_out_inv = compute_sqrt_and_inv(C_out)
            W_tilde = S_out_sqrt @ W_np @ S_in_sqrt
            U, S, Vt = np.linalg.svd(W_tilde, full_matrices=False)
            k = min(rank, len(S))
            U_k = U[:, :k]
            S_k = S[:k]
            Vt_k = Vt[:k, :]
            sqrt_S = np.sqrt(S_k)
            B_prime = S_out_inv @ (U_k * sqrt_S[None, :])
            A_prime = (sqrt_S[:, None] * Vt_k) @ S_in_inv
            has_bias = "bias" in orig_module and orig_module.bias is not None
            bias_val = orig_module.bias if has_bias else None
            lrl = LoRAONLynear.from_weights(A_prime, B_prime, linear=None, bias=bias_val)
            replacements.append((f"layers.{layer_id}", lrl))
        except Exception as e:
            if verbose: print(f"      ! Failed for {layer_id}: {e}")
            replacements.append((f"layers.{layer_id}", orig_module))

    for layer_id in stats_registry:
        full_path_key = f"layers.{layer_id}"
        already_handled = any(r[0] == full_path_key for r in replacements)
        if not already_handled:
            replacements.append((full_path_key, original_modules[layer_id]))

    if replacements:
        model.model.update_modules(tree_unflatten(replacements))
    
    model.unfreeze()
    mx.eval(model)
    print(f"[{PRETTY_HW}] Replaced {len([r for r in replacements if isinstance(r[1], LoRAONLynear)])} layers.")
    return model

# }}} === FISH ===
