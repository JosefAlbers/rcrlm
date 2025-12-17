import mlx.core as mx
import mlx.nn as nn
from .qwen3 import Qwen3ForCausalLM, Qwen3Model
from dataclasses import dataclass

@dataclass
class TamoConfig:
    table_hidden_size: int = 768
    table_num_layers: int = 3
    table_num_heads: int = 12
    max_rows: int = 64
    max_cols: int = 32
    cell_dim: int = 64

class HyperGraphLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.table_hidden_size
        self.row_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.col_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.update_gate = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)

    def __call__(self, cell_embeds):
        B, R, C, H = cell_embeds.shape
        row_context = mx.mean(cell_embeds, axis=2, keepdims=True)
        col_context = mx.mean(cell_embeds, axis=1, keepdims=True)
        row_msg = mx.broadcast_to(self.row_proj(row_context), (B, R, C, H))
        col_msg = mx.broadcast_to(self.col_proj(col_context), (B, R, C, H))
        combined = mx.concatenate([cell_embeds, row_msg, col_msg], axis=-1)
        update = nn.silu(self.update_gate(combined))
        return self.norm(cell_embeds + update)

class TableEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_cell = nn.Linear(config.cell_dim, config.table_hidden_size)
        self.pos_embed_row = nn.Embedding(config.max_rows, config.table_hidden_size)
        self.pos_embed_col = nn.Embedding(config.max_cols, config.table_hidden_size)
        
        self.layers = [HyperGraphLayer(config) for _ in range(config.table_num_layers)]
        self.out_norm = nn.LayerNorm(config.table_hidden_size)

    def __call__(self, table_features):
        B, R, C, D = table_features.shape
        x = self.embed_cell(table_features)
        row_ids = mx.arange(R)[None, :, None]
        col_ids = mx.arange(C)[None, None, :]
        x = x + self.pos_embed_row(row_ids) + self.pos_embed_col(col_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.out_norm(x)
        return x.reshape(B, R * C, -1)

class TAMOQwen3(nn.Module):
    def __init__(self, config, tamo_config=None):
        super().__init__()
        self.llm = Qwen3ForCausalLM(config) 
        if tamo_config is None:
            tamo_config = TamoConfig()
        self.table_encoder = TableEncoder(tamo_config)
        self.projector = nn.Linear(tamo_config.table_hidden_size, config.hidden_size)

    def encode_table(self, table_features):
        table_embeds = self.table_encoder(table_features)
        return self.projector(table_embeds)

    def __call__(self, input_ids, attention_mask=None, rope=None, cache=None, table_features=None, table_insert_indices=None):
        inputs_embeds = self.llm.model.embed_tokens(input_ids) #
        if table_features is not None and table_insert_indices is not None:
            table_embeds = self.encode_table(table_features)
            inputs_embeds = mx.concatenate([table_embeds, inputs_embeds], axis=1)
            if attention_mask is not None:
                B, L_text = input_ids.shape
                _, L_table, _ = table_embeds.shape
                pad_mask = mx.ones((B, L_table), dtype=mx.bool_)
                attention_mask = mx.concatenate([pad_mask, attention_mask], axis=1)
        x = inputs_embeds
        captures = []
        B, L, _ = x.shape
        if rope is None or rope[0].shape[2] != L:
            pass 
        for _idx, (c, layer) in enumerate(zip(cache, self.llm.model.layers)):
            x = layer(x, attention_mask=attention_mask, rope=rope, cache=c)
        x = self.llm.model.norm(x)
        if self.llm.tie:
            logits = self.llm.model.embed_tokens.as_linear(x)
        else:
            logits = self.llm.lm_head(x)
        return logits
