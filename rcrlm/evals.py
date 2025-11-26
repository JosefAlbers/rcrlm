import datasets
datasets.disable_progress_bar()

from .utils import Roper, create_causal_mask, infer
from tqdm import tqdm
import mlx.core as mx
import mlx.nn as nn

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval import simple_evaluate
from .utils import Roper, create_causal_mask, infer

@register_model("my_custom_mlx")
class MLXCustomEval(LM):
    def __init__(self, model, tokenizer, config, batch_size=1):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.batch_size_per_gpu = batch_size
        self.roper = Roper(self.config) 
        if isinstance(config.eos_token_id, int):
            self.eos_token_id = config.eos_token_id
        else:
            self.eos_token_id = config.eos_token_id[0]

    def loglikelihood(self, requests):
        results = []
        for request in tqdm(requests, desc="Evaluating loglikelihood"):
            if isinstance(request, tuple):
                context, continuation = request
            else:
                context, continuation = request.args
            ctx_ids = self.tokenizer.encode(context)
            cont_ids = self.tokenizer.encode(continuation)
            full_ids = ctx_ids + cont_ids
            input_ids = mx.array([full_ids]) # Batch size 1
            dummy_cache = [lambda x, y: (x, y)] * self.config.num_hidden_layers
            X = input_ids[:, :-1]
            y = input_ids[:, 1:]
            seq_len = X.shape[1]
            attention_mask = [True] * seq_len
            causal_mask = create_causal_mask([attention_mask]) 
            positions = mx.array([list(range(seq_len))])
            rope = self.roper(positions)
            logits = self.model(X, causal_mask, rope, dummy_cache)
            start_idx = len(ctx_ids) - 1
            end_idx = len(full_ids) - 1
            relevant_logits = logits[:, start_idx:end_idx, :]
            relevant_targets = mx.array(cont_ids)[None, :]
            nlls = nn.losses.cross_entropy(relevant_logits, relevant_targets, reduction='none')
            log_prob_sum = -nlls.sum().item()
            is_greedy = (relevant_logits.argmax(axis=-1) == relevant_targets).all().item()
            results.append((log_prob_sum, is_greedy))
            mx.eval(logits, nlls)
        return results

    def generate_until(self, requests):
        results = []
        for request in tqdm(requests, desc="Evaluating generation"):
            if isinstance(request, tuple):
                context, gen_kwargs = request
            else:
                context, gen_kwargs = request.args
            until = gen_kwargs.get("until", [])
            if isinstance(until, str): until = [until]
            max_gen_toks = gen_kwargs.get("max_gen_toks", 256)
            out = infer(
                prompts=[context],
                model=self.model,
                tokenizer=self.tokenizer,
                config=self.config,
                max_new_tokens=max_gen_toks,
                use_chat_template=True, 
                stream=False,
                verbose=False
            )
            response = out['out_str'][0]
            for term in until:
                if term in response:
                    response = response.split(term)[0]
            results.append(response)
        return results

    def loglikelihood_rolling(self, requests):
        pass 

def eval_lm(model, tokenizer, config,
    tasks=[
        "mmlu", 
        "gpqa", 
        "gsm8k", 
        "mgsm_direct", 
        # "mbpp", 
        # "humaneval", 
    ],
    limit=10,
    allow_code_eval=False,
):
    if allow_code_eval:
        import os
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    # mcq: "mmlu", "mmlu_redux", "gpqa"
    # gen: "gsm8k", "gsm8k_cot", "bbh_cot_fewshot", "minerva_math", "mgsm_direct"
    lm_obj = MLXCustomEval(model=model, tokenizer=tokenizer, config=config)
    
    print(f"Starting lm-evaluation-harness on: {tasks}")
    results = simple_evaluate(
        model=lm_obj,
        tasks=tasks,
        limit=limit,
        batch_size=1
    )

    from lm_eval.utils import make_table
    print(make_table(results))

