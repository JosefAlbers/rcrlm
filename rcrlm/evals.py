from .utils import Roper, create_causal_mask, infer
from datetime import datetime
from tqdm import tqdm
import mlx.core as mx
import mlx.nn as nn

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval import simple_evaluate
from .utils import Roper, create_causal_mask, infer

@register_model("my_custom_mlx")
class MLXCustomEval(LM):
    def __init__(self, model, tokenizer, config, max_new_tokens, batch_size=1, chat_template_kwargs=None):
        super().__init__()
        self._model = model
        self.tokenizer = tokenizer
        self.config = config
        self.batch_size_per_gpu = batch_size
        self.roper = Roper(self.config) 
        self.max_new_tokens=max_new_tokens
        self.chat_template_kwargs=chat_template_kwargs
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
            logits = self._model(X, causal_mask, rope, dummy_cache)
            start_idx = len(ctx_ids) - 1
            end_idx = len(full_ids) - 1
            relevant_logits = logits[:, start_idx:end_idx, :]
            relevant_targets = mx.array(cont_ids)[None, :]
            nlls = nn.losses.cross_entropy(relevant_logits, relevant_targets, reduction='none')
            log_prob_sum = -nlls.sum().item()
            is_greedy = (relevant_logits.argmax(axis=-1) == relevant_targets).all().item()
            result = (log_prob_sum, is_greedy)
            mx.async_eval(result)
            results.append(result)
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
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_new_tokens)
            out = infer(
                prompts=[context],
                model=self._model,
                tokenizer=self.tokenizer,
                config=self.config,
                max_new_tokens=max_gen_toks,
                use_chat_template=True, 
                stream=False,
                verbose=False,
                chat_template_kwargs=self.chat_template_kwargs,
            )
            response = out['out_str'][0]
            for term in until:
                if term in response:
                    response = response.split(term)[0]
            results.append(response)
        return results

    def loglikelihood_rolling(self, requests):
        pass 


def plot_results(eval_output):
    import matplotlib.pyplot as plt
    import numpy as np
    results_data = eval_output['results']
    metric_priority = [
        "exact_match,flexible-extract",
        "exact_match,get-answer",
        "exact_match,none",
        "acc,none",
        "acc_norm,none",
        "exact_match,strict-match"
    ]

    group_map = {
        "mmlu": "MMLU",
        "gsm8k": "GSM8k",
        "mgsm": "MGSM",
        "gpqa": "GPQA",
        "mbpp": "MBPP",
    }

    grouped_scores = {}

    for task_name, metrics in results_data.items():
        score = None
        
        for key in metric_priority:
            if key in metrics:
                score = metrics[key]
                break
        
        if score is None:
            candidates = [v for k, v in metrics.items() 
                          if isinstance(v, (int, float)) and ('acc' in k or 'match' in k)]
            if candidates:
                score = max(candidates)
            else:
                score = 0.0 

        display_name = task_name 
        
        for prefix, group_label in group_map.items():
            if task_name.startswith(prefix):
                display_name = group_label
                break
        
        if display_name not in grouped_scores:
            grouped_scores[display_name] = []
        
        grouped_scores[display_name].append(score)

    final_tasks = []
    final_values = []

    for name, scores_list in grouped_scores.items():
        avg_score = sum(scores_list) / len(scores_list)
        final_tasks.append(name)
        final_values.append(avg_score)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(final_tasks, final_values, color='#88c999', edgecolor='black', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.title("Model Evaluation Results (Aggregated & Filtered)")
    plt.ylabel("Score")
    plt.ylim(0, 1.1) 
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    fn = datetime.now().strftime(format="%Y%m%d_%H%M%S")
    plt.savefig(f'{fn}.png')
    
    from lm_eval.utils import make_table
    full_table = make_table(eval_output)
    summ_table = '\n'.join(f'✓ {k:20}: {v:5.2f}' for k, v in zip(final_tasks, final_values))
    with open(f'{fn}.txt', 'w') as f:
        f.write(full_table+'\n'+summ_table)
    print(full_table)
    # print(summ_table)
    # return dict(zip(final_tasks, final_values))
    return summ_table

def eval_lm(model, tokenizer, config,
    tasks=[
        "mmlu", 
        "gpqa_main_zeroshot", 
        "gsm8k", 
        "mgsm_direct_zh", 
        # "mbpp", 
        # "humaneval", 
    ],
    limit=20,
    max_new_tokens=512,
    chat_template_kwargs=None,
    allow_code_eval=False,
):
    if allow_code_eval:
        import os
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    # mcq: "mmlu", "mmlu_redux", "gpqa"
    # gen: "gsm8k", "gsm8k_cot", "bbh_cot_fewshot", "minerva_math", "mgsm_direct"
    lm_obj = MLXCustomEval(model=model, tokenizer=tokenizer, config=config, max_new_tokens=max_new_tokens, chat_template_kwargs=chat_template_kwargs)
    
    print(f"Starting lm-evaluation-harness on: {tasks}")
    results = simple_evaluate(
        model=lm_obj,
        tasks=tasks,
        limit=limit,
        batch_size=1,
        num_fewshot=0,
    )
    plotted = plot_results(results)
    return plotted
