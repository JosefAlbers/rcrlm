# rcr-lm

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XYyO04by5UCaMoH0C2MUmsLPOwddN5Gp?usp=sharing)

A lightweight, high-performance research framework for Large Language Models built on Apple's [MLX](https://github.com/ml-explore/mlx).

## Quickstart

`rcr-lm` is available on PyPI:

```bash
# macOS
pip install rcrlm

# Linux with CUDA
pip install rcrlm[cuda]

# Linux (CPU only)
pip install rcrlm[cpu]
```

To generate text with an LLM:

```bash
>>> rlm

┌────────────────────────────── Streaming ──────────────────────────────┐
<think>
Okay, the user wants a short introduction to a large language model. Let me start by recalling what I know about LLMs. They're big language models, right? So I should mention their ability to understand and generate text. Maybe start with the basics: they can process and generate text, not just a few words. Then explain their training data, like the amount of text they're trained on. Also, their capabilities: understanding and generating text, answering questions, etc. Need
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
Give me a short introduction to large language model.
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐
<think>
Okay, the user wants a short introduction to a large language model. Let me start by recalling what I know about LLMs. They're big language models, right? So I should mention their ability to understand and generate text. Maybe start with the basics: they can process and generate text, not just a few words. Then explain their training data, like the amount of text they're trained on. Also, their capabilities: understanding and generating text, answering questions, etc. Need
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    298.1 tokens/sec ( 18 tokens in 0.1s)
Tokens generation:    217.3 tokens/sec (100 tokens in 0.5s)
└───────────────────────────────────────────────────────────────────────┘
```

## Key Features

### Accelerated Inference

`rcr-lm` achieves generation speeds exceeding 200 tokens/sec, offering a measurable performance uplift over standard MLX implementations.

```python
from rcrlm import load, infer
m = load()
_ = infer("Write a story about Einstein\n", **m, max_new_tokens=256)
```

```
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
Write a story about Einstein
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐
<think>
Okay, the user wants a story about Einstein. Let me start by recalling Einstein's life. He was a genius, a scientist, and a philosopher. I need to make sure the story includes his contributions to science, maybe his work on relativity, and his personal life.

First, I should set the scene. Maybe start with his early life in Germany, where he was born. Then introduce his family, his parents, maybe his mother's influence. Then his education, the famous lectures, and his breakthroughs.

I need to highlight his scientific achievements, like the theory of relativity. Also, his personal struggles, like the time he spent in the Alps, the Alps being a place of isolation and inspiration.

I should include some quotes or references to his work. Maybe mention his quote about the universe being infinite. Also, his later years and how he passed away.

Wait, the user might want the story to be engaging and highlight his legacy. I need to make sure the story flows well, with a good narrative arc. Avoid clichés, but still capture his essence. Check for any inaccuracies, like his actual birth date and death year. Let me confirm: Einstein was born on April 14, 18
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    232.2 tokens/sec ( 14 tokens in 0.1s)
Tokens generation:    200.2 tokens/sec (256 tokens in 1.3s)
└───────────────────────────────────────────────────────────────────────┘
```

#### mlx-lm (for comparison)

```python
from mlx_lm import load, generate
model, tokenizer = load("Qwen/Qwen3-0.6B")
prompt = "Write a story about Einstein"
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
text = generate(model, tokenizer, prompt=prompt, verbose=True)
```

```
<think>
Okay, the user wants a story about Einstein. Let me start by recalling Einstein's life and achievements. He was a genius, but the story needs to be engaging. Maybe set it in his early years to highlight his early brilliance. I should include some key moments, like his work on the theory of relativity, but also show his personal life and challenges.

I need to make sure the story has a beginning, middle, and end. Maybe start with his childhood, then his education, his breakthroughs, and his later years. Including some quotes from his work would add depth. Also, the user might want to know about his legacy, so I should mention his impact on science and society.

Wait, the user didn't specify the genre. It could be a historical fiction or a modern story. Since Einstein is a well-known figure, maybe a historical account would be better. But I should make sure the story is engaging and not too technical. Maybe include some emotional elements, like his struggles with time and his family.

I should check for any inaccuracies. For example, his early life was not as famous as he is known. Maybe mention his parents, his education, and his eventual fame. Also, the story should end on a positive note,
==========
Prompt: 13 tokens, 34.744 tokens-per-sec
Generation: 256 tokens, 174.251 tokens-per-sec
Peak memory: 1.415 GB
```

### Batched Decoding

Also natively supports batched decoding.

```python
infer(["#write a quick sort algorithm\n", "Give me a short introduction to large language model.\n", "Write a neurology ICU admission note.\n", "Comparison of Sortino Ratio for Bitcoin and Ethereum."], **m)
```

```
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
#write a quick sort algorithm
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐
<think>
Okay, I need to write a quick sort algorithm. Let me think about how to approach this. Quick sort is a divide-and-conquer algorithm, right? The basic idea is to select a pivot element, partition the array into elements less than or equal to the pivot and greater than or equal to it, and then recursively sort the subarrays.

First, I should outline the steps. The algorithm should have a function that takes an array and a pivot index. Wait, but how do
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Inp 00001 ──────────────────────────────┐
<|im_start|>user
Give me a short introduction to large language model.
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00001 ──────────────────────────────┐
<think>
Okay, the user wants a short introduction to a large language model. Let me start by recalling what I know about LLMs. They're big language models, right? So I should mention their ability to understand and generate text. Maybe start with the basics: they can process and generate text, not just a few words. Then explain their training data, like the amount of text they're trained on. Also, their capabilities: understanding and generating text, answering questions, etc. Need
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Inp 00002 ──────────────────────────────┐
<|im_start|>user
Write a neurology ICU admission note.
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00002 ──────────────────────────────┐
<think>
Okay, I need to write a neurology ICU admission note. Let me start by recalling what an ICU admission note typically includes. It's a medical record that outlines the patient's condition, initial assessment, interventions, and any ongoing care.

First, the patient's name and date of admission. I should make sure to include that. Then, the patient's name, age, gender, and primary diagnosis. Since it's a neurology ICU, the main issue is likely a neurological
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Inp 00003 ──────────────────────────────┐
<|im_start|>user
Comparison of Sortino Ratio for Bitcoin and Ethereum.<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00003 ──────────────────────────────┐
<think>
Okay, the user is asking for a comparison between the Sortino Ratio for Bitcoin and Ethereum. Let me start by recalling what the Sortino Ratio is. It's a measure of risk-adjusted return for a portfolio, right? It's calculated as (Return - Risk-Free Rate) divided by the Risk (Standard Deviation). The Sortino Ratio is usually used for a single asset, so comparing it between Bitcoin and Ethereum would be useful.

First, I need to check the historical data
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:   1411.6 tokens/sec ( 72 tokens in 0.1s)
Tokens generation:    469.2 tokens/sec (400 tokens in 0.9s)
└───────────────────────────────────────────────────────────────────────┘
```

### Efficient Fine-Tuning

Supports DoRA (Weight-Decomposed Low-Rank Adaptation) for parameter-efficient training workflows.

```python
from rcrlm import train
m = load()
lora_test_path = 'test_lora.safetensors'
train("RandomNameAnd6/SVGenerator", **m, lora_cfg=dict(wt_to=lora_test_path))
del m
m = load()
_ = infer("medium red circle\n", **m, lora_path=lora_test_path, stream=False, max_new_tokens=256, use_jit=False)
```

```
〄 Testing DoRA training...
epoch=    0 avg_loss=    0.29 elp_train=   10.18
└ test output: ['<svg width="100" height="100" viewBox="-50 -5']
epoch=    1 avg_loss=    0.04 elp_train=   11.19
└ test output: ['<svg width="100" height="100" viewBox="-50 -5']
〄 Testing DoRA decoding...
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
medium red circle<|im_end|>
<|im_start|>assistant
<think>

</think>
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐
<svg width="100" height="100" viewBox="-50 -50 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="0" cy="0" r="18" fill="#f6f448"/></svg>
<|im_end|>
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    435.0 tokens/sec ( 15 tokens in 0.0s)
Tokens generation:    162.9 tokens/sec (256 tokens in 1.6s)
└───────────────────────────────────────────────────────────────────────┘
```

Also supports Knowledge Distillation (reverse KLD) from a teacher model.

```python
m = load()
m['model'] = collapse(m['model'])
_ = infer("Write a story about Einstein\n", **m, stream=False)
teacher = load()['model']
m['model'] = distill("HuggingFaceH4/instruction-dataset", **m, teacher=teacher)
_ = infer("Write a story about Einstein\n", **m, stream=False)
```

### Integrated Evaluation

Native integration with `lm-evaluation-harness`. Benchmark vanilla and customized models against standard metrics (MMLU, GSM8K, etc.) with a single command.

```python
from rcrlm.evals import eval_lm
m = load()
e_orgn = eval_lm(**m, chat_template_kwargs=dict(enable_thinking=False))
print('〄 Testing lm-eval on collapsed model...')
m['model'] = collapse(m['model'])
e_coll = eval_lm(**m, chat_template_kwargs=dict(enable_thinking=False))
print('〄 Testing lm-eval on healed model...')
teacher = load()['model']
m['model'] = distill("HuggingFaceH4/instruction-dataset", **m, teacher=teacher)
e_heal = eval_lm(**m, chat_template_kwargs=dict(enable_thinking=False))
print(f'Original:\n{e_orgn}\nCollapsed:\n{e_coll}\nHealed:\n{e_heal}')
```

```
Original:
✓ GPQA                :  0.33
✓ GSM8k               :  1.00
✓ MGSM                :  0.67
✓ MMLU                :  0.37
Collapsed:
✓ GPQA                :  0.33
✓ GSM8k               :  0.00
✓ MGSM                :  0.00
✓ MMLU                :  0.26
Healed:
✓ GPQA                :  0.00
✓ GSM8k               :  0.00
✓ MGSM                :  0.00
✓ MMLU                :  0.29
```

<details><summary>Codes adapted from [nnx-lm](https://pypi.org/project/nnx-lm/) to try some stuff</summary><pre>
## Etc

```
>>> import rcrlm
>>> rcrlm.main.test()

〄 Testing vanilla decoding...
┌────────────────────────────── Streaming ──────────────────────────────┐
<think>
Okay, the user wants a story about Einstein. Let me start by recalling Einstein's life. He was a genius, a scientist, and a philosopher. I need to make sure the story includes his contributions to science, maybe his work on relativity, and his personal life.

First, I should set the scene. Maybe start with his early life in Germany, where he was born. Then introduce his family, his parents, maybe his mother's influence. Then his education, the famous lectures, and his breakthroughs.

I need to highlight his scientific achievements, like the theory of relativity. Also, his personal struggles, like the time he spent in the Alps, the Alps being a place of isolation and inspiration.

I should include some quotes or references to his work. Maybe mention his quote about the universe being infinite. Also, his later years and how he passed away.

Wait, the user might want the story to be engaging and highlight his legacy. I need to make sure the story flows well, with a good narrative arc. Avoid clichés, but still capture his essence. Check for any inaccuracies, like his actual birth date and death year. Let me confirm: Einstein was born on April 14, 18
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
Write a story about Einstein
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐
<think>
Okay, the user wants a story about Einstein. Let me start by recalling Einstein's life. He was a genius, a scientist, and a philosopher. I need to make sure the story includes his contributions to science, maybe his work on relativity, and his personal life.

First, I should set the scene. Maybe start with his early life in Germany, where he was born. Then introduce his family, his parents, maybe his mother's influence. Then his education, the famous lectures, and his breakthroughs.

I need to highlight his scientific achievements, like the theory of relativity. Also, his personal struggles, like the time he spent in the Alps, the Alps being a place of isolation and inspiration.

I should include some quotes or references to his work. Maybe mention his quote about the universe being infinite. Also, his later years and how he passed away.

Wait, the user might want the story to be engaging and highlight his legacy. I need to make sure the story flows well, with a good narrative arc. Avoid clichés, but still capture his essence. Check for any inaccuracies, like his actual birth date and death year. Let me confirm: Einstein was born on April 14, 18
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    232.2 tokens/sec ( 14 tokens in 0.1s)
Tokens generation:    200.2 tokens/sec (256 tokens in 1.3s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing batch decoding...
┌────────────────────────────── Streaming ──────────────────────────────┐
<think>
Okay, the user is asking for a comparison between the Sortino Ratio for Bitcoin and Ethereum. Let me start by recalling what the Sortino Ratio is. It's a measure of risk-adjusted return for a portfolio, right? It's calculated as (Return - Risk-Free Rate) divided by the Risk (Standard Deviation). The Sortino Ratio is usually used for a single asset, so comparing it between Bitcoin and Ethereum would be useful.

First, I need to check the historical data
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
#write a quick sort algorithm
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐
<think>
Okay, I need to write a quick sort algorithm. Let me think about how to approach this. Quick sort is a divide-and-conquer algorithm, right? The basic idea is to select a pivot element, partition the array into elements less than or equal to the pivot and greater than or equal to it, and then recursively sort the subarrays.

First, I should outline the steps. The algorithm should have a function that takes an array and a pivot index. Wait, but how do
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Inp 00001 ──────────────────────────────┐
<|im_start|>user
Give me a short introduction to large language model.
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00001 ──────────────────────────────┐
<think>
Okay, the user wants a short introduction to a large language model. Let me start by recalling what I know about LLMs. They're big language models, right? So I should mention their ability to understand and generate text. Maybe start with the basics: they can process and generate text, not just a few words. Then explain their training data, like the amount of text they're trained on. Also, their capabilities: understanding and generating text, answering questions, etc. Need
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Inp 00002 ──────────────────────────────┐
<|im_start|>user
Write a neurology ICU admission note.
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00002 ──────────────────────────────┐
<think>
Okay, I need to write a neurology ICU admission note. Let me start by recalling what an ICU admission note typically includes. It's a medical record that outlines the patient's condition, initial assessment, interventions, and any ongoing care.

First, the patient's name and date of admission. I should make sure to include that. Then, the patient's name, age, gender, and primary diagnosis. Since it's a neurology ICU, the main issue is likely a neurological
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Inp 00003 ──────────────────────────────┐
<|im_start|>user
Comparison of Sortino Ratio for Bitcoin and Ethereum.<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00003 ──────────────────────────────┐
<think>
Okay, the user is asking for a comparison between the Sortino Ratio for Bitcoin and Ethereum. Let me start by recalling what the Sortino Ratio is. It's a measure of risk-adjusted return for a portfolio, right? It's calculated as (Return - Risk-Free Rate) divided by the Risk (Standard Deviation). The Sortino Ratio is usually used for a single asset, so comparing it between Bitcoin and Ethereum would be useful.

First, I need to check the historical data
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:   1411.6 tokens/sec ( 72 tokens in 0.1s)
Tokens generation:    469.2 tokens/sec (400 tokens in 0.9s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing DoRA training...
epoch=    0 avg_loss=    0.29 elp_train=   10.18
└ test output: ['<svg width="100" height="100" viewBox="-50 -5']
epoch=    1 avg_loss=    0.04 elp_train=   11.19
└ test output: ['<svg width="100" height="100" viewBox="-50 -5']
〄 Testing DoRA decoding...
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
medium red circle<|im_end|>
<|im_start|>assistant
<think>

</think>
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐
<svg width="100" height="100" viewBox="-50 -50 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="0" cy="0" r="18" fill="#f6f448"/></svg>
<|im_end|>
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    435.0 tokens/sec ( 15 tokens in 0.0s)
Tokens generation:    162.9 tokens/sec (256 tokens in 1.6s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing collapse...
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
Write a story about Einstein
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐
</think>
The story was written in the day of the day of the day of the day, but I can't understand it. it's a time I can't understand. it's a time I can say it's a day of the day. it's a time I can say it can't be. it's the day of the day. it's the day of the day. it's the day of the day. it's the day of the day. it's the day of the
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    511.4 tokens/sec ( 14 tokens in 0.0s)
Tokens generation:    231.0 tokens/sec (100 tokens in 0.4s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing healing...
teacher_to_student=[(16, 13)]
epoch=    0 avg_loss=    0.45 elp_train=   21.74
└ test output: ["I'm sorry, I can't help with that. It's a medium red circle. Let me"]
epoch=    1 avg_loss=    0.47 elp_train=   21.78
└ test output: ["I'm sorry, I can't help with that. It's a medium red circle. Let me"]
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
Write a story about Einstein
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐
<think>
Okay, I need to write a story about Einstein. Let me think about the themes. First, Einstein was a genius, so I should focus on his life. Maybe start with his early life. He was born in 1905, so I need to include his early years. Then his work on the theory of relativity. Maybe his work on the equation. His death and the impact on science. Then his contributions. Maybe his work on the equation. His death,
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    384.2 tokens/sec ( 14 tokens in 0.0s)
Tokens generation:    171.1 tokens/sec (100 tokens in 0.6s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing lm-eval on original model...
Starting lm-evaluation-harness on: ['mmlu', 'gpqa_main_zeroshot', 'gsm8k', 'mgsm_direct_zh']
Evaluating generation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:08<00:00,  1.43s/it]
Evaluating loglikelihood: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 696/696 [00:36<00:00, 19.12it/s]
|                 Tasks                 |Version|     Filter      |n-shot|  Metric   |   |Value |   |Stderr|
|---------------------------------------|------:|-----------------|-----:|-----------|---|-----:|---|-----:|
|gpqa_main_zeroshot                     |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|                                       |       |none             |     0|acc_norm   |↑  |0.3333|±  |0.3333|
|gsm8k                                  |      3|flexible-extract |     0|exact_match|↑  |1.0000|±  |0.0000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_zh                         |      3|flexible-extract |     0|exact_match|↑  |0.6667|±  |0.3333|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mmlu                                   |      2|none             |      |acc        |↑  |0.3743|±  |0.0370|
| - humanities                          |      2|none             |      |acc        |↑  |0.3590|±  |0.0769|
|  - formal_logic                       |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - high_school_european_history       |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - high_school_us_history             |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - high_school_world_history          |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - international_law                  |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - jurisprudence                      |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - logical_fallacies                  |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - moral_disputes                     |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - moral_scenarios                    |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - philosophy                         |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - prehistory                         |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - professional_law                   |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - world_religions                    |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
| - other                               |      2|none             |      |acc        |↑  |0.4103|±  |0.0850|
|  - business_ethics                    |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - clinical_knowledge                 |      1|none             |     0|acc        |↑  |1.0000|±  |0.0000|
|  - college_medicine                   |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - global_facts                       |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - human_aging                        |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - management                         |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - marketing                          |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - medical_genetics                   |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - miscellaneous                      |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - nutrition                          |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - professional_accounting            |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - professional_medicine              |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - virology                           |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
| - social sciences                     |      2|none             |      |acc        |↑  |0.3889|±  |0.0833|
|  - econometrics                       |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - high_school_geography              |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - high_school_government_and_politics|      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_macroeconomics         |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_microeconomics         |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - high_school_psychology             |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - human_sexuality                    |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - professional_psychology            |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - public_relations                   |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - security_studies                   |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - sociology                          |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - us_foreign_policy                  |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
| - stem                                |      2|none             |      |acc        |↑  |0.3509|±  |0.0582|
|  - abstract_algebra                   |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - anatomy                            |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - astronomy                          |      1|none             |     0|acc        |↑  |1.0000|±  |0.0000|
|  - college_biology                    |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - college_chemistry                  |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - college_computer_science           |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - college_mathematics                |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - college_physics                    |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - computer_security                  |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - conceptual_physics                 |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - electrical_engineering             |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - elementary_mathematics             |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_biology                |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - high_school_chemistry              |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - high_school_computer_science       |      1|none             |     0|acc        |↑  |1.0000|±  |0.0000|
|  - high_school_mathematics            |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - high_school_physics                |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - high_school_statistics             |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - machine_learning                   |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|

〄 Testing lm-eval on collapsed model...
Starting lm-evaluation-harness on: ['mmlu', 'gpqa_main_zeroshot', 'gsm8k', 'mgsm_direct_zh']
Evaluating generation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:13<00:00,  2.17s/it]
Evaluating loglikelihood: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 696/696 [00:36<00:00, 19.24it/s]
|                 Tasks                 |Version|     Filter      |n-shot|  Metric   |   |Value |   |Stderr|
|---------------------------------------|------:|-----------------|-----:|-----------|---|-----:|---|-----:|
|gpqa_main_zeroshot                     |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|                                       |       |none             |     0|acc_norm   |↑  |0.3333|±  |0.3333|
|gsm8k                                  |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_zh                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mmlu                                   |      2|none             |      |acc        |↑  |0.2573|±  |0.0331|
| - humanities                          |      2|none             |      |acc        |↑  |0.1795|±  |0.0628|
|  - formal_logic                       |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_european_history       |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_us_history             |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_world_history          |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - international_law                  |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - jurisprudence                      |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - logical_fallacies                  |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - moral_disputes                     |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - moral_scenarios                    |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - philosophy                         |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - prehistory                         |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - professional_law                   |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - world_religions                    |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
| - other                               |      2|none             |      |acc        |↑  |0.3590|±  |0.0769|
|  - business_ethics                    |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - clinical_knowledge                 |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - college_medicine                   |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - global_facts                       |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - human_aging                        |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - management                         |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - marketing                          |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - medical_genetics                   |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - miscellaneous                      |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - nutrition                          |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - professional_accounting            |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - professional_medicine              |      1|none             |     0|acc        |↑  |1.0000|±  |0.0000|
|  - virology                           |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
| - social sciences                     |      2|none             |      |acc        |↑  |0.2500|±  |0.0621|
|  - econometrics                       |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_geography              |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_government_and_politics|      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - high_school_macroeconomics         |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_microeconomics         |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - high_school_psychology             |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - human_sexuality                    |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - professional_psychology            |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - public_relations                   |      1|none             |     0|acc        |↑  |1.0000|±  |0.0000|
|  - security_studies                   |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - sociology                          |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - us_foreign_policy                  |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
| - stem                                |      2|none             |      |acc        |↑  |0.2456|±  |0.0608|
|  - abstract_algebra                   |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - anatomy                            |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - astronomy                          |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - college_biology                    |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - college_chemistry                  |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - college_computer_science           |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - college_mathematics                |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - college_physics                    |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - computer_security                  |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - conceptual_physics                 |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - electrical_engineering             |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - elementary_mathematics             |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - high_school_biology                |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - high_school_chemistry              |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_computer_science       |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_mathematics            |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - high_school_physics                |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - high_school_statistics             |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - machine_learning                   |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|

〄 Testing lm-eval on healed model...
teacher_to_student=[(16, 13)]
epoch=    0 avg_loss=    0.46 elp_train=   21.29
└ test output: ["I'm sorry, I can't help with that. It's a medium red circle. Let me"]
epoch=    1 avg_loss=    0.45 elp_train=   21.53
└ test output: ["I'm sorry, I can't help with that. It's a medium red circle. Let me"]
Starting lm-evaluation-harness on: ['mmlu', 'gpqa_main_zeroshot', 'gsm8k', 'mgsm_direct_zh']
Evaluating generation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:10<00:00,  1.73s/it]
Evaluating loglikelihood: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 696/696 [00:49<00:00, 13.99it/s]
|                 Tasks                 |Version|     Filter      |n-shot|  Metric   |   |Value |   |Stderr|
|---------------------------------------|------:|-----------------|-----:|-----------|---|-----:|---|-----:|
|gpqa_main_zeroshot                     |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|                                       |       |none             |     0|acc_norm   |↑  |0.0000|±  |0.0000|
|gsm8k                                  |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_zh                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mmlu                                   |      2|none             |      |acc        |↑  |0.2865|±  |0.0336|
| - humanities                          |      2|none             |      |acc        |↑  |0.2564|±  |0.0725|
|  - formal_logic                       |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_european_history       |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - high_school_us_history             |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - high_school_world_history          |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - international_law                  |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - jurisprudence                      |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - logical_fallacies                  |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - moral_disputes                     |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - moral_scenarios                    |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - philosophy                         |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - prehistory                         |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - professional_law                   |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - world_religions                    |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
| - other                               |      2|none             |      |acc        |↑  |0.3333|±  |0.0769|
|  - business_ethics                    |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - clinical_knowledge                 |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - college_medicine                   |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - global_facts                       |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - human_aging                        |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - management                         |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - marketing                          |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - medical_genetics                   |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - miscellaneous                      |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - nutrition                          |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - professional_accounting            |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - professional_medicine              |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - virology                           |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
| - social sciences                     |      2|none             |      |acc        |↑  |0.2500|±  |0.0680|
|  - econometrics                       |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_geography              |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - high_school_government_and_politics|      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_macroeconomics         |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_microeconomics         |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_psychology             |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - human_sexuality                    |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - professional_psychology            |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - public_relations                   |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - security_studies                   |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - sociology                          |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - us_foreign_policy                  |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
| - stem                                |      2|none             |      |acc        |↑  |0.2982|±  |0.0555|
|  - abstract_algebra                   |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - anatomy                            |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - astronomy                          |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - college_biology                    |      1|none             |     0|acc        |↑  |1.0000|±  |0.0000|
|  - college_chemistry                  |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - college_computer_science           |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - college_mathematics                |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - college_physics                    |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - computer_security                  |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - conceptual_physics                 |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - electrical_engineering             |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - elementary_mathematics             |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_biology                |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - high_school_chemistry              |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - high_school_computer_science       |      1|none             |     0|acc        |↑  |0.6667|±  |0.3333|
|  - high_school_mathematics            |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - high_school_physics                |      1|none             |     0|acc        |↑  |0.3333|±  |0.3333|
|  - high_school_statistics             |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - machine_learning                   |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|

Original:
✓ GPQA                :  0.33
✓ GSM8k               :  1.00
✓ MGSM                :  0.67
✓ MMLU                :  0.37
Collapsed:
✓ GPQA                :  0.33
✓ GSM8k               :  0.00
✓ MGSM                :  0.00
✓ MMLU                :  0.26
Healed:
✓ GPQA                :  0.00
✓ GSM8k               :  0.00
✓ MGSM                :  0.00
✓ MMLU                :  0.29
```
</pre></details><br>
