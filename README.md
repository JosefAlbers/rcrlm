# rcr-lm

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XYyO04by5UCaMoH0C2MUmsLPOwddN5Gp?usp=sharing)

A lightweight, high-performance research framework for Large Language Models built on Apple's [MLX](https://github.com/ml-explore/mlx).

## Quickstart

`rcr-lm` is available on PyPI:

```bash
# macOS (Apple Silicon)
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
┌────────────────────────────── Streaming ──────────────────────────────┐
<think>
Okay, the user wants a story about Einstein. Let me start by recalling Einstein's life. He was a genius, a scientist, and a philosopher. I need to make sure the story includes his contributions to science, maybe his work on relativity, and his personal life.

First, I should set the scene. Maybe start with his early life in Germany, where he was born. Then introduce his family, his parents, maybe his mother's influence. Then his education, the famous lectures, and his breakthroughs.

I need to highlight his scientific achievements, like the theory of relativity. Also, his personal struggles, like the time he spent in the Alps, the Alps being a place of isolation and inspiration.

I should include some quotes or references to his work. Maybe mention his quote about the universe being infinite. Also, his later years and how he passed away.

Wait, the user might want the story to be engaging and highlight his legacy. I need to make sure the story flows well, with a good narrative arc. Avoid clichÃ©s, but still capture his essence. Check for any inaccuracies, like his actual birth date and death year. Let me confirm: Einstein was born on April 14, 18
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
                                                                                                                                                                                                                Wait, the user might want the story to be engaging and highlight his legacy. I need to make sure the story flows well, with a good narrative arc. Avoid clichÃ©s, but still capture his essence. Check for any inaccuracies, like his actual birth date and death year. Let me confirm: Einstein was born on April 14, 18
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    237.8 tokens/sec ( 14 tokens in 0.1s)
Tokens generation:    200.3 tokens/sec (256 tokens in 1.3s)
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

### Efficient Fine-Tuning

Supports DoRA (Weight-Decomposed Low-Rank Adaptation) for parameter-efficient training workflows.

```python
from rcrlm import load, train, infer
m = load()
lora_test_path = 'test_lora.safetensors'
train("RandomNameAnd6/SVGenerator", **m, lora_cfg=dict(wt_to=lora_test_path))
del m
m = load()
_ = infer("medium red circle\n", **m, lora_path=lora_test_path, stream=False, max_new_tokens=256, use_jit=False)
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
eval_lm(**m, tasks=["mmlu", "gpqa", "gsm8k", "mgsm_direct", "mbpp", "humaneval"])
m['model'] = collapse(m['model'])
eval_lm(**m, tasks=["mmlu", "gpqa", "gsm8k", "mgsm_direct", "mbpp", "humaneval"])
teacher = load()['model']
m['model'] = distill("HuggingFaceH4/instruction-dataset", **m, teacher=teacher)
eval_lm(**m, tasks=["mmlu", "gpqa", "gsm8k", "mgsm_direct", "mbpp", "humaneval"])
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

Wait, the user might want the story to be engaging and highlight his legacy. I need to make sure the story flows well, with a good narrative arc. Avoid clichÃ©s, but still capture his essence. Check for any inaccuracies, like his actual birth date and death year. Let me confirm: Einstein was born on April 14, 18
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

Wait, the user might want the story to be engaging and highlight his legacy. I need to make sure the story flows well, with a good narrative arc. Avoid clichÃ©s, but still capture his essence. Check for any inaccuracies, like his actual birth date and death year. Let me confirm: Einstein was born on April 14, 18
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    245.6 tokens/sec ( 14 tokens in 0.1s)
Tokens generation:    200.6 tokens/sec (256 tokens in 1.3s)
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
Prompt processing:   1405.0 tokens/sec ( 72 tokens in 0.1s)
Tokens generation:    474.6 tokens/sec (400 tokens in 0.8s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing DoRA training...
epoch=    0 avg_loss=    0.32 elp_train=    9.69
└ test output: ['<svg width="100" height="100" viewBox="-50 -5']
epoch=    1 avg_loss=    0.05 elp_train=    9.66
└ test output: ['<svg width="100" height="100" viewBox="-50 -5']
〄 Testing DoRA decoding...
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
medium red circle
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐
<svg width="100" height="100" viewBox="-50 -50 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="0" cy="0" r="24" fill="#f6f4f5"/></svg>
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    358.0 tokens/sec ( 12 tokens in 0.0s)
Tokens generation:    139.3 tokens/sec (256 tokens in 1.8s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing collapse...
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
Write a story about Einstein
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐
<think>
Okay, the user wants me to write a story about Einstein. Let me start by thinking about the main character. Einstein is a famous scientist, so I need to pick a character who's relatable. The user might be looking for a story that's both intellectual and human. Let me brainstorm some ideas. The story should be engaging, so I need to pick a specific topic. Maybe start with a relatable scenario. The user might want to explore different aspects of Einstein's life.
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    481.1 tokens/sec ( 14 tokens in 0.0s)
Tokens generation:    239.5 tokens/sec (100 tokens in 0.4s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing healing...
epoch=    0 avg_loss=    0.83 elp_train=   19.72
└ test output: ['<think>\nOkay, the user is asking for a medium red circle. Let me start by understanding the']
epoch=    1 avg_loss=    0.85 elp_train=   19.52
└ test output: ['<think>\nOkay, the user is asking for a medium red circle. Let me think about how to']
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
Write a story about Einstein
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐
<think>
Okay, the user wants a story about Einstein. Let me start by thinking about the key points Einstein had. He was a genius, so I need to highlight his contributions to science. Maybe start with his early life, his work on the EPR paradox. Then move to his later years. I should mention his famous equations and the EPR paradox. Also, his work on relativity and quantum mechanics. Maybe include some quotes from his time. Then wrap up with his legacy. Make
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    393.8 tokens/sec ( 14 tokens in 0.0s)
Tokens generation:    188.9 tokens/sec (100 tokens in 0.5s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing lm-eval on original model...
Starting lm-evaluation-harness on: ['mmlu', 'gpqa', 'gsm8k', 'mgsm_direct']
|                 Tasks                 |Version|     Filter      |n-shot|  Metric   |   |Value |   |Stderr|
|---------------------------------------|------:|-----------------|-----:|-----------|---|-----:|---|-----:|
|gpqa_diamond_cot_n_shot                |      2|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_diamond_cot_zeroshot              |      1|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_diamond_generative_n_shot         |      2|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_diamond_n_shot                    |      2|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|                                       |       |none             |     0|acc_norm   |↑  |0.2000|±  |0.1333|
|gpqa_diamond_zeroshot                  |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|                                       |       |none             |     0|acc_norm   |↑  |0.1000|±  |0.1000|
|gpqa_extended_cot_n_shot               |      2|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_extended_cot_zeroshot             |      1|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_extended_generative_n_shot        |      2|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_extended_n_shot                   |      2|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|                                       |       |none             |     0|acc_norm   |↑  |0.2000|±  |0.1333|
|gpqa_extended_zeroshot                 |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|                                       |       |none             |     0|acc_norm   |↑  |0.4000|±  |0.1633|
|gpqa_main_cot_n_shot                   |      2|flexible-extract |     0|exact_match|↑  |0.3000|±  |0.1528|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_main_cot_zeroshot                 |      1|flexible-extract |     0|exact_match|↑  |0.2000|±  |0.1333|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_main_generative_n_shot            |      2|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_main_n_shot                       |      2|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|                                       |       |none             |     0|acc_norm   |↑  |0.1000|±  |0.1000|
|gpqa_main_zeroshot                     |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|                                       |       |none             |     0|acc_norm   |↑  |0.2000|±  |0.1333|
|gsm8k                                  |      3|flexible-extract |     5|exact_match|↑  |0.2000|±  |0.1333|
|                                       |       |strict-match     |     5|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_bn                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_de                         |      3|flexible-extract |     0|exact_match|↑  |0.2000|±  |0.1333|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_en                         |      3|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_es                         |      3|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_es_spanish_bench           |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_fr                         |      3|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_ja                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_ru                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_sw                         |      3|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_te                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_th                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_zh                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mmlu                                   |      2|none             |      |acc        |↑  |0.4211|±  |0.0202|
| - humanities                          |      2|none             |      |acc        |↑  |0.4077|±  |0.0421|
|  - formal_logic                       |      1|none             |     0|acc        |↑  |0.6000|±  |0.1633|
|  - high_school_european_history       |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - high_school_us_history             |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - high_school_world_history          |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - international_law                  |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - jurisprudence                      |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - logical_fallacies                  |      1|none             |     0|acc        |↑  |0.6000|±  |0.1633|
|  - moral_disputes                     |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|  - moral_scenarios                    |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|  - philosophy                         |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - prehistory                         |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - professional_law                   |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - world_religions                    |      1|none             |     0|acc        |↑  |0.7000|±  |0.1528|
| - other                               |      2|none             |      |acc        |↑  |0.4692|±  |0.0434|
|  - business_ethics                    |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - clinical_knowledge                 |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - college_medicine                   |      1|none             |     0|acc        |↑  |0.6000|±  |0.1633|
|  - global_facts                       |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - human_aging                        |      1|none             |     0|acc        |↑  |0.6000|±  |0.1633|
|  - management                         |      1|none             |     0|acc        |↑  |0.7000|±  |0.1528|
|  - marketing                          |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - medical_genetics                   |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - miscellaneous                      |      1|none             |     0|acc        |↑  |0.6000|±  |0.1633|
|  - nutrition                          |      1|none             |     0|acc        |↑  |0.7000|±  |0.1528|
|  - professional_accounting            |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - professional_medicine              |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - virology                           |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
| - social sciences                     |      2|none             |      |acc        |↑  |0.4500|±  |0.0444|
|  - econometrics                       |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - high_school_geography              |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - high_school_government_and_politics|      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - high_school_macroeconomics         |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - high_school_microeconomics         |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - high_school_psychology             |      1|none             |     0|acc        |↑  |0.7000|±  |0.1528|
|  - human_sexuality                    |      1|none             |     0|acc        |↑  |0.7000|±  |0.1528|
|  - professional_psychology            |      1|none             |     0|acc        |↑  |0.6000|±  |0.1633|
|  - public_relations                   |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - security_studies                   |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - sociology                          |      1|none             |     0|acc        |↑  |0.7000|±  |0.1528|
|  - us_foreign_policy                  |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
| - stem                                |      2|none             |      |acc        |↑  |0.3789|±  |0.0341|
|  - abstract_algebra                   |      1|none             |     0|acc        |↑  |0.6000|±  |0.1633|
|  - anatomy                            |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - astronomy                          |      1|none             |     0|acc        |↑  |0.7000|±  |0.1528|
|  - college_biology                    |      1|none             |     0|acc        |↑  |0.7000|±  |0.1528|
|  - college_chemistry                  |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - college_computer_science           |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - college_mathematics                |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - college_physics                    |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - computer_security                  |      1|none             |     0|acc        |↑  |0.7000|±  |0.1528|
|  - conceptual_physics                 |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - electrical_engineering             |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - elementary_mathematics             |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - high_school_biology                |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - high_school_chemistry              |      1|none             |     0|acc        |↑  |0.6000|±  |0.1633|
|  - high_school_computer_science       |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - high_school_mathematics            |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - high_school_physics                |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - high_school_statistics             |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|  - machine_learning                   |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|

〄 Testing lm-eval on collapsed model...
Starting lm-evaluation-harness on: ['mmlu', 'gpqa', 'gsm8k', 'mgsm_direct']
|                 Tasks                 |Version|     Filter      |n-shot|  Metric   |   |Value |   |Stderr|
|---------------------------------------|------:|-----------------|-----:|-----------|---|-----:|---|-----:|
|gpqa_diamond_cot_n_shot                |      2|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_diamond_cot_zeroshot              |      1|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_diamond_generative_n_shot         |      2|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_diamond_n_shot                    |      2|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|                                       |       |none             |     0|acc_norm   |↑  |0.1000|±  |0.1000|
|gpqa_diamond_zeroshot                  |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|                                       |       |none             |     0|acc_norm   |↑  |0.3000|±  |0.1528|
|gpqa_extended_cot_n_shot               |      2|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_extended_cot_zeroshot             |      1|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_extended_generative_n_shot        |      2|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_extended_n_shot                   |      2|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|                                       |       |none             |     0|acc_norm   |↑  |0.1000|±  |0.1000|
|gpqa_extended_zeroshot                 |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|                                       |       |none             |     0|acc_norm   |↑  |0.2000|±  |0.1333|
|gpqa_main_cot_n_shot                   |      2|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_main_cot_zeroshot                 |      1|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_main_generative_n_shot            |      2|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_main_n_shot                       |      2|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|                                       |       |none             |     0|acc_norm   |↑  |0.1000|±  |0.1000|
|gpqa_main_zeroshot                     |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|                                       |       |none             |     0|acc_norm   |↑  |0.3000|±  |0.1528|
|gsm8k                                  |      3|flexible-extract |     5|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |strict-match     |     5|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_bn                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_de                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_en                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_es                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_es_spanish_bench           |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_fr                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_ja                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_ru                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_sw                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_te                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_th                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_zh                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mmlu                                   |      2|none             |      |acc        |↑  |0.3035|±  |0.0190|
| - humanities                          |      2|none             |      |acc        |↑  |0.2769|±  |0.0375|
|  - formal_logic                       |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|  - high_school_european_history       |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|  - high_school_us_history             |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - high_school_world_history          |      1|none             |     0|acc        |↑  |0.7000|±  |0.1528|
|  - international_law                  |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - jurisprudence                      |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - logical_fallacies                  |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - moral_disputes                     |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - moral_scenarios                    |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|  - philosophy                         |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - prehistory                         |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - professional_law                   |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - world_religions                    |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
| - other                               |      2|none             |      |acc        |↑  |0.3308|±  |0.0413|
|  - business_ethics                    |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - clinical_knowledge                 |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - college_medicine                   |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - global_facts                       |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - human_aging                        |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - management                         |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|  - marketing                          |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - medical_genetics                   |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - miscellaneous                      |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - nutrition                          |      1|none             |     0|acc        |↑  |0.6000|±  |0.1633|
|  - professional_accounting            |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - professional_medicine              |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - virology                           |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
| - social sciences                     |      2|none             |      |acc        |↑  |0.2917|±  |0.0409|
|  - econometrics                       |      1|none             |     0|acc        |↑  |0.6000|±  |0.1633|
|  - high_school_geography              |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - high_school_government_and_politics|      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - high_school_macroeconomics         |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_microeconomics         |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|  - high_school_psychology             |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - human_sexuality                    |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - professional_psychology            |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - public_relations                   |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - security_studies                   |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - sociology                          |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - us_foreign_policy                  |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
| - stem                                |      2|none             |      |acc        |↑  |0.3105|±  |0.0333|
|  - abstract_algebra                   |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|  - anatomy                            |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - astronomy                          |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - college_biology                    |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - college_chemistry                  |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - college_computer_science           |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - college_mathematics                |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - college_physics                    |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - computer_security                  |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - conceptual_physics                 |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - electrical_engineering             |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - elementary_mathematics             |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - high_school_biology                |      1|none             |     0|acc        |↑  |0.6000|±  |0.1633|
|  - high_school_chemistry              |      1|none             |     0|acc        |↑  |0.7000|±  |0.1528|
|  - high_school_computer_science       |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - high_school_mathematics            |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - high_school_physics                |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - high_school_statistics             |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|  - machine_learning                   |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|

〄 Testing lm-eval on healed model...
epoch=    0 avg_loss=    0.85 elp_train=   19.24
└ test output: ['<think>\nOkay, the user is asking for a medium red circle. Let me start by understanding the']
epoch=    1 avg_loss=    0.82 elp_train=   19.16
└ test output: ['<think>\nOkay, the user is asking for a medium red circle. Let me think about how to']
Starting lm-evaluation-harness on: ['mmlu', 'gpqa', 'gsm8k', 'mgsm_direct']
|                 Tasks                 |Version|     Filter      |n-shot|  Metric   |   |Value |   |Stderr|
|---------------------------------------|------:|-----------------|-----:|-----------|---|-----:|---|-----:|
|gpqa_diamond_cot_n_shot                |      2|flexible-extract |     0|exact_match|↑  |0.2000|±  |0.1333|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_diamond_cot_zeroshot              |      1|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_diamond_generative_n_shot         |      2|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_diamond_n_shot                    |      2|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|                                       |       |none             |     0|acc_norm   |↑  |0.1000|±  |0.1000|
|gpqa_diamond_zeroshot                  |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|                                       |       |none             |     0|acc_norm   |↑  |0.3000|±  |0.1528|
|gpqa_extended_cot_n_shot               |      2|flexible-extract |     0|exact_match|↑  |0.2000|±  |0.1333|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_extended_cot_zeroshot             |      1|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_extended_generative_n_shot        |      2|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_extended_n_shot                   |      2|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|                                       |       |none             |     0|acc_norm   |↑  |0.2000|±  |0.1333|
|gpqa_extended_zeroshot                 |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|                                       |       |none             |     0|acc_norm   |↑  |0.2000|±  |0.1333|
|gpqa_main_cot_n_shot                   |      2|flexible-extract |     0|exact_match|↑  |0.3000|±  |0.1528|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_main_cot_zeroshot                 |      1|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_main_generative_n_shot            |      2|flexible-extract |     0|exact_match|↑  |0.2000|±  |0.1333|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|gpqa_main_n_shot                       |      2|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|                                       |       |none             |     0|acc_norm   |↑  |0.2000|±  |0.1333|
|gpqa_main_zeroshot                     |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|                                       |       |none             |     0|acc_norm   |↑  |0.3000|±  |0.1528|
|gsm8k                                  |      3|flexible-extract |     5|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |strict-match     |     5|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_bn                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_de                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_en                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_es                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_es_spanish_bench           |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_fr                         |      3|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_ja                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_ru                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_sw                         |      3|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.1000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_te                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_th                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_zh                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mmlu                                   |      2|none             |      |acc        |↑  |0.2930|±  |0.0190|
| - humanities                          |      2|none             |      |acc        |↑  |0.2462|±  |0.0379|
|  - formal_logic                       |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - high_school_european_history       |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|  - high_school_us_history             |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - high_school_world_history          |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - international_law                  |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - jurisprudence                      |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - logical_fallacies                  |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - moral_disputes                     |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|  - moral_scenarios                    |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - philosophy                         |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|  - prehistory                         |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|  - professional_law                   |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - world_religions                    |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
| - other                               |      2|none             |      |acc        |↑  |0.3154|±  |0.0414|
|  - business_ethics                    |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - clinical_knowledge                 |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - college_medicine                   |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - global_facts                       |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - human_aging                        |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - management                         |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - marketing                          |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - medical_genetics                   |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - miscellaneous                      |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - nutrition                          |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - professional_accounting            |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - professional_medicine              |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|  - virology                           |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
| - social sciences                     |      2|none             |      |acc        |↑  |0.3500|±  |0.0439|
|  - econometrics                       |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - high_school_geography              |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - high_school_government_and_politics|      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - high_school_macroeconomics         |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - high_school_microeconomics         |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|  - high_school_psychology             |      1|none             |     0|acc        |↑  |0.6000|±  |0.1633|
|  - human_sexuality                    |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - professional_psychology            |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - public_relations                   |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - security_studies                   |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - sociology                          |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - us_foreign_policy                  |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
| - stem                                |      2|none             |      |acc        |↑  |0.2737|±  |0.0320|
|  - abstract_algebra                   |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - anatomy                            |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - astronomy                          |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|  - college_biology                    |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - college_chemistry                  |      1|none             |     0|acc        |↑  |0.1000|±  |0.1000|
|  - college_computer_science           |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - college_mathematics                |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - college_physics                    |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - computer_security                  |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - conceptual_physics                 |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - electrical_engineering             |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - elementary_mathematics             |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_biology                |      1|none             |     0|acc        |↑  |0.4000|±  |0.1633|
|  - high_school_chemistry              |      1|none             |     0|acc        |↑  |0.5000|±  |0.1667|
|  - high_school_computer_science       |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - high_school_mathematics            |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
|  - high_school_physics                |      1|none             |     0|acc        |↑  |0.3000|±  |0.1528|
|  - high_school_statistics             |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - machine_learning                   |      1|none             |     0|acc        |↑  |0.2000|±  |0.1333|
```
</pre></details><br>
