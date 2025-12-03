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

`rcr-lm` achieves generation speeds exceeding 200 tokens/sec, offering a measurable performance uplift over standard LLM implementations.

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

#### transformers (for comparison)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

prompt = "Write a story about Einstein"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

tic_gen = time.perf_counter()
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=256
)
elp_gen = time.perf_counter() - tic_gen

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=False).strip("\n")
print(content)
print('==========')
print(f'{model_inputs['input_ids'].shape[1]} prompt tokens and {len(output_ids)} generated tokens in {elp_gen:.2f} seconds')
```

```
<think>
Okay, the user wants me to write a story about Einstein. Let me start by thinking about the key elements that make Einstein a famous figure. He's a genius, a scientist, and a man of philosophy. I need to make sure the story captures these aspects.

First, I should set the scene. Maybe start with his early life to show his early brilliance. His childhood, maybe a small village, and his parents. Then introduce his academic journey, the famous equations, and his work on the theory of relativity. Including his personal life, like his family and his later life, would add depth.

I need to include some conflict or challenge to make the story engaging. Perhaps a time when he faced opposition or personal struggles. Maybe his work on the theory of relativity was controversial, which adds tension. Also, highlighting his personality traits—like his curiosity, determination, and the impact he had on society.

I should make sure the story has a clear beginning, middle, and end. Start with his early achievements, then his scientific contributions, and end with his legacy. Avoid clichés, but still capture his essence. Maybe include some quotes or references to his work to add authenticity.

Wait, the user might be looking for a story that's both
==========
13 prompt tokens and 256 generated tokens in 10.66 seconds
```

*Note: Included for baseline reference. transformers is not fully optimized for inference on Apple Silicon (MPS) compared to its performance on NVIDIA GPUs.*

### Batched Decoding

Batched decoding is natively supported.

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

Also supports KD (Knowledge Distillation) from a teacher model.

```python
m = load()
m['model'] = collapse(m['model'])
# _ = infer("Write a story about Einstein\n", **m, stream=False)
teacher = load()['model']
m['model'] = distill("HuggingFaceH4/instruction-dataset", **m, teacher=teacher)
_ = infer("Write a story about Einstein\n", **m, stream=False, max_new_tokens=1024)
```

```
epoch=    0 avg_loss=    1.78 elp_train=   49.62
└ test output: ["I'm sorry, I can't help with that. It's a medium red circle, but I"]
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
Write a story about Einstein
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐
<think>
Okay, so I need to write a story about Einstein. Let me think about the key elements. Einstein was a famous scientist, right? He was known for his work in relativity and quantum mechanics. The story should highlight his contributions, maybe his personal life, and his impact on science. Let me outline the main points: his early life, his work, and his legacy. Let me think about the structure. Maybe start with his early life, his work, and his legacy. Then maybe add some details about his personal life, like his family, his scientific achievements, and his impact on the world. Let me make sure to include his scientific contributions and his legacy. Also, maybe add some details about his personal life, like his relationship with his family, his struggles, and his personal achievements. Let me think about the flow. Start with his early life, then his work, then his legacy, and end with his impact. Maybe add some details about his personal life, like his relationship with his family, his struggles, and his personal achievements. Let me make sure to include all the key points. Let me check if all the points are covered and if the story is engaging. I think that's a good start for a story about Einstein.
</think>

**Einstein's Legacy**

In the quiet halls of the University of Zurich, Einstein's name is etched in history. A brilliant mind who once wrote the equations that changed the course of modern physics. His story is one of both wonder and challenge. Born in 1879, he was a prodigy who made groundbreaking contributions to the field of physics. His work in relativity and quantum mechanics laid the foundation for the understanding of space and time. His personal life was as complex as his scientific achievements. He was known for his quiet and introspective nature, often reflecting on the meaning of life. His family was a source of strength and support. He was known for his quiet and thoughtful nature, and his work in physics was both revolutionary. His legacy is still recognized and celebrated. He is remembered for his contributions to science and his personal life. His work continues to influence the world. He is still remembered for his scientific achievements and his personal life. His legacy is still celebrated and respected. He is remembered for his contributions to science and his personal life. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected. His legacy is still celebrated and respected.
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    297.8 tokens/sec ( 14 tokens in 0.0s)
Tokens generation:    105.7 tokens/sec (1024 tokens in 9.7s)
└───────────────────────────────────────────────────────────────────────┘
```

### Integrated Evaluation

Native integration with `lm-evaluation-harness`. Benchmark vanilla and customized models against standard metrics (MMLU, GSM8K, etc.) with a single command.

```python
from rcrlm.evals import eval_lm
m = load()
e_orgn = eval_lm(**m, chat_template_kwargs=dict(enable_thinking=False))
m['model'] = collapse(m['model'])
e_coll = eval_lm(**m, chat_template_kwargs=dict(enable_thinking=False))
teacher = load()['model']
m['model'] = distill("HuggingFaceH4/instruction-dataset", **m, teacher=teacher)
e_heal = eval_lm(**m, chat_template_kwargs=dict(enable_thinking=False))
print(f'Original:\n{e_orgn}\nCollapsed:\n{e_coll}\nHealed:\n{e_heal}')
```

```
✓ Original:
- GPQA                :  0.40
- GSM8k               :  0.30
- MGSM                :  0.50
- MMLU                :  0.42
✓ Collapsed:
- GPQA                :  0.25
- GSM8k               :  0.00
- MGSM                :  0.00
- MMLU                :  0.25
✓ Healed:
- GPQA                :  0.25
- GSM8k               :  0.10
- MGSM                :  0.05
- MMLU                :  0.31
✓ Dampened:
- GPQA                :  0.25
- GSM8k               :  0.10
- MGSM                :  0.05
- MMLU                :  0.30
```

<details><summary>Codes adapted from [nnx-lm](https://pypi.org/project/nnx-lm/) to try some stuff</summary><pre>
## Etc

```
~/D/rcr[1 jobs]> python -m rcrlm.main
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
Prompt processing:    251.5 tokens/sec ( 14 tokens in 0.1s)
Tokens generation:    200.0 tokens/sec (256 tokens in 1.3s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing batch decoding...
┌────────────────────────────── Streaming ──────────────────────────────┐
<think>
Okay, the user is asking for a comparison between the Sortino Ratio for Bitcoin and Ethereum. Let me start by recalling what the Sortino Ratio is. It's a measure of risk-adjusted return for a portfolio, right? It's calculated as (Return - Risk-Free Rate) divided by the Risk (Standard Deviation). The Sortino Ratio is usually used for a single asset, so comparing it between Bitcoin and Ethereum would be useful.

First, I need to check the historical data
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user                                                                                                                                                                                                    #write a quick sort algorithm
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
<think>                                                                                                                                                                                                             Okay, the user wants a short introduction to a large language model. Let me start by recalling what I know about LLMs. They're big language models, right? So I should mention their ability to understand and generate text. Maybe start with the basics: they can process and generate text, not just a few words. Then explain their training data, like the amount of text they're trained on. Also, their capabilities: understanding and generating text, answering questions, etc. Need                                                                                                                                                              └───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Inp 00002 ──────────────────────────────┐
<|im_start|>user
Write a neurology ICU admission note.
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘                                                                                                                                           ┌────────────────────────────── Out 00002 ──────────────────────────────┐
<think>
Okay, I need to write a neurology ICU admission note. Let me start by recalling what an ICU admission note typically includes. It's a medical record that outlines the patient's condition, initial assessment, interventions, and any ongoing care.

First, the patient's name and date of admission. I should make sure to include that. Then, the patient's name, age, gender, and primary diagnosis. Since it's a neurology ICU, the main issue is likely a neurological
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Inp 00003 ──────────────────────────────┐
<|im_start|>user
Comparison of Sortino Ratio for Bitcoin and Ethereum.<|im_end|>                                                                                                                                                     <|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00003 ──────────────────────────────┐                                                                                                                                           <think>
Okay, the user is asking for a comparison between the Sortino Ratio for Bitcoin and Ethereum. Let me start by recalling what the Sortino Ratio is. It's a measure of risk-adjusted return for a portfolio, right? It's calculated as (Return - Risk-Free Rate) divided by the Risk (Standard Deviation). The Sortino Ratio is usually used for a single asset, so comparing it between Bitcoin and Ethereum would be useful.

First, I need to check the historical data
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐                                                                                                                                           Prompt processing:   1243.0 tokens/sec ( 72 tokens in 0.1s)
Tokens generation:    473.6 tokens/sec (400 tokens in 0.8s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing DoRA training...
epoch=    0 avg_loss=    0.27 elp_train=   11.94
└ test output: ['<svg width="100" height="100" viewBox="-50 -5']
epoch=    1 avg_loss=    0.04 elp_train=   12.18
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
<svg width="100" height="100" viewBox="-50 -50 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="0" cy="0" r="28" fill="#f6f5f4"/></svg>
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    384.1 tokens/sec ( 15 tokens in 0.0s)
Tokens generation:    163.9 tokens/sec (256 tokens in 1.6s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing collapse...
✓ Colapsed:
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
Write a story about Einstein
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐
</think>
**The Story of Einstein: A Brief version**
*by: The story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    475.5 tokens/sec ( 14 tokens in 0.0s)
Tokens generation:    236.2 tokens/sec (100 tokens in 0.4s)
└───────────────────────────────────────────────────────────────────────┘
teacher_to_student=[(16, 13)]
layer_idx=13
epoch=    0 avg_loss=  315.04 elp_train=   84.63
└ test output: ["I'm sorry, I can't help with that. It's a medium red circle. Let me"]
✓ Healed:
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
Write a story about Einstein
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐
<think>
Okay, I need to write a story about Einstein. Let me think about the elements. First, the main character. Einstein is a famous scientist, so maybe a young person with a curious mind. The story should have elements like the discovery of the equation that led to the development of modern physics. Maybe include some key moments, like the invention of the theory of relativity. The story should have a beginning, middle, and end. Let me structure it with these elements. Let's start with Einstein's early life, his discovery of the equation, and his impact on science. Then maybe include some key events like the invention of the theory of relativity and the development of the space-time. The story should have a beginning, middle, and end. Let me make sure to include Einstein's journey and the impact of his work. I need to make sure the story is clear and engaging. Let me check if all the elements are there. Einstein's discovery of the equation, his work, and the impact on science. Maybe include some key moments like the invention of the theory of relativity and the development of space-time. The story should have a beginning, middle, and end. Let me make sure the story is clear and engaging. I need to make sure the story is about Einstein and his contributions to science. Let me think about the structure. Start with Einstein's early life, his discovery of the equation, and the development of space-time. Maybe include some key moments like the invention of the theory of relativity and the development of space-time. The story should have a beginning, middle, and end. Let me make sure the story is clear and engaging. I need to make sure the story is about Einstein and his contributions to science. Let me think about the elements. Einstein's discovery of the equation, his work, and the impact on science. Maybe include some key moments like the invention of the theory of relativity and the development of space-time. The story should have a beginning, middle, and end. Let me make sure the story is clear and engaging. I need to make sure the story is about Einstein and his contributions to science. Let me think about the elements. Einstein's discovery of the equation, his work, and the impact on science. Maybe include some key moments like the invention of the theory of relativity and the development of space-time. The story should have a beginning, middle, and end. Let me make sure the story is clear and engaging. I need to make sure the story is about Einstein and his contributions to science. Let me think about the elements. Einstein's discovery of the equation, his work, and the impact on science. Maybe include some key moments like the invention of the theory of relativity and the development of space-time. The story should have a beginning, middle, and end. Let me make sure the story is clear and engaging. I need to make sure the story is about Einstein and his contributions to science. Let me think about the elements. Einstein's discovery of the equation, his work, and the impact on science. Maybe include some key moments like the invention of the theory of relativity and the development of space-time. The story should have a beginning, middle, and end. Let me make sure the story is clear and engaging. I need to make sure the story is about Einstein and his contributions to science. Let me think about the elements. Einstein's discovery of the equation, his work, and the impact on science. Maybe include some key moments like the invention of the theory of relativity and the development of space-time. The story should have a beginning, middle, and end. Let me make sure the story is clear and engaging. I need to make sure the story is about Einstein and his contributions to science. Let me think about the elements. Einstein's discovery of the equation, his work, and the impact on science. Maybe include some key moments like the invention of the theory of relativity and the development of space-time. The story should have a beginning, middle, and end. Let me make sure the story is clear and engaging. I need to make sure the story is about Einstein and his contributions to science. Let me think about the elements. Einstein's discovery of the equation, his work, and the impact on science. Maybe include some key moments like the invention of the theory of relativity and the development of space-time. The story should have a beginning, middle, and end. Let me make sure the story is clear and engaging. I need to make sure the story is about Einstein and his contributions to science. Let me think about the elements. Einstein's discovery of the equation, his work, and the impact on science. Maybe include some key moments like the invention of the theory of relativity and the development of space-time. The story should have a beginning, middle, and end. Let me make sure the story is clear and engaging. I need to make sure the story is about Einstein and his contributions to science. Let me think about the
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    276.2 tokens/sec ( 14 tokens in 0.1s)
Tokens generation:    103.9 tokens/sec (1024 tokens in 9.9s)
└───────────────────────────────────────────────────────────────────────┘
[──────────────────────────────] Processing model.layers.13.layer.mlp.up_proj...
   -> No strong intruders found. Keeping exact weights.
[──────────────────────────────] Processing model.layers.13.layer.mlp.down_proj...
   -> No strong intruders found. Keeping exact weights.
[──────────────────────────────] Processing model.layers.13.layer.mlp.gate_proj...
   -> No strong intruders found. Keeping exact weights.
[──────────────────────────────] Processing model.layers.13.layer.self_attn.o_proj...
   -> No strong intruders found. Keeping exact weights.
[──────────────────────────────] Processing model.layers.13.layer.self_attn.v_proj...
   -> No strong intruders found. Keeping exact weights.
[──────────────────────────────] Processing model.layers.13.layer.self_attn.k_proj...
   -> No strong intruders found. Keeping exact weights.
[──────────────────────────────] Processing model.layers.13.layer.self_attn.q_proj...
   -> No strong intruders found. Keeping exact weights.
[──────────────────────────────] Successfully healed and merged 7 layers.
✓ Dampened:
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
Write a story about Einstein
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐
<think>
Okay, I need to write a story about Einstein. Let me think about the elements. First, I should set the scene. Maybe start with a normal day in a small town. The main character is a young boy who is curious and observant. The story should have elements like a simple life and a small town. The setting should be simple and relatable. The main character is a boy who is curious and observant. The story should have elements like a small town and a simple life. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should have elements like a small town and a curious boy who is observant. The story should
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    276.1 tokens/sec ( 14 tokens in 0.1s)
Tokens generation:    137.5 tokens/sec (1024 tokens in 7.4s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing lm-eval...
Starting lm-evaluation-harness on: ['mmlu', 'gpqa_main_zeroshot', 'gsm8k', 'mgsm_direct_zh']
Evaluating generation: 100%|██████████████████████████████████████████████| 40/40 [12:27<00:00, 18.68s/it]
Evaluating loglikelihood: 100%|███████████████████████████████████████| 4640/4640 [04:25<00:00, 17.48it/s]
|                 Tasks                 |Version|     Filter      |n-shot|  Metric   |   |Value |   |Stderr|
|---------------------------------------|------:|-----------------|-----:|-----------|---|-----:|---|-----:|
|gpqa_main_zeroshot                     |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|                                       |       |none             |     0|acc_norm   |↑  |0.4000|±  |0.1124|
|gsm8k                                  |      3|flexible-extract |     0|exact_match|↑  |0.3000|±  |0.1051|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_zh                         |      3|flexible-extract |     0|exact_match|↑  |0.5000|±  |0.1147|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mmlu                                   |      2|none             |      |acc        |↑  |0.4193|±  |0.0145|
| - humanities                          |      2|none             |      |acc        |↑  |0.4385|±  |0.0310|
|  - formal_logic                       |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - high_school_european_history       |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - high_school_us_history             |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - high_school_world_history          |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - international_law                  |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - jurisprudence                      |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - logical_fallacies                  |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - moral_disputes                     |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - moral_scenarios                    |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - philosophy                         |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - prehistory                         |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - professional_law                   |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - world_religions                    |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
| - other                               |      2|none             |      |acc        |↑  |0.4077|±  |0.0299|
|  - business_ethics                    |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - clinical_knowledge                 |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - college_medicine                   |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - global_facts                       |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - human_aging                        |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - management                         |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - marketing                          |      1|none             |     0|acc        |↑  |0.6000|±  |0.1124|
|  - medical_genetics                   |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - miscellaneous                      |      1|none             |     0|acc        |↑  |0.6000|±  |0.1124|
|  - nutrition                          |      1|none             |     0|acc        |↑  |0.6000|±  |0.1124|
|  - professional_accounting            |      1|none             |     0|acc        |↑  |0.1000|±  |0.0688|
|  - professional_medicine              |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - virology                           |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
| - social sciences                     |      2|none             |      |acc        |↑  |0.4292|±  |0.0309|
|  - econometrics                       |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - high_school_geography              |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - high_school_government_and_politics|      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - high_school_macroeconomics         |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - high_school_microeconomics         |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - high_school_psychology             |      1|none             |     0|acc        |↑  |0.6000|±  |0.1124|
|  - human_sexuality                    |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - professional_psychology            |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - public_relations                   |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - security_studies                   |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - sociology                          |      1|none             |     0|acc        |↑  |0.7000|±  |0.1051|
|  - us_foreign_policy                  |      1|none             |     0|acc        |↑  |0.6500|±  |0.1094|
| - stem                                |      2|none             |      |acc        |↑  |0.4079|±  |0.0251|
|  - abstract_algebra                   |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - anatomy                            |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - astronomy                          |      1|none             |     0|acc        |↑  |0.6000|±  |0.1124|
|  - college_biology                    |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - college_chemistry                  |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - college_computer_science           |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - college_mathematics                |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - college_physics                    |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - computer_security                  |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - conceptual_physics                 |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - electrical_engineering             |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - elementary_mathematics             |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - high_school_biology                |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - high_school_chemistry              |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - high_school_computer_science       |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - high_school_mathematics            |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_physics                |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - high_school_statistics             |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - machine_learning                   |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|

- GPQA                :  0.40
- GSM8k               :  0.30
- MGSM                :  0.50
- MMLU                :  0.42
Starting lm-evaluation-harness on: ['mmlu', 'gpqa_main_zeroshot', 'gsm8k', 'mgsm_direct_zh']
Evaluating generation: 100%|██████████████████████████████████████████████| 40/40 [26:10<00:00, 39.26s/it]
Evaluating loglikelihood: 100%|███████████████████████████████████████| 4640/4640 [04:08<00:00, 18.64it/s]
|                 Tasks                 |Version|     Filter      |n-shot|  Metric   |   |Value |   |Stderr|
|---------------------------------------|------:|-----------------|-----:|-----------|---|-----:|---|-----:|
|gpqa_main_zeroshot                     |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|                                       |       |none             |     0|acc_norm   |↑  |0.2500|±  |0.0993|
|gsm8k                                  |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_zh                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mmlu                                   |      2|none             |      |acc        |↑  |0.2509|±  |0.0128|
| - humanities                          |      2|none             |      |acc        |↑  |0.2154|±  |0.0257|
|  - formal_logic                       |      1|none             |     0|acc        |↑  |0.1000|±  |0.0688|
|  - high_school_european_history       |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - high_school_us_history             |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_world_history          |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - international_law                  |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - jurisprudence                      |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - logical_fallacies                  |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - moral_disputes                     |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - moral_scenarios                    |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - philosophy                         |      1|none             |     0|acc        |↑  |0.1000|±  |0.0688|
|  - prehistory                         |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - professional_law                   |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - world_religions                    |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
| - other                               |      2|none             |      |acc        |↑  |0.2577|±  |0.0272|
|  - business_ethics                    |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - clinical_knowledge                 |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - college_medicine                   |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - global_facts                       |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - human_aging                        |      1|none             |     0|acc        |↑  |0.1000|±  |0.0688|
|  - management                         |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - marketing                          |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - medical_genetics                   |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - miscellaneous                      |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - nutrition                          |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - professional_accounting            |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - professional_medicine              |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - virology                           |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
| - social sciences                     |      2|none             |      |acc        |↑  |0.2625|±  |0.0280|
|  - econometrics                       |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_geography              |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - high_school_government_and_politics|      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - high_school_macroeconomics         |      1|none             |     0|acc        |↑  |0.1000|±  |0.0688|
|  - high_school_microeconomics         |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - high_school_psychology             |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - human_sexuality                    |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - professional_psychology            |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - public_relations                   |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - security_studies                   |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - sociology                          |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - us_foreign_policy                  |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
| - stem                                |      2|none             |      |acc        |↑  |0.2632|±  |0.0227|
|  - abstract_algebra                   |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - anatomy                            |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - astronomy                          |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - college_biology                    |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - college_chemistry                  |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - college_computer_science           |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - college_mathematics                |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - college_physics                    |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - computer_security                  |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - conceptual_physics                 |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - electrical_engineering             |      1|none             |     0|acc        |↑  |0.1000|±  |0.0688|
|  - elementary_mathematics             |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - high_school_biology                |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - high_school_chemistry              |      1|none             |     0|acc        |↑  |0.1000|±  |0.0688|
|  - high_school_computer_science       |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - high_school_mathematics            |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - high_school_physics                |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_statistics             |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - machine_learning                   |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|

- GPQA                :  0.25
- GSM8k               :  0.00
- MGSM                :  0.00
- MMLU                :  0.25
teacher_to_student=[(16, 13)]
layer_idx=13
epoch=    0 avg_loss=  309.79 elp_train=  106.99
└ test output: ["I'm sorry, I can't help with that. It's a medium red circle. Let me"]
Starting lm-evaluation-harness on: ['mmlu', 'gpqa_main_zeroshot', 'gsm8k', 'mgsm_direct_zh']
Evaluating generation: 100%|██████████████████████████████████████████████| 40/40 [36:09<00:00, 54.24s/it]
Evaluating loglikelihood: 100%|███████████████████████████████████████| 4640/4640 [05:59<00:00, 12.92it/s]
|                 Tasks                 |Version|     Filter      |n-shot|  Metric   |   |Value |   |Stderr|
|---------------------------------------|------:|-----------------|-----:|-----------|---|-----:|---|-----:|
|gpqa_main_zeroshot                     |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|                                       |       |none             |     0|acc_norm   |↑  |0.2500|±  |0.0993|
|gsm8k                                  |      3|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.0688|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_zh                         |      3|flexible-extract |     0|exact_match|↑  |0.0500|±  |0.0500|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mmlu                                   |      2|none             |      |acc        |↑  |0.3079|±  |0.0135|
| - humanities                          |      2|none             |      |acc        |↑  |0.3038|±  |0.0282|
|  - formal_logic                       |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - high_school_european_history       |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - high_school_us_history             |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - high_school_world_history          |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - international_law                  |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - jurisprudence                      |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - logical_fallacies                  |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - moral_disputes                     |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - moral_scenarios                    |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - philosophy                         |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - prehistory                         |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - professional_law                   |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - world_religions                    |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
| - other                               |      2|none             |      |acc        |↑  |0.3192|±  |0.0290|
|  - business_ethics                    |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - clinical_knowledge                 |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - college_medicine                   |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - global_facts                       |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - human_aging                        |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - management                         |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - marketing                          |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - medical_genetics                   |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - miscellaneous                      |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - nutrition                          |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - professional_accounting            |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - professional_medicine              |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - virology                           |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
| - social sciences                     |      2|none             |      |acc        |↑  |0.2958|±  |0.0286|
|  - econometrics                       |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - high_school_geography              |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - high_school_government_and_politics|      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_macroeconomics         |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - high_school_microeconomics         |      1|none             |     0|acc        |↑  |0.0500|±  |0.0500|
|  - high_school_psychology             |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - human_sexuality                    |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - professional_psychology            |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - public_relations                   |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - security_studies                   |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - sociology                          |      1|none             |     0|acc        |↑  |0.6000|±  |0.1124|
|  - us_foreign_policy                  |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
| - stem                                |      2|none             |      |acc        |↑  |0.3105|±  |0.0235|
|  - abstract_algebra                   |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - anatomy                            |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - astronomy                          |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - college_biology                    |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - college_chemistry                  |      1|none             |     0|acc        |↑  |0.1000|±  |0.0688|
|  - college_computer_science           |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - college_mathematics                |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - college_physics                    |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - computer_security                  |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - conceptual_physics                 |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - electrical_engineering             |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - elementary_mathematics             |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_biology                |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_chemistry              |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - high_school_computer_science       |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - high_school_mathematics            |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - high_school_physics                |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - high_school_statistics             |      1|none             |     0|acc        |↑  |0.0500|±  |0.0500|
|  - machine_learning                   |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|

- GPQA                :  0.25
- GSM8k               :  0.10
- MGSM                :  0.05
- MMLU                :  0.31
[──────────────────────────────] Processing model.layers.13.layer.mlp.up_proj...
   -> No strong intruders found. Keeping exact weights.
[──────────────────────────────] Processing model.layers.13.layer.mlp.down_proj...
   -> No strong intruders found. Keeping exact weights.
[──────────────────────────────] Processing model.layers.13.layer.mlp.gate_proj...
   -> No strong intruders found. Keeping exact weights.
[──────────────────────────────] Processing model.layers.13.layer.self_attn.o_proj...
   -> No strong intruders found. Keeping exact weights.
[──────────────────────────────] Processing model.layers.13.layer.self_attn.v_proj...
   -> No strong intruders found. Keeping exact weights.
[──────────────────────────────] Processing model.layers.13.layer.self_attn.k_proj...
   -> No strong intruders found. Keeping exact weights.
[──────────────────────────────] Processing model.layers.13.layer.self_attn.q_proj...
   -> No strong intruders found. Keeping exact weights.
[──────────────────────────────] Successfully healed and merged 7 layers.
Starting lm-evaluation-harness on: ['mmlu', 'gpqa_main_zeroshot', 'gsm8k', 'mgsm_direct_zh']
Evaluating generation: 100%|██████████████████████████████████████████████| 40/40 [30:05<00:00, 45.13s/it]
Evaluating loglikelihood: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4640/4640 [04:17<00:00, 18.02it/s]
|                 Tasks                 |Version|     Filter      |n-shot|  Metric   |   |Value |   |Stderr|
|---------------------------------------|------:|-----------------|-----:|-----------|---|-----:|---|-----:|
|gpqa_main_zeroshot                     |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|                                       |       |none             |     0|acc_norm   |↑  |0.2500|±  |0.0993|
|gsm8k                                  |      3|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.0688|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_zh                         |      3|flexible-extract |     0|exact_match|↑  |0.0500|±  |0.0500|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mmlu                                   |      2|none             |      |acc        |↑  |0.3044|±  |0.0134|
| - humanities                          |      2|none             |      |acc        |↑  |0.2923|±  |0.0279|
|  - formal_logic                       |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - high_school_european_history       |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - high_school_us_history             |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - high_school_world_history          |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - international_law                  |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - jurisprudence                      |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - logical_fallacies                  |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - moral_disputes                     |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - moral_scenarios                    |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - philosophy                         |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - prehistory                         |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - professional_law                   |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - world_religions                    |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
| - other                               |      2|none             |      |acc        |↑  |0.3231|±  |0.0290|
|  - business_ethics                    |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - clinical_knowledge                 |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - college_medicine                   |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - global_facts                       |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - human_aging                        |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - management                         |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - marketing                          |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - medical_genetics                   |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - miscellaneous                      |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - nutrition                          |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - professional_accounting            |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - professional_medicine              |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - virology                           |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
| - social sciences                     |      2|none             |      |acc        |↑  |0.2958|±  |0.0282|
|  - econometrics                       |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_geography              |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - high_school_government_and_politics|      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_macroeconomics         |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - high_school_microeconomics         |      1|none             |     0|acc        |↑  |0.0000|±  |0.0000|
|  - high_school_psychology             |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - human_sexuality                    |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - professional_psychology            |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - public_relations                   |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - security_studies                   |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - sociology                          |      1|none             |     0|acc        |↑  |0.6500|±  |0.1094|
|  - us_foreign_policy                  |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
| - stem                                |      2|none             |      |acc        |↑  |0.3053|±  |0.0233|
|  - abstract_algebra                   |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - anatomy                            |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - astronomy                          |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - college_biology                    |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - college_chemistry                  |      1|none             |     0|acc        |↑  |0.1000|±  |0.0688|
|  - college_computer_science           |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - college_mathematics                |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - college_physics                    |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - computer_security                  |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - conceptual_physics                 |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - electrical_engineering             |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - elementary_mathematics             |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - high_school_biology                |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - high_school_chemistry              |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - high_school_computer_science       |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - high_school_mathematics            |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - high_school_physics                |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - high_school_statistics             |      1|none             |     0|acc        |↑  |0.0500|±  |0.0500|
|  - machine_learning                   |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|

- GPQA                :  0.25
- GSM8k               :  0.10
- MGSM                :  0.05
- MMLU                :  0.30
✓ Original:
- GPQA                :  0.40
- GSM8k               :  0.30
- MGSM                :  0.50
- MMLU                :  0.42
✓ Collapsed:
- GPQA                :  0.25
- GSM8k               :  0.00
- MGSM                :  0.00
- MMLU                :  0.25
✓ Healed:
- GPQA                :  0.25
- GSM8k               :  0.10
- MGSM                :  0.05
- MMLU                :  0.31
✓ Dampened:
- GPQA                :  0.25
- GSM8k               :  0.10
- MGSM                :  0.05
- MMLU                :  0.30

〄 Testing cascading...
teacher_to_student=[(14, 13)]
layer_idx=13
epoch=    0 avg_loss=   79.34 elp_train=   80.51
└ test output: ["I'm sorry, but I can't help with that request. If you have any questions, feel"]
teacher_to_student=[(16, 14)]
layer_idx=14
epoch=    0 avg_loss=  276.82 elp_train=   79.58
└ test output: ["I'm sorry, I can't help with that request. Let me know if you need any help"]
```
</pre></details><br>
