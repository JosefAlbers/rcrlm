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
Original:
✓ GPQA                :  0.35
✓ GSM8k               :  0.35
✓ MGSM                :  0.35
✓ MMLU                :  0.42
Collapsed:
✓ GPQA                :  0.25
✓ GSM8k               :  0.00
✓ MGSM                :  0.00
✓ MMLU                :  0.25
Healed:
✓ GPQA                :  0.25
✓ GSM8k               :  0.00
✓ MGSM                :  0.10
✓ MMLU                :  0.27
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
Prompt processing:    242.8 tokens/sec ( 14 tokens in 0.1s)
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
Prompt processing:   1400.7 tokens/sec ( 72 tokens in 0.1s)
Tokens generation:    464.7 tokens/sec (400 tokens in 0.9s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing DoRA training...
epoch=    0 avg_loss=    0.27 elp_train=   11.01
└ test output: ['<svg width="100" height="100" viewBox="-50 -5']
epoch=    1 avg_loss=    0.04 elp_train=   10.91
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
<svg width="100" height="100" viewBox="-50 -50 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="0" cy="0" r="15" fill="#f6f4f5"/></svg><|im_end|>
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    435.7 tokens/sec ( 15 tokens in 0.0s)
Tokens generation:    165.3 tokens/sec (256 tokens in 1.5s)
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
**The Story of Einstein: A Brief version**  
*by: The story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of the story of
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    524.8 tokens/sec ( 14 tokens in 0.0s)
Tokens generation:    242.3 tokens/sec (100 tokens in 0.4s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing healing...
teacher_to_student=[(16, 13)]   
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
〄 Testing lm-eval on original model...
Starting lm-evaluation-harness on: ['mmlu', 'gpqa_main_zeroshot', 'gsm8k', 'mgsm_direct_zh']
Evaluating generation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [01:12<00:00,  1.81s/it]
Evaluating loglikelihood: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4640/4640 [04:04<00:00, 19.01it/s]
|                 Tasks                 |Version|     Filter      |n-shot|  Metric   |   |Value |   |Stderr|
|---------------------------------------|------:|-----------------|-----:|-----------|---|-----:|---|-----:|
|gpqa_main_zeroshot                     |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|                                       |       |none             |     0|acc_norm   |↑  |0.3500|±  |0.1094|
|gsm8k                                  |      3|flexible-extract |     0|exact_match|↑  |0.3500|±  |0.1094|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_zh                         |      3|flexible-extract |     0|exact_match|↑  |0.3500|±  |0.1094|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mmlu                                   |      2|none             |      |acc        |↑  |0.4158|±  |0.0144|
| - humanities                          |      2|none             |      |acc        |↑  |0.4538|±  |0.0310|
|  - formal_logic                       |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - high_school_european_history       |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - high_school_us_history             |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - high_school_world_history          |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - international_law                  |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - jurisprudence                      |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - logical_fallacies                  |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - moral_disputes                     |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - moral_scenarios                    |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - philosophy                         |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - prehistory                         |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - professional_law                   |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - world_religions                    |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
| - other                               |      2|none             |      |acc        |↑  |0.4192|±  |0.0303|
|  - business_ethics                    |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - clinical_knowledge                 |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - college_medicine                   |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - global_facts                       |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - human_aging                        |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - management                         |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - marketing                          |      1|none             |     0|acc        |↑  |0.6000|±  |0.1124|
|  - medical_genetics                   |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - miscellaneous                      |      1|none             |     0|acc        |↑  |0.6000|±  |0.1124|
|  - nutrition                          |      1|none             |     0|acc        |↑  |0.6000|±  |0.1124|
|  - professional_accounting            |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - professional_medicine              |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - virology                           |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
| - social sciences                     |      2|none             |      |acc        |↑  |0.4250|±  |0.0307|
|  - econometrics                       |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - high_school_geography              |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - high_school_government_and_politics|      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - high_school_macroeconomics         |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - high_school_microeconomics         |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - high_school_psychology             |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - human_sexuality                    |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - professional_psychology            |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - public_relations                   |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - security_studies                   |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - sociology                          |      1|none             |     0|acc        |↑  |0.7500|±  |0.0993|
|  - us_foreign_policy                  |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
| - stem                                |      2|none             |      |acc        |↑  |0.3816|±  |0.0248|
|  - abstract_algebra                   |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - anatomy                            |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - astronomy                          |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - college_biology                    |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - college_chemistry                  |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - college_computer_science           |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - college_mathematics                |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - college_physics                    |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - computer_security                  |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - conceptual_physics                 |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - electrical_engineering             |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - elementary_mathematics             |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - high_school_biology                |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - high_school_chemistry              |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - high_school_computer_science       |      1|none             |     0|acc        |↑  |0.5000|±  |0.1147|
|  - high_school_mathematics            |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - high_school_physics                |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_statistics             |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - machine_learning                   |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|

〄 Testing lm-eval on collapsed model...
Starting lm-evaluation-harness on: ['mmlu', 'gpqa_main_zeroshot', 'gsm8k', 'mgsm_direct_zh']
Evaluating generation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [02:21<00:00,  3.53s/it]
Evaluating loglikelihood: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4640/4640 [04:04<00:00, 19.02it/s]
|                 Tasks                 |Version|     Filter      |n-shot|  Metric   |   |Value |   |Stderr|
|---------------------------------------|------:|-----------------|-----:|-----------|---|-----:|---|-----:|
|gpqa_main_zeroshot                     |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|                                       |       |none             |     0|acc_norm   |↑  |0.2500|±  |0.0993|
|gsm8k                                  |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_zh                         |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mmlu                                   |      2|none             |      |acc        |↑  |0.2465|±  |0.0128|
| - humanities                          |      2|none             |      |acc        |↑  |0.2115|±  |0.0256|
|  - formal_logic                       |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - high_school_european_history       |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - high_school_us_history             |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_world_history          |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - international_law                  |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - jurisprudence                      |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - logical_fallacies                  |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - moral_disputes                     |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - moral_scenarios                    |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - philosophy                         |      1|none             |     0|acc        |↑  |0.1000|±  |0.0688|
|  - prehistory                         |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - professional_law                   |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - world_religions                    |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
| - other                               |      2|none             |      |acc        |↑  |0.2769|±  |0.0278|
|  - business_ethics                    |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - clinical_knowledge                 |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - college_medicine                   |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - global_facts                       |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - human_aging                        |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - management                         |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - marketing                          |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - medical_genetics                   |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - miscellaneous                      |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - nutrition                          |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - professional_accounting            |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - professional_medicine              |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - virology                           |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
| - social sciences                     |      2|none             |      |acc        |↑  |0.2542|±  |0.0278|
|  - econometrics                       |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_geography              |      1|none             |     0|acc        |↑  |0.1000|±  |0.0688|
|  - high_school_government_and_politics|      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - high_school_macroeconomics         |      1|none             |     0|acc        |↑  |0.1000|±  |0.0688|
|  - high_school_microeconomics         |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - high_school_psychology             |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - human_sexuality                    |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - professional_psychology            |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - public_relations                   |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - security_studies                   |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - sociology                          |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - us_foreign_policy                  |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
| - stem                                |      2|none             |      |acc        |↑  |0.2447|±  |0.0223|
|  - abstract_algebra                   |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - anatomy                            |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - astronomy                          |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - college_biology                    |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - college_chemistry                  |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - college_computer_science           |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - college_mathematics                |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - college_physics                    |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - computer_security                  |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - conceptual_physics                 |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - electrical_engineering             |      1|none             |     0|acc        |↑  |0.1000|±  |0.0688|
|  - elementary_mathematics             |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_biology                |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - high_school_chemistry              |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - high_school_computer_science       |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_mathematics            |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_physics                |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - high_school_statistics             |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - machine_learning                   |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|


teacher_to_student=[(16, 13)]   
epoch=    0 avg_loss=    1.75 elp_train=   49.47
└ test output: ["I'm sorry! I can't help with that. It's a medium red circle. Let me"]
〄 Testing lm-eval on healed model...
Starting lm-evaluation-harness on: ['mmlu', 'gpqa_main_zeroshot', 'gsm8k', 'mgsm_direct_zh']
Evaluating generation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [01:05<00:00,  1.64s/it]
Evaluating loglikelihood: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4640/4640 [05:19<00:00, 14.51it/s]
|                 Tasks                 |Version|     Filter      |n-shot|  Metric   |   |Value |   |Stderr|
|---------------------------------------|------:|-----------------|-----:|-----------|---|-----:|---|-----:|
|gpqa_main_zeroshot                     |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|                                       |       |none             |     0|acc_norm   |↑  |0.2500|±  |0.0993|
|gsm8k                                  |      3|flexible-extract |     0|exact_match|↑  |0.0000|±  |0.0000|
|                                       |       |strict-match     |     0|exact_match|↑  |0.0000|±  |0.0000|
|mgsm_direct_zh                         |      3|flexible-extract |     0|exact_match|↑  |0.1000|±  |0.0688|
|                                       |       |remove_whitespace|     0|exact_match|↑  |0.0000|±  |0.0000|
|mmlu                                   |      2|none             |      |acc        |↑  |0.2728|±  |0.0131|
| - humanities                          |      2|none             |      |acc        |↑  |0.2846|±  |0.0279|
|  - formal_logic                       |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - high_school_european_history       |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - high_school_us_history             |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_world_history          |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - international_law                  |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - jurisprudence                      |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - logical_fallacies                  |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - moral_disputes                     |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - moral_scenarios                    |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - philosophy                         |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - prehistory                         |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - professional_law                   |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - world_religions                    |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
| - other                               |      2|none             |      |acc        |↑  |0.3115|±  |0.0287|
|  - business_ethics                    |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - clinical_knowledge                 |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - college_medicine                   |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - global_facts                       |      1|none             |     0|acc        |↑  |0.5500|±  |0.1141|
|  - human_aging                        |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - management                         |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - marketing                          |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - medical_genetics                   |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - miscellaneous                      |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - nutrition                          |      1|none             |     0|acc        |↑  |0.4500|±  |0.1141|
|  - professional_accounting            |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - professional_medicine              |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - virology                           |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
| - social sciences                     |      2|none             |      |acc        |↑  |0.2125|±  |0.0262|
|  - econometrics                       |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - high_school_geography              |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - high_school_government_and_politics|      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_macroeconomics         |      1|none             |     0|acc        |↑  |0.0500|±  |0.0500|
|  - high_school_microeconomics         |      1|none             |     0|acc        |↑  |0.0500|±  |0.0500|
|  - high_school_psychology             |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - human_sexuality                    |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - professional_psychology            |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - public_relations                   |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - security_studies                   |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - sociology                          |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - us_foreign_policy                  |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
| - stem                                |      2|none             |      |acc        |↑  |0.2763|±  |0.0226|
|  - abstract_algebra                   |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - anatomy                            |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - astronomy                          |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - college_biology                    |      1|none             |     0|acc        |↑  |0.6000|±  |0.1124|
|  - college_chemistry                  |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - college_computer_science           |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - college_mathematics                |      1|none             |     0|acc        |↑  |0.1500|±  |0.0819|
|  - college_physics                    |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - computer_security                  |      1|none             |     0|acc        |↑  |0.3000|±  |0.1051|
|  - conceptual_physics                 |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - electrical_engineering             |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - elementary_mathematics             |      1|none             |     0|acc        |↑  |0.2000|±  |0.0918|
|  - high_school_biology                |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_chemistry              |      1|none             |     0|acc        |↑  |0.4000|±  |0.1124|
|  - high_school_computer_science       |      1|none             |     0|acc        |↑  |0.3500|±  |0.1094|
|  - high_school_mathematics            |      1|none             |     0|acc        |↑  |0.1000|±  |0.0688|
|  - high_school_physics                |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|
|  - high_school_statistics             |      1|none             |     0|acc        |↑  |0.0500|±  |0.0500|
|  - machine_learning                   |      1|none             |     0|acc        |↑  |0.2500|±  |0.0993|


Original:
✓ GPQA                :  0.35
✓ GSM8k               :  0.35
✓ MGSM                :  0.35
✓ MMLU                :  0.42
Collapsed:
✓ GPQA                :  0.25
✓ GSM8k               :  0.00
✓ MGSM                :  0.00
✓ MMLU                :  0.25
Healed:
✓ GPQA                :  0.25
✓ GSM8k               :  0.00
✓ MGSM                :  0.10
✓ MMLU                :  0.27
```
</pre></details><br>
