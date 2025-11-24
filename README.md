[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XYyO04by5UCaMoH0C2MUmsLPOwddN5Gp?usp=sharing)

# Accelerated LLM inference

`rcr-lm` generates tokens at 200+ tokens/sec, significantly outperforming `mlx-lm`'s 175 token/sec:

## rcrlm

```python
from rcrlm import load, generate
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

## mlx-lm

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

<details><summary>WIP (adapted codes from [nnx-lm](https://pypi.org/project/nnx-lm/) to try some stuff)</summary><pre>
# Etc

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
Prompt processing:    247.8 tokens/sec ( 14 tokens in 0.1s)
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
<think>                                                                                                                                                                                                             Okay, the user wants a short introduction to a large language model. Let me start by recalling what I know about LLMs. They're big language models, right? So I should mention their ability to understand and generate text. Maybe start with the basics: they can process and generate text, not just a few words. Then explain their training data, like the amount of text they're trained on. Also, their capabilities: understanding and generating text, answering questions, etc. Need
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
Comparison of Sortino Ratio for Bitcoin and Ethereum.<|im_end|>                                                                                                                                                     <|im_start|>assistant                                                                                                                                                                                               └───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00003 ──────────────────────────────┐
<think>
Okay, the user is asking for a comparison between the Sortino Ratio for Bitcoin and Ethereum. Let me start by recalling what the Sortino Ratio is. It's a measure of risk-adjusted return for a portfolio, right? It's calculated as (Return - Risk-Free Rate) divided by the Risk (Standard Deviation). The Sortino Ratio is usually used for a single asset, so comparing it between Bitcoin and Ethereum would be useful.

First, I need to check the historical data
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐                                                                                                                                           Prompt processing:   1407.7 tokens/sec ( 72 tokens in 0.1s)
Tokens generation:    475.1 tokens/sec (400 tokens in 0.8s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing DoRA training...
epoch=    0 avg_loss=    0.34 elp_train=    9.82
└ test output: ['<svg width="100" height="100" viewBox="-50 -5']
epoch=    1 avg_loss=    0.05 elp_train=    9.79
└ test output: ['<svg width="100" height="100" viewBox="-50 -5']
〄 Testing DoRA decoding...
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
medium red circle
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐
<svg width="100" height="100" viewBox="-50 -50 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="0" cy="0" r="38" fill="#f6f4f5"/></svg>                                                                      └───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    324.2 tokens/sec ( 12 tokens in 0.0s)                                                                                                                                                         Tokens generation:    164.4 tokens/sec (256 tokens in 1.6s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing collapse...
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
Write a story about Einstein
<|im_end|>                                                                                                                                                                                                          <|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐
The story of a person who is not only a scientist, but also a part of the society that is more than a mere physical space. The story of a person who is not only in the same as the same place, as a whole, and thus the same as the same place. The story of the person who is not only in the same as the time and space, as a whole, as a whole, and thus the same as the whole. The story of the person who is not
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    568.8 tokens/sec ( 14 tokens in 0.0s)
Tokens generation:    253.6 tokens/sec (100 tokens in 0.4s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing healing...                                                                                                                                                                                               epoch=    0 avg_loss=    2.07 elp_train=   11.81
└ test output: ['the medium red color is a symbol of the first or the second level of the design of the design']
epoch=    1 avg_loss=    2.08 elp_train=   12.21                                                                                                                                                                    └ test output: ['the medium red color is a symbol of the first or the second level of the design of the design']
┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
Write a story about Einstein
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘                                                                                                                                           ┌────────────────────────────── Out 00000 ──────────────────────────────┐
Assistant's response is a response to the first question of the first part of the first part of the first part of the story about the first part of the first part of the answer of the question. the answer is a response to the question about the first part of the answer of the question, which is the first part of the answer of the question, and the answer is a part of the answer of the question, and the answer is a part of the answer of the answer of the question,
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    442.1 tokens/sec ( 14 tokens in 0.0s)
Tokens generation:    198.0 tokens/sec (100 tokens in 0.5s)
└───────────────────────────────────────────────────────────────────────┘
```
</pre></details><br>
