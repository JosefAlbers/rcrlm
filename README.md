Adapted codes from [nnx-lm](https://pypi.org/project/nnx-lm/) to try some stuff (WIP)

## Vanilla decoding

```
┌────────────────────────────── Inp 000 ──────────────────────────────┐
<|im_start|>user
Write a story about Einstein
<|im_end|>
<|im_start|>assistant
└─────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 000 ──────────────────────────────┐
<think>
Okay, the user wants a story about Einstein. Let me start by recalling Einstein's life. He was a genius, right? He lived in Germany, worked on relativity, and had a lot of contributions. The story needs to highlight his scientific achievements and personality.

First, I should set the scene. Maybe start with his early life in Germany. His parents were scientists, so that's a good point. Then introduce his work on relativity. I need to mention the E=mcÂ² and the time dilation. Also, his personal life, like his family and his struggles with the scientific community.

I should include some key moments: his time at the University of Zurich, the E=mcÂ² experiment, and his later years. Maybe add some quotes from his work to make it more engaging. Also, think about the themes of genius, perseverance, and the impact of science on society.

Wait, the user might want the story to be engaging and highlight Einstein's legacy. Need to make sure the story flows well, with a good narrative structure. Avoid any inaccuracies, like mentioning specific events or dates. Keep it concise but informative. Let me check if I have all the key points covered: early life, scientific contributions, personal life, and legacy
└─────────────────────────────────────────────────────────────────────┘
=== Benchmarks ===
Prompt processing: 252.3 tokens/sec (14 tokens in 0.1s)
Token generation: 136.9 tokens/sec (256 tokens in 1.9s)
```

## Batch decoding

```
┌────────────────────────────── Inp 000 ──────────────────────────────┐
<|im_start|>user
#write a quick sort algorithm
<|im_end|>
<|im_start|>assistant
└─────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 000 ──────────────────────────────┐
<think>
Okay, I need to write a quick sort algorithm. Let me think about how to approach this. Quick sort is a divide-and-conquer algorithm, right? The basic idea is to select a pivot element, partition the array into elements less than or equal to the pivot and greater than or equal to it, and then recursively sort the subarrays.

First, I should outline the steps. The algorithm should have a function that takes an array and a pivot index. Wait, but how do
└─────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Inp 001 ──────────────────────────────┐
<|im_start|>user
Give me a short introduction to large language model.
<|im_end|>
<|im_start|>assistant
└─────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 001 ──────────────────────────────┐
<think>
Okay, the user wants a short introduction to a large language model. Let me start by recalling what I know about LLMs. They're big language models, right? So I should mention their ability to understand and generate text. Maybe start with the basics: they can understand and generate text, which is a key feature. Then, talk about their applications, like writing, answering questions, etc. Also, highlight their capabilities beyond just text, like understanding context and generating creative content.
└─────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Inp 002 ──────────────────────────────┐
<|im_start|>user
Write a neurology ICU admission note.
<|im_end|>
<|im_start|>assistant
└─────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 002 ──────────────────────────────┐
<think>
Okay, I need to write a neurology ICU admission note. Let me start by recalling what an ICU admission note typically includes. It's a medical record that outlines the patient's condition, initial assessment, interventions, and any ongoing care.

First, the patient's name and date of admission. I should make sure to include that. Then, the patient's name, age, gender, and primary diagnosis. Since it's a neurology ICU, the main issue is likely a neurological
└─────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Inp 003 ──────────────────────────────┐
<|im_start|>user
Comparison of Sortino Ratio for Bitcoin and Ethereum.<|im_end|>
<|im_start|>assistant
└─────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 003 ──────────────────────────────┐
<think>
Okay, the user is asking for a comparison between the Sortino Ratio for Bitcoin and Ethereum. Let me start by recalling what the Sortino Ratio is. It's a measure of risk-adjusted return for a portfolio, right? It's calculated as (Return - Risk-Free Rate) divided by the Risk (Standard Deviation). The Sortino Ratio is usually used for a single asset, so comparing it between Bitcoin and Ethereum would be useful.

First, I need to check the historical performance
└─────────────────────────────────────────────────────────────────────┘
=== Benchmarks ===
Prompt processing: 926.6 tokens/sec (72 tokens in 0.1s)
Token generation: 294.2 tokens/sec (400 tokens in 1.4s)
```

## Vanilla DoRA training (and decoding with the trained adapter)

```
epoch=0 avg_loss=2.651796875 elp_train=14.37

┌────────────────────────────── Inp 000 ──────────────────────────────┐
<|im_start|>user
medium red circle
<|im_end|>
<|im_start|>assistant
└─────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 000 ──────────────────────────────┐
<|assistant>
ilingual

<Rigidbody]
[bokeh]
[bokeh: 100
└─────────────────────────────────────────────────────────────────────┘
=== Benchmarks ===
Prompt processing: 417.7 tokens/sec (12 tokens in 0.0s)
Token generation: 79.6 tokens/sec (20 tokens in 0.3s)

epoch=1 avg_loss=1.158203125 elp_train=14.42

┌────────────────────────────── Inp 000 ──────────────────────────────┐
<|im_start|>user
medium red circle
<|im_end|>
<|im_start|>assistant
└─────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 000 ──────────────────────────────┐
<svg width="100" height="100" viewBox="-50 -5
└─────────────────────────────────────────────────────────────────────┘
=== Benchmarks ===
Prompt processing: 531.2 tokens/sec (12 tokens in 0.0s)
Token generation: 80.0 tokens/sec (20 tokens in 0.2s)
metadata={'layers': 'all', 'targets': '["self_attn.o_proj"]', 'rank': '32', 'scale': '0.1', 'dropout': '0.0', 'thaws': '["norm"]', 'wt_from': 'test_lora.safetensors'}
File 'models/Qwen3-0.6B/config.json' already exists. Skipping.
File 'models/Qwen3-0.6B/tokenizer.json' already exists. Skipping.
File 'models/Qwen3-0.6B/tokenizer_config.json' already exists. Skipping.
File 'models/Qwen3-0.6B/model.safetensors' already exists. Skipping.
lora_cfg={'wt_from': 'test_lora.safetensors', 'targets': ['self_attn.o_proj'], 'scale': 0.1, 'rank': 32, 'layers': 'all', 'thaws': ['norm'], 'dropout': 0.0}
<svg width="100" height="100" viewBox="-50 -50 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="0" cy="0" r="10" fill="#f0f0f0f0f0f0f0f0f0f0" width="100" height="

┌────────────────────────────── Inp 000 ──────────────────────────────┐
<|im_start|>user
medium red circle
<|im_end|>
<|im_start|>assistant
└─────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 000 ──────────────────────────────┐
<svg width="100" height="100" viewBox="-50 -50 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="0" cy="0" r="10" fill="#f0f0f0f0f0f0f0f0f0f0" width="100" height="100" viewBox="-5 50
└─────────────────────────────────────────────────────────────────────┘
=== Benchmarks ===
Prompt processing: 351.1 tokens/sec (12 tokens in 0.0s)
Token generation: 78.2 tokens/sec (100 tokens in 1.3s)
```


