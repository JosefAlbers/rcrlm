[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XYyO04by5UCaMoH0C2MUmsLPOwddN5Gp?usp=sharing)

Adapted codes from [nnx-lm](https://pypi.org/project/nnx-lm/) to try some stuff (WIP)

```
〄 Testing vanilla decoding...
┌────────────────────────────── Streaming ──────────────────────────────┐
<think>
Okay, the user wants a story about Einstein. Let me start by recalling Einstein's life. He was a genius, a scientist, and a philosopher. I need to make sure the story includes his contributions to science, maybe his work on relativity, and his personal life.

First, I should set the scene. Maybe start with his early life in Germany, where he was born. Then introduce his family, his parents, maybe his mother's influence. Then his education, the famous lectures, and his breakthroughs.

I need to highlight his scientific achievements, like the theory of relativity. Also, his personal struggles, like the time he spent in the Alps, the Alps being a place of isolation and inspiration.

I should include some quotes or references to his work. Maybe mention his quote about the universe being infinite. Also, his later years and how he passed away.

Wait, the user might want the story to be engaging and highlight his legacy. I need to make sure the story flows well, with a good narrative arc. Avoid clichÃ©s, but still capture his essence. Check for any inaccuracies, like his actual birth date and death year. confirm: Einstein was born on April 14, 18
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
Prompt processing:    234.9 tokens/sec (14 tokens in 0.1s)
Tokens generation:    167.0 tokens/sec (256 tokens in 1.5s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing batch decoding...
┌────────────────────────────── Streaming ──────────────────────────────┐
<think>
Okay, the user is asking for a comparison between the Sortino Ratio for Bitcoin and Ethereum. Let me start by recalling what the Sortino Ratio is. It's a measure of risk-adjusted return for a portfolio, right? It's calculated as (Return - Risk-Free Rate) divided by the Risk (Standard Deviation). The Sortino Ratio is usually used for a single asset, so comparing it between Bitcoin and Ethereum would be useful, I need to check the historical data
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
Okay, the user wants a short introduction to a large language model. Let me start by recalling what I know about LLMs. They're big language models, right? So I should mention their ability to understand and generate text. Maybe start with the basics: they can process and generate text, not just a few words. Then explain their training data, like the amount of text they're trained on. Also, their capabilities: understanding and generating text, answering questions, etc. Need                                                                                                                                                              └───────────────────────────────────────────────────────────────────────┘                                                                                                                                           ┌────────────────────────────── Inp 00002 ──────────────────────────────┐
<|im_start|>user
Write a neurology ICU admission note.
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00002 ──────────────────────────────┐
<think>
Okay, I need to write a neurology ICU admission note. Let me start by recalling what an ICU admission note typically includes. It's a medical record that outlines the patient's condition, initial assessment, interventions, and any ongoing care.
                                                                                                                                                                                                                    First, the patient's name and date of admission. I should make sure to include that. Then, the patient's name, age, gender, and primary diagnosis. Since it's a neurology ICU, the main issue is likely a neurological
└───────────────────────────────────────────────────────────────────────┘                                                                                                                                           ┌────────────────────────────── Inp 00003 ──────────────────────────────┐
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
Prompt processing:   1437.3 tokens/sec (72 tokens in 0.1s)
Tokens generation:    331.1 tokens/sec (400 tokens in 1.2s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing DoRA training...
epoch=    0 avg_loss=    2.66 elp_train=   14.21
└ test output: ['<|assistant>\nresher\n\n</think>\n\nHello! ï¸ıropic\nHello! ï¸ıropic']                                                                                                                             epoch=    1 avg_loss=    1.16 elp_train=   14.16
└ test output: ['<svg width="100" height="100" viewBox="-50 -5']
〄 Testing DoRA decoding...                                                                                                                                                                                         ┌────────────────────────────── Inp 00000 ──────────────────────────────┐
<|im_start|>user
medium red circle
<|im_end|>
<|im_start|>assistant
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Out 00000 ──────────────────────────────┐                                                                                                                                           <svg width="100" height="100" viewBox="-50 -50 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="0" cy="0" r="10" fill="#f0f0f0f0f0f0f0f0f0" width="100" height="100" viewBox="-5 50 1
└───────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── Benchmark ──────────────────────────────┐
Prompt processing:    368.3 tokens/sec (12 tokens in 0.0s)
Tokens generation:     91.1 tokens/sec (100 tokens in 1.1s)
└───────────────────────────────────────────────────────────────────────┘
〄 Testing collapse...
U.shape=(1024, 1024)
S.shape=(1024,)
Vt.shape=(1024, 57344)
U_k=array([[-1.2065296e-02,  4.2261211e-03, -7.1944403e-03, ...,                                                                                                                                                             6.4040911e-03, -6.1178685e-04,  1.0498078e-02],
       [ 3.2725159e-02, -2.6491731e-02,  8.7199891e-03, ...,
        -4.4848169e-03, -2.1432072e-02,  9.7146053e-03],                                                                                                                                                                   [-8.2383193e-02,  6.8220747e-01,  3.2420199e-02, ...,
         6.7272745e-02,  4.8834421e-02, -9.1014996e-02],
       ...,
       [-4.2217653e-03, -1.2476263e-02,  1.3765656e-02, ...,
        -2.3909433e-02, -3.6274742e-02, -3.4406967e-02],
       [-4.7757355e-03, -2.2323474e-03, -8.0787847e-03, ...,
         1.5312760e-02,  1.6842240e-02, -3.2537167e-03],                                                                                                                                                                   [-6.3891648e-03,  7.3768338e-03,  4.3904696e-02, ...,
        -1.7012123e-02, -1.6488006e-02, -6.6838279e-02]],
      shape=(1024, 8), dtype=float32)
S_k=array([14.270441,  9.109717,  8.651474,  8.618107,  8.571209,  8.522299,
        8.439376,  8.402976], dtype=float32)
Vt_k=array([[ 0.0046215 ,  0.00096893,  0.00395941, ...,  0.00063386,
        -0.00073552, -0.00023721],
       [ 0.00524898,  0.00137871,  0.00548535, ...,  0.00061748,
        -0.0083794 ,  0.00150371],
       [-0.00235571, -0.00298685, -0.00143359, ...,  0.00154116,                                                                                                                                                            -0.00174423,  0.0009006 ],                                                                                                                                                                                         ...,
       [ 0.00518476,  0.00237203,  0.00602671, ..., -0.00164276,
         0.0022371 ,  0.00045477],
       [-0.00393195,  0.001701  , -0.00100759, ...,  0.00171548,
        -0.00572792,  0.00245836],
       [ 0.00279137,  0.00140623, -0.00068882, ...,  0.00858562,
         0.00291208,  0.0006799 ]], shape=(8, 57344), dtype=float32)
B=array([[-1.72177091e-01,  3.84987704e-02, -6.22425154e-02, ...,
         5.45775779e-02, -5.16309915e-03,  8.82150978e-02],
       [ 4.67002451e-01, -2.41332188e-01,  7.54407570e-02, ...,
        -3.82209495e-02, -1.80873305e-01,  8.16315934e-02],
       [-1.17564452e+00,  6.21471739e+00,  2.80482501e-01, ...,
         5.73318422e-01,  4.12132025e-01, -7.64796853e-01],
       ...,
       [-6.02464527e-02, -1.13655224e-01,  1.19093217e-01, ...,
        -2.03763336e-01, -3.06136191e-01, -2.89120913e-01],
       [-6.81518540e-02, -2.03360543e-02, -6.98933974e-02, ...,
         1.30499914e-01,  1.42138004e-01, -2.73409039e-02],
       [-9.11761969e-02,  6.72008693e-02,  3.79840344e-01, ...,
        -1.44982398e-01, -1.39148489e-01, -5.61640441e-01]],
      shape=(1024, 8), dtype=float32)
Ws[0]=array([[ 0.01446533,  0.02587891, -0.00445557, ..., -0.00811768,
         0.03637695, -0.00259399],
       [ 0.00952148,  0.00506592,  0.00317383, ..., -0.01806641,
         0.00631714,  0.00982666],
       [ 0.01794434,  0.01397705,  0.04492188, ..., -0.00665283,
        -0.00296021,  0.03540039],
       ...,
       [-0.02600098,  0.00738525, -0.00588989, ..., -0.00970459,
         0.02148438,  0.00147247],
       [-0.01989746, -0.00010061, -0.00848389, ...,  0.06201172,
         0.00738525, -0.0112915 ],
       [-0.00823975,  0.00210571,  0.00087738, ..., -0.02722168,
         0.01501465, -0.02282715]], shape=(1024, 2048), dtype=float32)
w0_recon=array([[ 0.01446532,  0.02587891, -0.00445557, ..., -0.00811768,
         0.03637693, -0.002594  ],
       [ 0.00952149,  0.00506592,  0.00317383, ..., -0.0180664 ,
         0.00631714,  0.00982667],
       [ 0.01794434,  0.01397705,  0.04492187, ..., -0.00665282,
        -0.0029602 ,  0.0354004 ],
       ...,
       [-0.02600099,  0.00738525, -0.00588989, ..., -0.00970459,
         0.02148437,  0.00147246],
       [-0.01989746, -0.00010061, -0.00848387, ...,  0.06201171,
         0.00738526, -0.01129153],
       [-0.00823975,  0.00210571,  0.00087739, ..., -0.0272217 ,
         0.01501463, -0.02282714]], shape=(1024, 2048), dtype=float32)
```
