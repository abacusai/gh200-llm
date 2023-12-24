# Inference Performance

Comparing inference performance between 4 H100 (on a 8xH100 DGX) and 4 GH200 MGX nodes using Infiniband interconnect.


We tested inference performance for Abacus Giraffe (a finetuned Llama2 70B 32k context LLM) using Flash Attention 2 and VLLM for serving.

Test 1:

Input Tokens: 31242
Output Tokens: 100
Concurrency: 1

H100 - 18.229 seconds
GH200 - 9.699 seconds


Test 2:
Input Tokens: 3349
Output Tokens: 100
Concurrency: 8

H100 - 13.836
GH200: 8.271

For inference, GH200s are not such a good value for serving large models with large context just yet (our current software stack is based on CuDA 12.2, and we hope to see improvements using the just released CuDA 12.3), given that they are likely to be around 2x the price of H100s currently.

# Training Performance

We compared full finetuning on a 13B base model (not LoRA) between 2 H100s and 1 GH200:

1x GH200 micro-batch = batch = 128 w/ grad offload = 10s / iteration
2x H100 micro-batch=64, batch = 128 w/ grad offload = 14s / iteration

This tests the increased CPU offload performance that comes with the GH200 because of 7X memory bandwidth between CPU and GPU.
