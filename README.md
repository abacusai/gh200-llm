# GH200s

The new NVIDIA GH200 chips hold a lot of promise because of their high CPU->GPU memory bandwidth (7x the H100 DGX setup). Node to node communication along is supported by InfiniBand HDR 200Gbps non-blocking and ConnectX-7 NIC ratio for GPUDirect RDMA. NVIDIA has published some [impressive raw numbers](https://docs.nvidia.com/gh200-benchmarking-guide.pdf). 

We ran some initial real world tests comparing a new batch of **GH200 96GB GPU memory, 480GB node memory** nodes that we got from Lambda Labs. We did the initial tests on CUDA 12.2 and are waiting to repeat this with CUDA 12.3 which has some additional optimizations for CPU->GPU memory paging.

To run tests on the GH200, we had to compile various libraries like Flash Attention, xformers, VLLM on CUDA 12.2 and ARM64 (the Grace Hopper CPU has ARM silicon). You can find the Docker image published at: `ghcr.io/abacusai/gh200-llm/llm-train-serve:latest`

The main use cases that we were interested in testing were inference on large context requests on large models and full finetuning of smaller 13B and 7B models.  

# Large Context 70B Inference Performance

We compared **4 H100 (on a 8xH100 DGX)** and **4 GH200 MGX nodes** serving [Abacus Giraffe](https://huggingface.co/abacusai/Giraffe-v2-70b-32k) (a finetuned Llama2 70B 32k context LLM. We used Flash Attention 2 and VLLM for serving these models. 

|      | Input Tokens | Output Tokens | Concurrency | 4xH100 Time (secs)  | 4xGH200 Time (secs)  | Improvement |
|------|--------------|---------------|-------------|---------------------|----------------------|-------------|
| 1    | 31,242       | 100           | 1           | 18.229              | 9.699                | 87.94%      |
| 2    | 3,349        | 100           | 8           | 13.836              | 8.271                | 67.28%      |

Numbers are promising, we get almost a 2x improvement in latency at larger contexts. 


# Small Model Finetuning Performance

We compared full finetuning on a 13B base model (not LoRA) between 2 H100s and 1 GH200 with Deepspeed gradient offloads to CPU.

| Hardware | Micro Batch Size | Batch Size | Secs/iteration|
|----------|------------------|------------|---------------|
| 2xH100   | 64               | 128        | 14            |
| 1xGH200  | 128              | 128        | 10            |

A 56% improved with 1/2 the number of nodes makes this fairly appealing especially given that keeping the training to a single node reduces a lot of operational complexity.
