# GH200s

The new NVIDIA GH200 chips hold a lot of promise because of their high CPU->GPU memory bandwidth (7x the H100 DGX setup). Node to node communication along is supported by InfiniBand HDR 200Gbps non-blocking and ConnectX-7 NIC ratio for GPUDirect RDMA. NVIDIA has published some [impressive raw numbers](https://docs.nvidia.com/gh200-benchmarking-guide.pdf).

This repo maintains a ready to use docker image which works on H100s and GH200s (multi architecture) with the latest versions of VLLM, XFormers and Flash Attention to easily be able to serve large models and finetune 8B ones.

These days, you can also quantize a 70B or a 72B model (both excellent candidates for finetuning and distillation from reasoner models), by running the `training/quantize_fp8.py` on a on a 2xH100 node, and serve on a single GH200 node. The 92GB GPU allows you to host it at full context length in fp8 mode. [Lambda Labs](https://lambdalabs.com/) has some great rates for single node GH200 on-demand rentals.

We can also serve DeepSeek-R1 on a 8xGH200 node (connected using Infiniband) using this docker along with SGLang or vLLM and Ray

Docker image is published at: `ghcr.io/abacusai/gh200-llm/llm-train-serve:latest`
