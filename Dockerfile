FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN pip install --upgrade pip wheel

RUN pip install \
        accelerate \
        aioprometheus \
        deepspeed \
        fastapi \
        outlines \
        peft \
        ray==2.9.2 \
        sentencepiece \
        transformers \
        trl \
        uvicorn

RUN pip install --no-deps stanford-stk

RUN mkdir /packages/

ADD https://github.com/acollins3/triton/releases/download/triton-2.1.0-arm64/triton-2.1.0-cp310-cp310-linux_aarch64.whl /packages
RUN pip install --no-deps --find-links /packages triton>=2.1.0

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/flash_attn-2.5.6-cp310-cp310-linux_aarch64.whl /packages
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/flash_attn-2.5.6-cp310-cp310-linux_x86_64.whl /packages
RUN pip install --no-deps --find-links /packages flash-attn==2.5.6

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/xformers-0.0.26%2B132da6a.d20240324-cp310-cp310-linux_aarch64.whl /packages/xformers-0.0.26+132da6a.d20240324-cp310-cp310-linux_aarch64.whl
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/xformers-0.0.26%2B132da6a.d20240324-cp310-cp310-linux_x86_64.whl /packages/xformers-0.0.26+132da6a.d20240324-cp310-cp310-linux_x86_64.whl
RUN pip install --no-deps --find-links /packages xformers==0.0.26

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/megablocks-0.5.1-cp310-cp310-linux_aarch64.whl /packages
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/megablocks-0.5.1-cp310-cp310-linux_x86_64.whl /packages
RUN pip install --no-deps --find-links /packages megablocks==0.5.1

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/bitsandbytes-0.44.0.dev0-cp310-cp310-linux_aarch64.whl /packages/
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/bitsandbytes-0.44.0.dev0-cp310-cp310-linux_x86_64.whl /packages/
RUN pip install --no-deps --find-links /packages bitsandbytes==0.44.0.dev0

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/vllm-0.3.0.forked%2Bcu123-cp310-cp310-linux_aarch64.whl /packages/vllm-0.3.0+cu123-cp310-cp310-linux_aarch64.whl
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/vllm-0.3.0.forked%2Bcu123-cp310-cp310-linux_x86_64.whl /packages/vllm-0.3.0+cu123-cp310-cp310-linux_x86_64.whl
RUN pip install --no-deps --find-links /packages vllm==0.3.0

RUN rm -r /packages
