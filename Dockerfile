FROM nvcr.io/nvidia/pytorch:23.12-py3

RUN pip install --upgrade pip wheel

RUN pip install \
        accelerate \
        aioprometheus \
        deepspeed \
        fastapi \
        peft \
        ray \
        sentencepiece \
        transformers \
        trl \
        uvicorn

RUN pip install --no-deps stanford-stk

RUN mkdir /packages/

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/flash_attn-2.4.1-cp310-cp310-linux_aarch64.whl /packages

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/vllm-0.2.6%2Bcu123-cp310-cp310-linux_aarch64.whl /packages

ADD https://github.com/acollins3/triton/releases/download/triton-2.1.0-arm64/triton-2.1.0-cp310-cp310-linux_aarch64.whl /packages

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/xformers-0.0.24%2B042abc8.d20231225-cp310-cp310-linux_aarch64.whl /packages

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.3/megablocks-0.5.0-cp310-cp310-linux_aarch64.whl /packages

RUN pip install --no-deps --find-links /packages flash-attn==2.4.1

RUN pip install --no-deps --find-links /packages vllm==0.2.6

RUN pip install --no-deps --find-links /packages triton==2.1.0

RUN pip install --no-deps --find-links /packages xformers==0.0.24

RUN pip install --no-deps --find-links /packages megablocks==0.5.0

RUN rm -r /packages
