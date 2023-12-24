FROM nvcr.io/nvidia/pytorch:23.10-py3

LABEL org.opencontainers.image.source="https://github.com/abacusai/gh200-llm"

RUN pip install -U pip wheel

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

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/flash_attn-2.3.6-cp310-cp310-linux_aarch64.whl /packages

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/vllm-0.2.5%2Bcu122-cp310-cp310-linux_aarch64.whl /packages

ADD https://github.com/acollins3/triton/releases/download/triton-2.1.0-arm64/triton-2.1.0-cp310-cp310-linux_aarch64.whl /packages

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/xformers-0.0.24%2B40d3967.d20231210-cp310-cp310-linux_aarch64.whl /packages

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/megablocks-0.5.0-cp310-cp310-linux_aarch64.whl /packages

RUN pip install --no-deps --find-links /packages flash-attn==2.3.6

RUN pip install --no-deps --find-links /packages vllm==0.2.5

RUN pip install --no-deps --find-links /packages triton==2.1.0

RUN pip install --no-deps --find-links /packages xformers==0.0.24

RUN pip install --no-deps --find-links /packages megablocks==0.5.0

RUN rm -r /packages
