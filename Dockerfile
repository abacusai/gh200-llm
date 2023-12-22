FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN pip install -U pip wheel

RUN pip install \
        accelerate \
        aioprometheus \
        deepspeed \
        peft \
        ray \
        transformers \
        trl

RUN pip install --no-deps stanford-stk

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/vllm-0.2.5+cu122-cp310-cp310-linux_aarch64.whl /packages

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/flash_attn-2.3.6-cp310-cp310-linux_aarch64.whl /packages

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/xformers-0.0.24+40d3967.d20231210-cp310-cp310-linux_aarch64.whl /packages

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/megablocks-0.5.0-cp310-cp310-linux_aarch64.whl /packages

ADD https://github.com/acollins3/triton/releases/download/triton-2.1.0-arm64/triton-2.1.0-cp310-cp310-linux_aarch64.whl /packages

RUN pip install --no-deps --find-links . megablocks

RUN pip uninstall -y flash-attn && \
    pip install --no-deps --find-links . flash-attn

RUN pip install --no-deps --find-links . xformers

RUN pip install --no-deps --find-links . vllm

RUN pip install --no-deps --find-links . triton

RUN rm -r /packages
