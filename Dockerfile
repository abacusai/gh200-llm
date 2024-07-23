FROM nvcr.io/nvidia/pytorch:24.06-py3

RUN chown root:root /usr/lib
RUN apt update -y && apt install -y build-essential curl openssh-server openssh-client pdsh

RUN pip install --upgrade pip wheel

RUN pip install \
        accelerate \
        deepspeed \
        openai \
        peft \
        pyarrow==14.0.2 \
        sentencepiece \
        tiktoken \
        transformers \
        trl

RUN pip install stanford-stk --no-deps

RUN pip install \
        aioprometheus \
        fastapi \
        fschat[model_worker,webui] \
        lm-format-enforcer==0.10.3 \
        outlines \
        prometheus-fastapi-instrumentator \
        protobuf==3.20.3 \
        ray==2.9.2 \
        typer==0.9.4 \
        uvicorn

RUN mkdir /packages/

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.5/flash_attn-2.6.1-cp310-cp310-linux_aarch64.whl /packages/flash_attn-2.6.1-cp310-cp310-linux_aarch64.whl
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.5/flash_attn-2.6.1-cp310-cp310-linux_x86_64.whl /packages/flash_attn-2.6.1-cp310-cp310-linux_x86_64.whl
RUN pip install --no-deps --find-links /packages flash-attn==2.6.1

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.5/vllm_flash_attn-2.5.9.post1-cp310-cp310-linux_aarch64.whl /packages/vllm_flash_attn-2.5.9.post1-cp310-cp310-linux_aarch64.whl
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.5/vllm_flash_attn-2.5.9.post1-cp310-cp310-linux_x86_64.whl /packages/vllm_flash_attn-2.5.9.post1-cp310-cp310-linux_x86_64.whl
RUN pip install --no-index --no-deps --find-links /packages vllm-flash-attn==2.5.9.post1

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.5/xformers-0.0.27%2B184b280.d20240717-cp310-cp310-linux_aarch64.whl /packages/xformers-0.0.27+184b280.d20240717-cp310-cp310-linux_aarch64.whl
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.5/xformers-0.0.27%2B184b280.d20240718-cp310-cp310-linux_x86_64.whl /packages/xformers-0.0.27+184b280.d20240718-cp310-cp310-linux_x86_64.whl
RUN pip install --no-deps --find-links /packages xformers==0.0.27

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.5/megablocks-0.5.1-cp310-cp310-linux_aarch64.whl /packages/megablocks-0.5.1-cp310-cp310-linux_aarch64.whl
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.5/megablocks-0.5.1-cp310-cp310-linux_x86_64.whl /packages/megablocks-0.5.1-cp310-cp310-linux_x86_64.whl
RUN pip install --no-deps --find-links /packages megablocks==0.5.1

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.5/bitsandbytes-0.43.2.dev0-cp310-cp310-linux_aarch64.whl /packages/bitsandbytes-0.43.2.dev0-cp310-cp310-linux_aarch64.whl
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.5/bitsandbytes-0.43.2.dev0-cp310-cp310-linux_x86_64.whl /packages/bitsandbytes-0.43.2.dev0-cp310-cp310-linux_x86_64.whl
RUN pip install --no-index --no-deps --find-links /packages bitsandbytes==0.43.2.dev0

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.5/vllm-0.5.3.post1%2Bcu125-cp310-cp310-linux_aarch64.whl /packages/vllm-0.5.3.post1+cu125-cp310-cp310-linux_aarch64.whl
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/cuda12.5/vllm-0.5.3.post1%2Bcu125-cp310-cp310-linux_x86_64.whl /packages/vllm-0.5.3.post1+cu125-cp310-cp310-linux_x86_64.whl
RUN pip install --no-deps --find-links /packages vllm==0.5.3.post1

RUN rm -r /packages
