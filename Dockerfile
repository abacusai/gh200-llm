FROM nvcr.io/nvidia/pytorch:24.12-py3

RUN chown root:root /usr/lib
RUN apt update -y && apt install -y build-essential curl kmod openssh-server openssh-client pdsh tmux

RUN pip install --upgrade pip wheel

RUN pip install --root-user-action=ignore \
        accelerate \
        deepspeed \
        openai \
        mistral_common \
        msgspec \
        peft \
        pyarrow \
        sentencepiece \
        tiktoken \
        torchao \
        transformers \
        trl

RUN pip install stanford-stk torchao --no-deps --root-user-action=ignore

RUN pip install --root-user-action=ignore \
        aioprometheus \
        blake3 \
        fastapi \
        fschat[model_worker,webui] \
        gguf \
        lm-format-enforcer \
        llmcompressor \
        outlines==0.0.44 \
        partial_json_parser \
        prometheus-fastapi-instrumentator \
        ray \
        setproctitle \
        setuptools_scm \
        typer \
        uvicorn[standard]

RUN pip uninstall --root-user-action=ignore -y pynvml

RUN pip install --root-user-action=ignore nvidia-ml-py

RUN mkdir /packages/

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/flash_attn-2.7.2.post1-cp312-cp312-linux_aarch64.whl /packages/flash_attn-2.7.2.post1-cp312-cp312-linux_aarch64.whl
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/flash_attn-2.7.2.post1-cp312-cp312-linux_x86_64.whl /packages/flash_attn-2.7.2.post1-cp312-cp312-linux_x86_64.whl
RUN pip install --root-user-action=ignore --no-deps --no-index --upgrade --find-links /packages flash-attn

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/xformers-0.0.30%2B46a02df6.d20250103-cp312-cp312-linux_aarch64.whl /packages/xformers-0.0.30+46a02df6.d20250103-cp312-cp312-linux_aarch64.whl
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/xformers-0.0.30%2B46a02df6.d20250103-cp312-cp312-linux_x86_64.whl /packages/xformers-0.0.30+46a02df6.d20250103-cp312-cp312-linux_x86_64.whl
RUN pip install --root-user-action=ignore --no-deps --no-index --find-links /packages xformers

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/megablocks-0.7.0-cp312-cp312-linux_aarch64.whl /packages/megablocks-0.7.0-cp312-cp312-linux_aarch64.whl
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/megablocks-0.7.0-cp312-cp312-linux_x86_64.whl /packages/megablocks-0.7.0-cp312-cp312-linux_x86_64.whl
RUN pip install --root-user-action=ignore --no-deps --no-index --find-links /packages megablocks

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/bitsandbytes-0.45.1.dev0-cp312-cp312-linux_aarch64.whl /packages/bitsandbytes-0.45.1.dev0-cp312-cp312-linux_aarch64.whl
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/bitsandbytes-0.45.1.dev0-cp312-cp312-linux_x86_64.whl /packages/bitsandbytes-0.45.1.dev0-cp312-cp312-linux_x86_64.whl
RUN pip install --root-user-action=ignore --no-deps --no-index --find-links /packages bitsandbytes

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/vllm-0.7.2%2Bcu126-cp312-cp312-linux_aarch64.whl /packages/vllm-0.7.2+cu126-cp312-cp312-linux_aarch64.whl
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/vllm-0.7.2%2Bcu126-cp312-cp312-linux_x86_64.whl /packages/vllm-0.7.2+cu126-cp312-cp312-linux_x86_64.whl
RUN pip install --root-user-action=ignore --no-deps --no-index --find-links /packages vllm

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/sglang-0.4.3-py3-none-any.whl /packages/sglang-0.4.3-py3-none-any.whl
RUN pip install --root-user-action=ignore --no-deps --no-index --find-links /packages sglang

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/sglang_router-0.1.4-cp312-cp312-linux_aarch64.whl /packages/sglang_router-0.1.4-cp312-cp312-linux_aarch64.whl
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/sglang_router-0.1.4-cp312-cp312-linux_x86_64.whl /packages/sglang_router-0.1.4-cp312-cp312-linux_x86_64.whl
RUN pip install --root-user-action=ignore --no-deps --no-index --find-links /packages sglang-router

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/sgl_kernel-0.0.3.post6-cp39-abi3-manylinux2014_aarch64.whl /packages/sgl_kernel-0.0.3.post6-cp39-abi3-manylinux2014_aarch64.whl
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/sgl_kernel-0.0.3.post6-cp39-abi3-manylinux2014_x86_64.whl /packages/sgl_kernel-0.0.3.post6-cp39-abi3-manylinux2014_x86_64.whl
RUN pip install --root-user-action=ignore --no-deps --no-index --find-links /packages sgl-kernel

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/flashinfer_python-0.2.0.post2-cp312-cp312-linux_aarch64.whl /packages/flashinfer_python-0.2.0.post2-cp312-cp312-linux_aarch64.whl
ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/flashinfer_python-0.2.0.post2-cp312-cp312-linux_x86_64.whl /packages/flashinfer_python-0.2.0.post2-cp312-cp312-linux_x86_64.whl
RUN pip install --root-user-action=ignore --no-deps --no-index --find-links /packages flashinfer-python

ADD https://static.abacus.ai/pypi/abacusai/gh200-llm/pytorch-2412-cuda126/decord-0.6.0-patched-cp312-cp312-linux_aarch64.whl /packages/decord-0.6.0-cp312-cp312-linux_aarch64.whl
ADD https://files.pythonhosted.org/packages/11/79/936af42edf90a7bd4e41a6cac89c913d4b47fa48a26b042d5129a9242ee3/decord-0.6.0-py3-none-manylinux2010_x86_64.whl /packages/decord-0.6.0-py3-none-manylinux2010_x86_64.whl
RUN pip install --root-user-action=ignore --no-deps --no-index --find-links /packages decord

RUN rm -r /packages
