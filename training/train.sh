#!/bin/bash
set -ex

TRAINING_ROOT=/shared/gh200-llm/training
DATASET_PATH=${TRAINING_ROOT}/sample_train.jsonl
MODEL_ROOT=/shared/models


NCCL_NVLS_ENABLE=0 \
WANDB_DISABLED=true \
TRANSFORMERS_VERBOSITY=warning \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 \
deepspeed --hostfile hosts.txt $TRAINING_ROOT/finetune_w8a16.py \
    --deepspeed=$TRAINING_ROOT/deepspeed_w8a16.json \
    --context_length=2048 \
    --per_device_train_batch_size=2 --gradient_accumulation_steps=2 \
    --per_device_eval_batch_size=4 \
    --output_dir=output \
    --dataset_path=$DATASET_PATH \
    --model_name_or_path $MODEL_ROOT/Meta-Llama-3.1-405B-Instruct \
    --conv_config=$TRAINING_ROOT/conversation.json \
    --do_eval --save_steps 20 --max_steps 60 \
    --learning_rate 2e-5 --lr_scheduler_type linear --warmup_ratio 0.02
