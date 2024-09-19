#!/usr/bin/python

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Based off the lora script here: https://github.com/artidoro/qlora/blob/main/qlora.py
import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import deepspeed
import deepspeed.comm as dist
import torch
import transformers
from data.finetune_data import filter_long
from data.process_chat_template import process_chat_template
from datasets import Dataset, load_from_disk
from datasets.fingerprint import Hasher
from deepspeed.linear import LoRAConfig, QuantizationConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainer,
                          set_seed)

IGNORE_INDEX = -100
TRAIN_FILE = 'train.jsonl'
EVAL_FILE = 'eval.jsonl'


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={'help': 'Set to model path or HF model repo.'})
    tokenizer_name_or_path: Optional[str] = field(default=None, metadata={'help': 'Set to tokenizer if different from model path.'})


@dataclass
class DataArguments:
    dataset_path: str = field(
        metadata={'help': 'Path to dataset.'}
    )
    conv_config: str = field(metadata={'help': 'Path to json file with conversation format info.'})
    eval_dataset_size: int = field(
        default=1024, metadata={'help': 'Size of validation dataset.'}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            'help': 'For debugging purposes or quicker training, truncate the number of training examples to this '
            'value if set.'
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            'help': 'For debugging purposes or quicker training, truncate the number of evaluation examples to this '
            'value if set.'
        },
    )
    context_length: int = field(
        default=2048,
        metadata={'help': 'Maximum source sequence length. Sequences will be right padded (and possibly truncated).'},
    )


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    lora_r: int = field(
        default=8,
        metadata={'help': 'Lora R dimension.'}
    )
    lora_alpha: float = field(
        default=16,
        metadata={'help': ' Lora alpha.'}
    )
    output_dir: str = field(default='./output', metadata={'help': 'The output dir for logs and checkpoints'})
    per_device_train_batch_size: int = field(default=2, metadata={'help': 'The training batch size per GPU. Increase for better speed.'})
    per_device_eval_batch_size: int = field(default=4, metadata={'help': 'The eval batch size per GPU. Increase for better speed.'})
    max_steps: int = field(default=10000, metadata={'help': 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={'help': 'The L2 weight decay rate of AdamW'})  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={'help': 'The learning rate'})
    max_grad_norm: float = field(default=1.0, metadata={'help': 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={'help': 'Use gradient checkpointing. You want to use this.'})
    activation_checkpointing: bool = field(default=True, metadata={'help': 'Use gradient checkpointing. You want to use this.'})

    lr_scheduler_type: str = field(default='constant', metadata={'help': 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={'help': 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={'help': 'The frequency of update steps after which to log the loss'})
    save_strategy: str = field(default='steps', metadata={'help': 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={'help': 'How often to save a model'})
    save_total_limit: int = field(default=2, metadata={'help': 'How many checkpoints to save before the oldest is overwritten'})


def rank0_log(*args):
    if dist.get_rank() == 0:
        logging.info(*args)


def get_accelerate_model(args: argparse.Namespace):
    lora_config = LoRAConfig(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        base_weight_sharding=dist.get_world_size(),
        offload=False,
        target_mods=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    )

    quantization_config = QuantizationConfig(q_bits=8)

    logging.info(f'Loading base model {args.model_name_or_path}...')
    with deepspeed.linear.Init(lora_config=lora_config, quant_config=quantization_config):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
        )

    logging.info('Created model')
    model.config.torch_dtype = torch.bfloat16
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config.use_cache = False  # turn off when gradient checkpointing is enabled

    return model


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f'trainable params: {trainable_params} || '
        f'all params: {all_param} || '
        f'trainable: {100 * trainable_params / all_param}'
    )


def apply_tokenizer(x: dict, tokenizer: transformers.PreTrainedTokenizer):
    return {
        'input_ids': tokenizer.apply_chat_template(
            x['input_ids'],
            tokenize=True,
            add_generation_prompt=False,
            truncation=False,
            padding='max_length')
    }


def prepare_training_data(rank: int, tokenizer: transformers.PreTrainedTokenizer, filename: Path) -> Dataset:
    if rank > 0:
        dist.barrier()
    fp = Hasher.hash((filename, tokenizer.init_inputs, tokenizer.init_kwargs, tokenizer.chat_template, apply_tokenizer))
    rank0_log(f'{filename} = {fp}')
    cached = f'cache/{fp}'
    if rank == 0 and not os.path.exists(cached):  # Only test on rank 0 to avoid filling negative NFS cache.
        rank0_log(f'Caching {filename}')
        os.makedirs('cache', exist_ok=True)
        data = [json.loads(line) for line in open(filename, 'r')]
        dataset = Dataset.from_dict({'input_ids': data})
        dataset = dataset.map(
            apply_tokenizer,
            fn_kwargs={'tokenizer': tokenizer},
            num_proc=32,
            new_fingerprint=fp,
        ).map(lambda x: {'labels': x['input_ids']}, num_proc=8, batch_size=128)
        dataset = filter_long(dataset, tokenizer.model_max_length)
        dataset.save_to_disk(cached)
    else:
        dataset = load_from_disk(cached)

    if rank == 0:
        dist.barrier()

    # Wait for everyone to get here.
    dist.barrier()
    return dataset


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args: argparse.Namespace) -> Dict:
    conv_config = json.load(open(args.conv_config))
    tokenizer, data_collator = process_chat_template(
        tokenizer, args.tokenizer_name_or_path, None,
        True,  # Force template encoding
        conv_config['response'], conv_config['instruction'])
    rank = dist.get_rank()
    base_path = Path(args.dataset_path)
    rank0_log(f'Using {base_path}/{TRAIN_FILE} as the train data')
    train_dataset = prepare_training_data(rank, tokenizer, base_path / TRAIN_FILE)

    eval_dataset = None
    if args.do_eval:
        rank0_log(f'Using {base_path}/{EVAL_FILE} as the eval data')
        eval_dataset = prepare_training_data(rank, tokenizer, base_path / EVAL_FILE)

    if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    # TODO: Need to think about how to make this work for better training efficiency.
    # if args.group_by_length:
    #    train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    return dict(
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=eval_dataset
    )


def train():
    dist.init_distributed()
    torch.autograd.set_detect_anomaly(True)
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments
    ))
    model_args, data_args, training_args, _ = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))

    rank0_log(args)
    args.tokenizer_name_or_path = args.tokenizer_name_or_path or args.model_name_or_path

    # Tokenizer
    logging.info(f'Loading tokenizer {args.tokenizer_name_or_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        padding_side='right',
        model_max_length=args.context_length,
        use_fast=False,  # Fast tokenizer giving issues.
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    set_seed(args.seed)
    data_module = make_data_module(tokenizer=tokenizer, args=args)

    model = get_accelerate_model(args)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    trainer.model.train()
    # Does this save do something sensible?
    # model.save_pretrained(args.output_dir)
    if dist.get_rank() == 0:
        print_trainable_parameters(model)

    all_metrics = {'run_name': args.run_name}
    # Training
    rank0_log('*** Train ***')
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()
    all_metrics.update(metrics)

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as fout:
        fout.write(json.dumps(all_metrics))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train()
