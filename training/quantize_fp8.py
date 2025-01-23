#!/usr/bin/python

import argparse
import functools
import json
import os

from datasets import Dataset, load_dataset
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer


def process_and_tokenize(tokenizer, max_seq_length, example):
    text = tokenizer.apply_chat_template(example['messages'], add_generation_prompt=False, tokenize=False)
    return tokenizer(text, padding=False, max_length=max_seq_length, truncation=True, add_special_tokens=False, return_attention_mask=True)


def quantize(args):
    pretrained_model_dir = args.model_dir
    model_root = os.path.dirname(pretrained_model_dir).rstrip('/')
    model_name = os.path.basename(pretrained_model_dir).rstrip('/')
    quantized_model_dir = args.output_dir or f'{model_root}/{model_name}-FP8-dynamic'

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)

    ignore = ['lm_head']  # Works for Llama + Qwen
    recipe = QuantizationModifier(targets='Linear', scheme='FP8_DYNAMIC', ignore=ignore)
    ds = None
    if args.calibration_dataset:
        if args.calibration_dataset.endswith('.json'):
            ds = Dataset.from_dict({'messages': [json.loads(line) for line in open(args.calibration_dataset, 'rt')]})
        else:
            ds = load_dataset(args.calibration_dataset, split=args.calibration_split)
        ds = ds.shuffle(seed=42).select(range(args.num_samples))
        ds = ds.map(functools.partial(process_and_tokenize, tokenizer, args.max_seq_len), remove_columns=ds.column_names)

        recipe = [
            QuantizationModifier(
                targets='Linear',
                scheme='FP8',
                ignore=['lm_head'],
            ),
        ]

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_dir,
        device_map='auto',
        torch_dtype='bfloat16',
        attn_implementation='flash_attention_2',
    )
    print(model)

    oneshot(
        model=model,
        recipe=recipe,
        dataset=ds,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.num_samples,
    )
    model.save_pretrained(quantized_model_dir, save_compressed=True)
    tokenizer.save_pretrained(quantized_model_dir)


def main():
    parser = argparse.ArgumentParser(description='Quantize a model into fp8')
    parser.add_argument('--model-dir', required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', help='Override default output path')
    parser.add_argument('--calibration-dataset', help='Dataset for calibration')
    parser.add_argument('--calibration-split', default='train_sft')
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--max-seq-len', type=int, default=2048)

    args = parser.parse_args()
    quantize(args)


if __name__ == '__main__':
    main()
