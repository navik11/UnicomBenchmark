#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

MODEL="Qwen/Qwen-VL-Chat-Int4"
# Make sure this path is correct for your Kaggle environment
DATA="../dataset/VQA/Crop_Disease_train_qwenvl.json"

export CUDA_VISIBLE_DEVICES=0

# LAUNCHER: Use 'deepspeed' not 'python'
# FLAGS: All boolean flags are now just the flag name without 'True'
deepspeed finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --fp16 \
    --fix_vit \
    --output_dir output_qwen \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 2048 \
    --lazy_preprocess \
    --gradient_checkpointing \
    --use_lora \
    --deepspeed finetune/ds_config_zero2.json