#!/bin/bash
topk_knn=$1
align=$2
atemp=$3
tau=$4
dataset=$5
llm_model=$6 

python main.py --rs_type General --clear_checkpoints --dataset $dataset --model_name AlphaFree --topk_knn $topk_knn --patience 20 --cuda 0 --no_wandb --train_norm --pred_norm --neg_sample 256 --lm_model $llm_model --tau $tau --infonce 1 --align $align --align_temperature $atemp --seed 101
exit 0
