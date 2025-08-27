#!/usr/bin/env bash

time=$(date "+%Y%m%d-%H%M%S")
name=In_Model_Merging_backbone_training_l10_e30

CUDA_VISIBLE_DEVICES=$1 python -u train.py --model vgg19 --b 324 --name ${name} > logs/${time}_train_${name}.log 2>&1 &
