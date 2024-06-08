#!/bin/bash
export PYTHONPATH=/root/fairseq:$PYTHONPATH # set your fairseq path
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
cd /root/EmoBench/EmoBench/preprocess

export CUDA_VISIBLE_DEVICES=0


python extract_features.py  \
--source_path='/mnt/lustre/sjtu/home/zym22/data/emotion_recognition/Emotion/RAVDESS' \
--target_path='/mnt/lustre/sjtu/home/zym22/models/data2vec/RAVDESS' \
--fairseq_root='/root/fairseq' \
--checkpoint_dir='/mnt/lustre/sjtu/home/zym22/models/data2vec_base.pt' \
--user_module_path='/mnt/lustre/sjtu/home/zym22/fairseq/example/data2vec' \
--model_name='data2vec' \
--layer=12 \
--granularity='utterance' \
--dataset_name 'ravdess' \