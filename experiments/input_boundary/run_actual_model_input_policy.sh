#!/usr/bin/env bash
set -euo pipefail

cd /home2/sangjin/codex
rm -f actual_model_input_policy_orin.csv

modes=(pageable pinned mapped_zc managed device_preloaded)

for trial in 1 2 3; do
  for mode in "${modes[@]}"; do
    echo "trial=${trial} victim=mobilenetv2 attacker=none mode=${mode}"
    python3 actual_model_input_policy.py \
      --victim-model mobilenetv2 \
      --victim-image-size 640 \
      --input-mode "${mode}" \
      --attacker-model none \
      --repeats 50 \
      --warmup 10 \
      --csv actual_model_input_policy_orin.csv
  done

  for mode in "${modes[@]}"; do
    echo "trial=${trial} victim=mobilenetv2 attacker=gpt2 mode=${mode}"
    python3 actual_model_input_policy.py \
      --victim-model mobilenetv2 \
      --victim-image-size 640 \
      --input-mode "${mode}" \
      --attacker-model gpt2 \
      --attacker-policy default \
      --attacker-seq-len 512 \
      --repeats 50 \
      --warmup 10 \
      --attacker-warmup-s 3 \
      --csv actual_model_input_policy_orin.csv
  done

  for mode in "${modes[@]}"; do
    echo "trial=${trial} victim=gpt2 attacker=none mode=${mode}"
    python3 actual_model_input_policy.py \
      --victim-model gpt2 \
      --victim-seq-len 512 \
      --input-mode "${mode}" \
      --attacker-model none \
      --repeats 50 \
      --warmup 10 \
      --csv actual_model_input_policy_orin.csv
  done

  for mode in "${modes[@]}"; do
    echo "trial=${trial} victim=gpt2 attacker=mobilenetv2 mode=${mode}"
    python3 actual_model_input_policy.py \
      --victim-model gpt2 \
      --victim-seq-len 512 \
      --input-mode "${mode}" \
      --attacker-model mobilenetv2 \
      --attacker-policy default \
      --attacker-image-size 640 \
      --repeats 50 \
      --warmup 10 \
      --attacker-warmup-s 3 \
      --csv actual_model_input_policy_orin.csv
  done
done
