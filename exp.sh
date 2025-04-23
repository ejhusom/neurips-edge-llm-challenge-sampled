#!/bin/bash

# Define datasets and models
datasets=("BIG-Bench-Hard" "CommonSenseQA" "GSM8K" "HumanEval" "TruthfulQA")
models=(
    # "gemma2:2b-instruct-q3_K_S"
    # "gemma2:2b-instruct-q3_K_M" 
    # "gemma2:2b-instruct-q3_K_L" 
    "qwen2.5:0.5b-instruct-q3_K_S"
    "qwen2.5:0.5b-instruct-q3_K_M" 
    "qwen2.5:0.5b-instruct-q3_K_L" 
    "qwen2.5:0.5b-instruct-q4_K_S"
    "qwen2.5:0.5b-instruct-q4_0" 
    "qwen2.5:0.5b-instruct-q4_K_M"
    "qwen2.5:0.5b-instruct-q4_1" 
    "qwen2.5:0.5b-instruct-q8_0" 
    "qwen2.5:0.5b-instruct-fp16"
    "qwen2.5:1.5b-instruct-q3_K_S"
    "qwen2.5:1.5b-instruct-q3_K_M" 
    "qwen2.5:1.5b-instruct-q3_K_L" 
    "qwen2.5:1.5b-instruct-q4_K_S"
    "qwen2.5:1.5b-instruct-q4_0" 
    "qwen2.5:1.5b-instruct-q4_K_M"
    "qwen2.5:1.5b-instruct-q4_1" 
    "llama3.2:1b-instruct-q3_K_S"
    "llama3.2:1b-instruct-q3_K_M" 
    "llama3.2:1b-instruct-q3_K_L" 
    "llama3.2:1b-instruct-q4_K_S" 
    "llama3.2:1b-instruct-q4_0" 
    "llama3.2:1b-instruct-q4_K_M"
    "llama3.2:1b-instruct-q4_1" 
    "llama3.2:1b-instruct-q8_0" 
    "llama3.2:1b-instruct-fp16"
)

# Loop over datasets and models
for dataset in "${datasets[@]}"; do
    input_file=""
    case $dataset in
        "BIG-Bench-Hard")
            input_file="BIG-Bench-Hard/collated_bbh_200_samples.jsonl"
            ;;
        "CommonSenseQA")
            input_file="CommonsenseQA/commonsenseqa_200_samples.jsonl"
            ;;
        "GSM8K")
            input_file="GSM8K/gsm8k_200_samples.jsonl"
            ;;
        "HumanEval")
            input_file="HumanEval/HumanEval.jsonl"
            ;;
        "TruthfulQA")
            input_file="TruthfulQA/truthfulQA_MC_200_samples.jsonl"
            ;;
    esac

    for model in "${models[@]}"; do
        output_file="$(echo $dataset | tr '[:upper:]' '[:lower:]')_${model//:/_}.jsonl"
        python3 src/inference.py --input "$input_file" --output "$output_file" --instruction --dataset_type "$dataset" --model "$model" --repeat 5 --timeout 20
    done
done
