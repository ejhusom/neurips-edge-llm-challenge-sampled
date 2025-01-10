# Run script for the following test cases
# Datasets:
# 1. GSM8K
# 2. TruthfulQA
# 3. HumanEval
# 4. CommonSenseQA
# 5. BIG-Bench-Hard
# Models:
# 1. gemma2:2b-instruct-q3_K_S
# 2. qwen2.5:0.5b
# 3. llama3.2:1b
# All test cases should be run with and without instruction

# Datasets and models
datasets=("GSM8K" "TruthfulQA" "HumanEval" "CommonSenseQA" "BIG-Bench-Hard")
models=("gemma2:2b-instruct-q3_K_S" "qwen2.5:0.5b" "llama3.2:1b")

# Loop over each model and dataset
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    input_file="${dataset,,}/${dataset,,}_50_samples.jsonl"
    output_file_with_instruction="test_${model,,}_${dataset,,}_with_instruction.jsonl"
    output_file_without_instruction="test_${model,,}_${dataset,,}_without_instruction.jsonl"

    # Run inference with instruction
    python3 src/inference.py --input "$input_file" --output "$output_file_with_instruction" --instruction --model "$model" --dataset_type "$dataset"

    # Run inference without instruction
    python3 src/inference.py --input "$input_file" --output "$output_file_without_instruction" --model "$model" --dataset_type "$dataset"
  done
done