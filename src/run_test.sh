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

# Extract model family from the model name, e.g., llama3.2:1b -> llama
extract_model_family() {
  echo "$1" | cut -d':' -f1 | sed 's/[0-9.]*$//'
}

# Loop over each model and dataset
for model in "${models[@]}"; do
  model_family=$(extract_model_family "$model")
  for dataset in "${datasets[@]}"; do
    # Convert dataset name to lower case and remove hyphens
    dataset_filename=$(echo "$dataset" | tr '[:upper:]' '[:lower:]' | tr -d '-')
    echo $model
    echo $model_family
    echo $dataset_filename
    input_file="${dataset}/${dataset_filename}_50_samples.jsonl"
    output_file_with_instruction="test_${model_family}_${dataset_filename}_with_instruction.jsonl"
    output_file_without_instruction="test_${model_family}_${dataset_filename}_without_instruction.jsonl"
    echo $input_file
    echo $output_file_with_instruction
    echo $output_file_without_instruction

    # Run inference with instruction
    python3 src/inference.py --input "$input_file" --output "$output_file_with_instruction" --instruction --model "$model" --dataset_type "$dataset"

    # Run inference without instruction
    python3 src/inference.py --input "$input_file" --output "$output_file_without_instruction" --model "$model" --dataset_type "$dataset"
  done
done