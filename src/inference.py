# Author: andrekaa (https://github.com/andrekaa)
import ollama
import jsonlines
import argparse
import time
import os

def process_prompt(dataset_type, prompt):
    """
    Process the prompt based on the dataset type and format it for the model.
    """
    if dataset_type == "BIG-Bench-Hard":
        return f"{prompt['input']}"
    
    elif dataset_type == "CommonSenseQA":
        question = prompt['question']['stem']
        choices = "\n".join(
            [f"({choice['label']}) {choice['text']}" for choice in prompt['question']['choices']]
        )
        return f"Question: {question}\nOptions:\n{choices}"

    elif dataset_type == "GSM8K":
        return prompt['question']

    elif dataset_type == "HumanEval":
        return f"{prompt['prompt']}\n# Complete this implementation"

    elif dataset_type == "TruthfulQA":
        question = prompt['question']
        choices = "\n".join(prompt['mc1_targets'].keys())
        return f"Question: {question}\nChoices:\n{choices}"

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def read_first_line(file_path):
    with jsonlines.open(file_path, mode='r') as reader:
        for line in reader:
            return line

def test_process_prompt():
    datasets = {
        "BIG-Bench-Hard": "./BIG-Bench-Hard/collated_bbh_200_samples.jsonl",
        "CommonSenseQA": "./CommonsenseQA/commonsenseqa_200_samples.jsonl",
        "GSM8K": "./GSM8K/gsm8k_200_samples.jsonl",
        "HumanEval": "./HumanEval/HumanEval.jsonl",
        "TruthfulQA": "./TruthfulQA/truthfulQA_MC_200_samples.jsonl"
    }

    for dataset_type, file_path in datasets.items():
        if os.path.exists(file_path):
            first_prompt = read_first_line(file_path)
            formatted_prompt = process_prompt(dataset_type, first_prompt)
            print(f"Formatted prompt for {dataset_type}:\n{formatted_prompt}\n")
        else:
            print(f"File not found for dataset {dataset_type}: {file_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a JSONL file with a specified model.")
    parser.add_argument('--input', type=str, help="Path to the input JSONL file.")
    parser.add_argument('--output', type=str, help="Path to the output JSONL file.")
    parser.add_argument('--model', type=str, nargs='+', help="Model name to use for processing.")
    parser.add_argument('--dataset_type', type=str, help="Dataset type (e.g., BIG-Bench-Hard, CommonSenseQA, GSM8K, HumanEval, TruthfulQA).")
    parser.add_argument('--test', action='store_true', help="Test the process_prompt function with the first line of the input file.")

    args = parser.parse_args()

    if args.test:
        test_process_prompt()
        return

    if not (args.input and args.output and args.model and args.dataset_type):
        parser.error("the following arguments are required: --input, --output, --model, --dataset_type")

    input_file = args.input
    output_file = args.output
    model_names = args.model
    dataset_type = args.dataset_type

    counter = 0

    # Process the input file, run inference, save result
    with jsonlines.open(input_file, mode='r') as prompts:
        with jsonlines.open(output_file, mode='a') as writer:
            for model_name in model_names:
                ollama.pull(model_name)
                
                for prompt in prompts:
                    formatted_prompt = process_prompt(dataset_type, prompt)
                    response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': formatted_prompt}])
                    response_dict = response.__dict__
                    response_dict["message"] = response_dict["message"].__dict__
                    writer.write({'indata': prompt, 'formatted_prompt': formatted_prompt, 'output': response_dict})
                    print(f"Completed prompt {counter} for model {model_name} at time {time.time()}"
                    
                    counter += 1

                ollama.delete(model_name)

if __name__ == "__main__":
    main()
