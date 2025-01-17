import ollama
import jsonlines
import argparse
import os
import time
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def process_prompt(dataset_type, prompt, instruction=""):
    """
    Process the prompt based on the dataset type and format it for the model.
    """
    if dataset_type == "BIG-Bench-Hard":
        return f"{prompt['input']}" + instruction
    
    elif dataset_type == "CommonSenseQA":
        question = prompt['question']['stem']
        choices = "\n".join(
            [f"({choice['label']}) {choice['text']}" for choice in prompt['question']['choices']]
        )
        return f"Question: {question}\nOptions:\n{choices}" + instruction

    elif dataset_type == "GSM8K":
        return prompt['question'] + instruction

    elif dataset_type == "HumanEval":
        return f"{prompt['prompt']}" + instruction

    elif dataset_type == "TruthfulQA":
        question = prompt['question']
        choices = "\n".join(prompt['mc1_targets'].keys())
        return f"Question: {question}\nChoices:\n{choices}" + instruction

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def read_first_line(file_path):
    with jsonlines.open(file_path, mode='r') as reader:
        for line in reader:
            return line

def test_process_prompt(instruction_general="", instruction_humaneval=""):
    datasets = {
        "BIG-Bench-Hard": "./BIG-Bench-Hard/collated_bbh_200_samples.jsonl",
        "CommonSenseQA": "./CommonsenseQA/commonsenseqa_200_samples.jsonl",
        "GSM8K": "./GSM8K/gsm8k_200_samples.jsonl",
        "HumanEval": "./HumanEval/HumanEval.jsonl",
        "TruthfulQA": "./TruthfulQA/truthfulQA_MC_200_samples.jsonl"
    }

    for dataset_type, file_path in datasets.items():
        if dataset_type == "HumanEval":
            instruction = instruction_humaneval
        else:
            instruction = instruction_general
        if os.path.exists(file_path):
            first_prompt = read_first_line(file_path)
            formatted_prompt = process_prompt(dataset_type, first_prompt, instruction)
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
    parser.add_argument('--instruction', action='store_true', help="Include instruction in the prompt.")
    parser.add_argument('--timeout', type=int, default=3600, help="Timeout in seconds for each prompt.")

    args = parser.parse_args()

    # instruction_general = "\n\nNever print any extra explanations about how the response was generated. Print only the answer surrounded by <ANS> and </ANS>."
    # instruction_humaneval = "\n\n# Complete the function implementation based on the provided docstring, and never print any extra explanations about how the code was generated. Print only the answer surrounded by <ANS> and </ANS>."
    instruction_general = "\n\nPrint only the answer surrounded by <ANS> and </ANS>. Never print any extra explanations about how the response was generated."
    instruction_humaneval = "\n\n# Complete the function implementation based on the provided docstring, and print only the completed function surrounded by <ANS> and </ANS>. Never print any extra explanations about how the code was generated."

    if args.instruction:
        if args.dataset_type == "HumanEval":
            instruction = instruction_humaneval
        else:
            instruction = instruction_general
    else:
        instruction = ""

    if args.test:
        test_process_prompt(instruction_general, instruction_humaneval)
        return

    if not (args.input and args.output and args.model and args.dataset_type):
        parser.error("the following arguments are required: --input, --output, --model, --dataset_type")

    input_file = args.input
    output_file = args.output
    model_names = args.model
    dataset_type = args.dataset_type
    timeout = args.timeout

    counter = 0

    # Process the input file, run inference, save result
    with jsonlines.open(input_file, mode='r') as prompts:
        with jsonlines.open(output_file, mode='a') as writer:
            for model_name in model_names:
                ollama.pull(model_name)
                
                for prompt in prompts:
                    formatted_prompt = process_prompt(dataset_type, prompt, instruction)
                    print(formatted_prompt)

                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout)
                    try:
                        # Uncomment below for chat API
                        # response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': formatted_prompt}])
                        # response_dict = response.__dict__
                        # response_dict["message"] = response_dict["message"].__dict__
                        # writer.write({'indata': prompt, 'formatted_prompt': formatted_prompt, 'output': response_dict})
                        # print(response_dict["message"]["content"])
                        # print(f"Completed prompt {counter} for model {model_name} at time {time.time()}")

                        # Generate API
                        response = ollama.generate(
                            model=model_name, 
                            # messages=[{'role': 'user', 'content': formatted_prompt}], 
                            prompt=formatted_prompt,
                            options = {
                                'temperature': 0
                            }
                        )
                        response_dict = response.__dict__
                        writer.write({'indata': prompt, 'formatted_prompt': formatted_prompt, 'output': response_dict})
                        print(response.response)
                        print(f"Completed prompt {counter} for model {model_name} at time {time.time()}")
                        print("========================================================")
                    except TimeoutException:
                        print(f"Timeout for prompt {counter} for model {model_name} at time {time.time()}")
                        print("========================================================")
                    finally:
                        signal.alarm(0)
                    
                    counter += 1
if __name__ == "__main__":
    main()
