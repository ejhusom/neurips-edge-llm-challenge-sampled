# Author: andrekaa (https://github.com/andrekaa)
import ollama
import jsonlines
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a JSONL file with a specified model.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument('--output', type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument('--model', type=str, nargs='+', required=True, help="Model name to use for processing.")
    
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    model_names = args.model

    # Process the input file, run inference, save result
    with jsonlines.open(input_file, mode='r') as prompts:
        with jsonlines.open(output_file, mode='a') as writer:
            for model_name in model_names:
                ollama.pull(model_name)
                for prompt in prompts:
                    response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': prompt['input']}])
                    writer.write({'indata': prompt, 'output': response})
                ollama.delete(model_name)

if __name__ == "__main__":
    main()
