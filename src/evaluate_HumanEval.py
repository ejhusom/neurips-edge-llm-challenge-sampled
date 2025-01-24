# %%
import json
from evaluate import load
import argparse
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# %%
def extract_and_write_fields(input_file):
    # Generate the output file name by appending "_extracted" to the input file name 
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_extracted{ext}"

    extracted_data = []

    with open(input_file, 'r') as infile:
        for line in infile:
            try:
                data = json.loads(line)
                extracted_entry = {
                    "task_id": data.get("indata", {}).get("task_id"),
                    "test": data.get("indata", {}).get("test"),
                    "model": data.get("output", {}).get("model"),
                    "response": data.get("output", {}).get("response")
                }
                extracted_data.append(extracted_entry)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")

    # Write the extracted fields to a new JSONL file
    with open(output_file, 'w') as outfile:
        for entry in extracted_data:
            outfile.write(json.dumps(entry) + '\n')

    print(f"Extracted data has been written to {output_file}")
    return output_file

# %%
def postprocess_responses(responses):
    """Remove <ANS> and </ANS> from responses."""
    return [response.replace("<ANS>", "").replace("</ANS>", "") for response in responses]

# %%
def postprocess_extracted_jsonl_file(input_file):
    """Read, process, and write cleaned JSONL responses."""

    # Generate the output file name by appending "_extracted" to the input file name 
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_cleared{ext}"

    cleaned_data = []
    
    # Read the JSONL file
    with open(input_file, 'r') as infile:
        for line in infile:
            data = json.loads(line.strip())
            
            # Ensure 'response' exists, and process it
            if 'response' in data:
                data['response'] = data['response'].replace("<ANS>", "").replace("</ANS>", "")
            
            cleaned_data.append(data)
    
    # Write the cleaned data back to a new JSONL file
    with open(output_file, 'w') as outfile:
        for entry in cleaned_data:
            outfile.write(json.dumps(entry) + '\n')
            
    print(f"Cleaned responses have been written to {output_file}.")
    return output_file

# %%
def evaluate_pass_at_k(cleared_jsonl_file, k=[1]):
    # Load the evaluation function
    code_eval = load("code_eval")
    
    # Prepare lists for test cases, responses, and model tracking
    test_cases = []
    candidates = []
    model_list = []
    
    # Read the JSONL file
    with open(cleared_jsonl_file, 'r') as infile:
        for line in infile:
            entry = json.loads(line.strip())
            
            # Extract the test case and response
            test = entry.get("test", "")
            response = entry.get("response", "")
            model = entry.get("model", "unknown_model")  # Use 'unknown_model' if the field is missing
            
            if test and response:
                # Append the test, response, and model to the respective lists
                test_cases.append(test)
                candidates.append([response])  # Candidates need to be nested for pass@k
                model_list.append(model)
    
    # Compute pass@k
    pass_at_k, results = code_eval.compute(references=test_cases, predictions=candidates, k=k)

    # Create a dictionary to track models and their performances
    model_performance = {model: pass_at_k for model in set(model_list)}
    
    return pass_at_k, model_performance


# %%
def main(input_file, k_values=[1]):
    # Step 1: Extract and write required fields to calculate pass@k
    extracted_file = extract_and_write_fields(input_file)
    
    # Step 2: Postprocess the extracted JSONL file, clear responses from <ANS> tags
    cleaned_file = postprocess_extracted_jsonl_file(extracted_file)
    
    # Step 3: Evaluate pass@k
    pass_at_k, model_performance = evaluate_pass_at_k(cleaned_file, k=k_values)

    # Print the results
    print(f"Overall pass@k for k={k_values}: {pass_at_k}")
    print("Model performance breakdown:", model_performance)


# %%
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate pass@k for a given input JSONL file.")
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file.")
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1],
        help="List of k values to calculate pass@k (default is [1])."
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.input_file, args.k_values)

