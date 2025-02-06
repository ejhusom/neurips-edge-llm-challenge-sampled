# %%
import argparse
import os
import sys
import re
import json
from collections import defaultdict

# %%
def extract_ans_from_answer(answer: str, eos=None):
    """
    Extracts and cleans the answer from a string, ensuring numeric values are properly converted.
    Returns an integer if possible, otherwise returns the cleaned string.
    """
    if eos:
        answer = answer.split(eos)[0].strip()

    # Ensure '####' is in the string before splitting
    if '####' in answer:
        answer = answer.split('####')[-1].strip()

    # Remove unwanted characters while keeping numbers
    answer = answer.strip(" $%g")

    # Remove commas from numeric values (e.g., "100,000" → "100000")
    answer = re.sub(r',', '', answer)

    # Try converting to an integer
    try:
        return int(answer)
    except ValueError:
        return answer  # Return as a string if it's not a valid integer

# %%
def extract_real_answers_from_responses(response: str):
    """
    Extract the final numerical answer from a formatted response string.
    Returns an integer if a valid answer is found, None otherwise.
    
    Args:
        response (str): The input string containing an answer
        
    Returns:
        int or None: The extracted answer as an integer, or None if invalid
    """
    # Return None for empty or invalid basic inputs
    if not response or response == "ANS":
        return None
        
    # Clean up the response by removing basic ANS tags and percentage signs
    cleaned = re.sub(r"(</?ANS>|ANS\s*<|ANS\s*>)", " ", response).strip()
    
    # Function to convert string with possible commas to number
    def parse_number(num_str):
        # Remove commas and percentage signs
        num_str = num_str.replace(',', '').replace('%', '')
        try:
            return int(float(num_str))
        except (ValueError, TypeError):
            return None
    
    # Look for final result after equals sign first
    equals_match = re.search(r'=\s*\$?([0-9,]+(?:\.[0-9]+)?%?)\s*(?:>>)?(?:\s*>)?$', cleaned)
    if equals_match:
        return parse_number(equals_match.group(1))
    
    # Handle special case with </1 pattern
    special_match = re.search(r'<\/\d+<\/ANS>', cleaned)
    if special_match:
        num_match = re.search(r'<\/(\d+)', cleaned)
        if num_match:
            return parse_number(num_match.group(1))
            
    # If no equals sign with final result, look for simple numeric value
    # Updated to handle commas, percentages, and dollar signs
    simple_match = re.search(r'\$?([0-9,]+(?:\.[0-9]+)?%?)\s*(?:>>)?(?:\s*>)?$', cleaned)
    if simple_match:
        return parse_number(simple_match.group(1))
            
    # Look for properly terminated expressions
    if '>>' in cleaned or cleaned.endswith('>'):
        expression_match = re.search(r'(?:\/[0-9]+)?[^=]*?(\d+(?:\.[0-9]+)?%?)\s*(?:>>|>)$', cleaned)
        if expression_match:
            return parse_number(expression_match.group(1))
    
    # If we haven't found a valid pattern, return None
    return None

# %%
def extract_and_write_fields(input_file):
    # Generate output file name
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_extracted{ext}"

    extracted_data = []

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                data = json.loads(line)
                extracted_entry = {
                    "answer": data.get("indata", {}).get("answer", "").strip(),
                    "model": data.get("output", {}).get("model", ""),
                    "response": data.get("output", {}).get("response", "").strip()
                }
                extracted_data.append(extracted_entry)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")

    # Write extracted data to new JSONL file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in extracted_data:
            json.dump(entry, outfile)
            outfile.write('\n')  

    print(f"Extracted data has been written to {output_file}")
    return output_file


# %%
def postprocess_extracted_jsonl_file(input_file):
    # Generate output file name
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_cleared{ext}"

    cleaned_data = []
    invalid_count = 0  # Counter for invalid responses

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                data = json.loads(line)

                # Ensure fields exist, else default to empty string
                raw_answer = data.get("answer", "").strip()
                raw_response = data.get("response", "").strip()

                # Processed answer and response
                cleaned_answer = extract_ans_from_answer(raw_answer)
                cleaned_response = extract_real_answers_from_responses(raw_response)

                # Count invalid responses
                if cleaned_response is None:
                    invalid_count += 1

                cleaned_entry = {
                    "answer": cleaned_answer,
                    "model": data.get("model", ""),
                    "response": cleaned_response if cleaned_response is not None else "INVALID"
                }
                cleaned_data.append(cleaned_entry)

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")

    # Write cleaned data to new JSONL file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in cleaned_data:
            json.dump(entry, outfile)
            outfile.write('\n')  # Ensures newline for JSONL format

    print(f"Post-processing completed. Cleaned data saved to {output_file}")
    print(f"Number of invalid responses: {invalid_count}")
    return output_file


# %%
def evaluate_accuracy(input_file):
    """Evaluate accuracy by comparing extracted answers with model responses, including invalid ones."""
    total_responses = 0  # Includes both valid and invalid responses
    correct = 0
    valid_responses = 0
    removed_responses = 0

    model_correct_counts = defaultdict(int)
    model_total_counts = defaultdict(int)

    invalid_responses = []  # Store invalid responses

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                data = json.loads(line)
                
                # Ensure required fields exist
                real_answer = data.get("answer")
                model_response = data.get("response")
                model_name = data.get("model", "Unknown Model")

                total_responses += 1  # Count every response, even invalid ones
                model_total_counts[model_name] += 1  # Track total responses per model

                # Only consider valid numerical responses
                if isinstance(model_response, int):
                    valid_responses += 1

                    if model_response == real_answer:
                        correct += 1
                        model_correct_counts[model_name] += 1
                else:
                    removed_responses += 1
                    invalid_responses.append(model_response)

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")

    # Calculate overall accuracy over ALL responses (valid + invalid)
    overall_accuracy = (correct / total_responses * 100) if total_responses > 0 else 0

    # Calculate per-model accuracy
    model_accuracies = {
        model: (model_correct_counts[model] / model_total_counts[model] * 100) if model_total_counts[model] > 0 else 0
        for model in model_total_counts
    }

    # Print results
    print("\nAccuracy Results by Model:")
    for model, accuracy in model_accuracies.items():
        print(f"- {model}: {model_correct_counts[model]}/{model_total_counts[model]} correct ({accuracy:.2f}%)")

    print(f"\nOverall Accuracy: {correct}/{total_responses} correct ({overall_accuracy:.2f}%)")
    print(f"Valid Responses: {valid_responses}, Removed Responses (Invalid): {removed_responses}")

    #if invalid_responses:
    #    print("\nSample Invalid Responses (First 5):")
    #    print(invalid_responses[:5])  # Display first 5 invalid responses

    return {
        "overall_accuracy": overall_accuracy,
        "model_accuracies": model_accuracies
    }


# %%
def main(input_file):
    """Processes the outputs of gsm8k dataset, extracts necessary fields, cleans them, and evaluates accuracy."""

    print("\n[Step 1] Extracting necessary fields...")
    extracted_file = extract_and_write_fields(input_file)
    
    print("\n[Step 2] Cleaning extracted responses...")
    cleaned_file = postprocess_extracted_jsonl_file(extracted_file)
    
    print("\n[Step 3] Evaluating accuracy...")
    results = evaluate_accuracy(cleaned_file)

    print("\n[Process Complete] Accuracy Evaluation Done.\n")
    
    return results


# %%
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract, clean, and evaluate model responses from a JSONL dataset."
    )
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file.")

    # Parse arguments
    args = parser.parse_args()

    # Validate input file existence
    if not os.path.isfile(args.input_file):
        print(f"Error: The file '{args.input_file}' does not exist.")
        sys.exit(1)

    print("\nStarting processing for:", args.input_file)

    # Call the main function
    results = main(args.input_file)

    # Handle the case where no results are returned
    if results is None:
        print("Processing failed. No accuracy results available.")
        sys.exit(1)

    # Print the results in a structured format
    print("\nModel Performance Breakdown:")
    print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
    print("\nPer-Model Accuracies:")
    for model, accuracy in results['model_accuracies'].items():
        print(f"{model}: {accuracy:.2f}%")

    print("\nProcess completed successfully!")


