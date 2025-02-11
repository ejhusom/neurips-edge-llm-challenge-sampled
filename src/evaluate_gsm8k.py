# %%
import argparse
import os
import sys
import re
import json
from collections import defaultdict
import pandas as pd
from typing import List

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
def extract_real_answers(response: str):
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
def extract_fields_from_jsonl(file_path: str, fields: List[str]) -> pd.DataFrame:
    """
    Extract specified fields from a JSONL file and return them as a DataFrame.
    
    :param file_path: Path to the JSONL file.
    :param fields: List of field names to extract, using dot notation for nested fields.
    :return: DataFrame with extracted columns if available.
    """
    data = []
    
    # Read JSONL file and load data
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line[:100]}...\nError: {e}")
    
    if not data:
        print("No valid data found in the file.")
        return pd.DataFrame()
    
    # Normalize nested JSON data
    df = pd.json_normalize(data)
    
    # Check for missing columns
    missing_columns = [col for col in fields if col not in df.columns]
    if missing_columns:
        print(f"Columns not found: {missing_columns}. Available columns are:\n{df.columns.tolist()}")
    
    # Return DataFrame with only requested fields if they exist
    available_columns = [col for col in fields if col in df.columns]
    return df[available_columns] if available_columns else pd.DataFrame()


# %%
def clean_dataframe(df):
    cleaned_data = []
    invalid_count = 0  

    for index, row in df.iterrows():
        try:
            # Extract answer and response from the respective columns in the dataframe
            raw_answer = row.get("indata.answer", "").strip()
            raw_response = row.get("output.response", "").strip()

            # Processed answer and response
            cleaned_answer = extract_ans_from_answer(raw_answer)
            cleaned_response = extract_real_answers(raw_response)

            # Count invalid responses
            if cleaned_response is None:
                invalid_count += 1

            # Store cleaned data in a dictionary
            cleaned_entry = {
                "answer": cleaned_answer,
                "model": row.get("output.model", ""),
                "response": cleaned_response if cleaned_response is not None else "INVALID"
            }
            cleaned_data.append(cleaned_entry)
            #print(cleaned_entry)  
            
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    # Convert cleaned data into a DataFrame
    cleaned_df = pd.DataFrame(cleaned_data)

    # Return the cleaned DataFrame
    print(f"Number of invalid responses: {invalid_count}")
    return cleaned_df


# %%
def evaluate_accuracy(df_cleaned):
    """Evaluate accuracy by comparing extracted answers with model responses, including invalid ones."""
    total_responses = 0  # Includes both valid and invalid responses
    correct = 0
    valid_responses = 0
    removed_responses = 0

    model_correct_counts = defaultdict(int)
    model_total_counts = defaultdict(int)

    invalid_responses = []  

    for index, row in df_cleaned.iterrows():
        try:
            # Extract real answer and model response from respective columns
            cleaned_answer = row.get("answer")
            cleaned_response = row.get("response")
            model_name = row.get("model", "Unknown Model")

            total_responses += 1  # Count every response, even invalid ones
            model_total_counts[model_name] += 1  # Track total responses per model

            # Only consider valid numerical responses (if available)
            if isinstance(cleaned_response, int):
                valid_responses += 1
                if cleaned_response == cleaned_answer:
                    correct += 1
                    model_correct_counts[model_name] += 1
            else:
                removed_responses += 1
                invalid_responses.append(cleaned_response) 

        except Exception as e:
            print(f"Error processing row {index}: {e}")

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

    # Optionally, print a sample of invalid responses
    if invalid_responses:
        print("\nSample Invalid Responses (First 5):")
        print(invalid_responses[:5])  # Display first 5 invalid responses

    return {
        "overall_accuracy": overall_accuracy,
        "model_accuracies": model_accuracies
    }


# %%
def main(input_file):
    """Processes the outputs of gsm8k dataset, extracts necessary fields, cleans them, and evaluates accuracy."""

    print("\n[Step 1] Extracting necessary fields...")
    df_extracted = extract_fields_from_jsonl(input_file, ["indata.answer", "output.model", "output.response"])
    
    print("\n[Step 2] Cleaning extracted responses...")
    df_cleaned = clean_dataframe(df_extracted)
    
    print("\n[Step 3] Evaluating accuracy...")
    results = evaluate_accuracy(df_cleaned)

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


