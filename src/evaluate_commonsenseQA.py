# %%
import json
import argparse
from collections import Counter, defaultdict
import re
import sys
import os
import pandas as pd
from typing import List

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
def clean_response(response):
        """Extract the actual response from various ANS formats and handle invalid cases."""
        if not isinstance(response, str):  # Handle cases where response isn't a string
            return ""

        # Try to extract the response between <ANS> and </ANS>
        match = re.search(r"<ANS>(.*?)</ANS>", response, re.IGNORECASE)
        if not match:
            # Handle cases like "ANS > D", "ANS >C", "ANS > ", "ANS<EM>dauntless</EM>"
            match = re.search(r"ANS\s*>?\s*(\S.*)", response, re.IGNORECASE)

        # Extracted response if found, otherwise set empty string for invalid responses
        extracted = match.group(1).strip() if match else response.strip()

        # Handle invalid cases like "</ANS>", "<ANS>", "ANS >", "" by setting them to empty string
        if extracted in {"</ANS>", "<ANS>", "ANS >", ">", ""}:
            extracted = ""

        return extracted

# %%
def clean_responses_further(response):
    """Clean the response by removing HTML tags, extra characters, and extracting meaningful text."""
    # Remove HTML-like tags 
    response = re.sub(r"<.*?>", "", response).strip()
    
    # Handle cases where response is just ">", empty, or clearly invalid
    if response in {">", ""}:
        return None

    # Extract letter if response is in format "(X)" or "X."
    match = re.match(r"\(?([A-E])\)?\.?", response)
    if match:
        return match.group(1)  # Extracted letter label (A-E)
    
    # Remove extra characters or trailing punctuation that might be in the raw response
    response = re.sub(r"[^\w\s]", "", response).strip()

    return response if response else None  # If cleaned response is empty, return None


# %%
def postprocess_responses(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Clean the specified column in the given DataFrame and add a new column 'cleaned_{column_name}'.
    
    :param df: DataFrame containing the specified column.
    :param column_name: The name of the column to be cleaned.
    :return: DataFrame with an added 'cleaned_{column_name}' column.
    """
    # Apply cleaning function to the specified column and store in "cleaned_{column_name}"
    if column_name in df.columns:
        cleaned_column_name = f"cleaned_{column_name}"
        df[cleaned_column_name] = df[column_name].apply(clean_response)
        df[cleaned_column_name] = df[cleaned_column_name].apply(clean_responses_further)
    else:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    return df


# %%
def match_response_to_label(response, choices):
    """Match a response (either raw text or label) to the corresponding choice label (A-E)."""
    # If the response is None, return None immediately
    if response is None:
        return None
    
    # If the response is already a valid label (A-E), return it
    valid_labels = {choice["label"] for choice in choices}
    if response in valid_labels:
        return response
    
    # Try to match the response with the text values in choices
    for choice in choices:
        if response.lower() == choice["text"].lower(): 
            return choice["label"]  

    return None  # Return None if no match is found


# %%
def evaluate_accuracy(df: pd.DataFrame) -> dict:
    """Evaluate accuracy by counting correct responses from a DataFrame."""
    total = len(df)
    correct = 0
    valid_responses = 0  # Count of valid responses
    removed_responses = 0  # Count of invalid responses

    model_correct_counts = defaultdict(int)
    model_total_counts = defaultdict(int)

    invalid_responses = []  # Store invalid responses for debugging

    for _, row in df.iterrows():
        answer_key = row.get("indata.answerKey")
        choices = row.get("indata.question.choices", [])
        cleaned_response = row.get("cleaned_output.response", "")
        model = row.get("output.model", "unknown_model")

        # Match cleaned response to choice labels
        if cleaned_response:
            matched_label = match_response_to_label(cleaned_response, choices)

            if matched_label is not None:
                valid_responses += 1
                model_total_counts[model] += 1  # Count total number of responses for the model
                
                if matched_label == answer_key:
                    model_correct_counts[model] += 1  # Count correct responses
            else:
                removed_responses += 1  # No valid label found
                model_total_counts[model] += 1
                invalid_responses.append((cleaned_response, "No matching label"))
        else:
            removed_responses += 1  # Empty response
            model_total_counts[model] += 1
            invalid_responses.append((cleaned_response, "Empty response"))

    # Compute accuracy per model
    model_accuracies = {
        model: (model_correct_counts[model] / model_total_counts[model]) * 100 if model_total_counts[model] > 0 else 0
        for model in model_total_counts.keys()
    }

    # Print summary
    print(f"Total responses: {total}")
    print(f"Valid responses: {valid_responses}")
    print(f"Removed responses: {removed_responses}")
    for model, accuracy in model_accuracies.items():
        print(f"Model: {model} | Correct: {model_correct_counts[model]} / {model_total_counts[model]} | Accuracy: {accuracy:.2f}%")

    # Print invalid responses
    if invalid_responses:
        print("\nInvalid responses detected:")
        for response, reason in invalid_responses[:10]:  # Show first 10 invalid cases
            print(f"Invalid Response: {response} | Reason: {reason}")

    return model_accuracies


# %%
def main(input_file):
    print("\n[Step 1] Extracting necessary fields...")
    df_extracted = extract_fields_from_jsonl(input_file, ["indata.answerKey", "indata.id", "indata.question.choices", "output.model", "output.response"])
    
    print("\n[Step 2] Postprocessing responses...")
    df_cleaned = postprocess_responses(df_extracted, "output.response")
    
    print("\n[Step 3] Evaluating accuracy...")
    model_accuracies = evaluate_accuracy(df_cleaned)
    
    #print(f"\nModel performance: {model_accuracies}")
    return model_accuracies
    
    

# %%
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate accuracy for a given input JSONL file.")
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file.")
    
    # Parse arguments
    args = parser.parse_args()

    # Check if the file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: The file {args.input_file} does not exist.")
        sys.exit(1)

    # Call the main function with the parsed argument
    model_accuracies = main(args.input_file)
    
    # Print the results
    print(f"Model performance breakdown: {model_accuracies}")


