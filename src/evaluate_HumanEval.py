# %%
import json
from evaluate import load
import argparse
import os
import pandas as pd
from typing import List
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

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
def clean_response(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Clean the specified column in the given DataFrame and add a new column 'cleaned_{column_name}'.
    
    :param df: DataFrame containing the specified column.
    :param column_name: The name of the column to be cleaned.
    :return: DataFrame with an added 'cleaned_{column_name}' column.
    """
    if column_name in df.columns:
        cleaned_column_name = f"cleaned_{column_name}"
        df[cleaned_column_name] = df[column_name].astype(str).str.replace("<ANS>", "").str.replace("</ANS>", "")
    else:
        print(f"Column '{column_name}' not found in DataFrame.")
    
    return df

# %%
def evaluate_pass_at_k(df: pd.DataFrame, k=[1]) -> dict:
    """
    Evaluate the pass@k metric over a cleaned DataFrame.
    
    :param df: DataFrame containing test cases, model information, and cleaned responses.
    :param k: List of k values for pass@k evaluation.
    :return: Dictionary with pass@k scores per model.
    """
    # Load the evaluation function
    code_eval = load("code_eval")
    
    if not all(col in df.columns for col in ["indata.test", "output.model", "cleaned_output.response"]):
        print("Missing required columns for evaluation.")
        return {}
    
    # Respective lists for the test, cleaned response, and model
    test_cases = df["indata.test"].tolist()
    candidates = df["cleaned_output.response"].apply(lambda x: [x] if pd.notna(x) else []).tolist()
    model_list = df["output.model"].tolist()
    
    # Compute pass@k for each model
    pass_at_k, results = code_eval.compute(references=test_cases, predictions=candidates, k=k)
    
    # Track models and their performances
    model_performance = {model: pass_at_k for model in set(model_list)}
    return model_performance

# %%
def main(input_file: str, k_values=[1]):
    """
    Main function to extract, clean, and evaluate pass@k.
    
    :param input_file: Path to the input JSONL file.
    :param k_values: List of k values for pass@k evaluation.
    """
    print("\n[Step 1] Extracting necessary fields...")
    df_extracted = extract_fields_from_jsonl(input_file, ["indata.task_id", "indata.test", "output.model", "output.response"])
    
    print("\n[Step 2] Cleaning responses...")
    df_cleaned = clean_response(df_extracted, "output.response")
    
    print("\n[Step 3] Evaluating accuracy...")
    performance = evaluate_pass_at_k(df_cleaned, k=k_values)
    
    print(f"\nModel performance: {performance}")
    return performance

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


