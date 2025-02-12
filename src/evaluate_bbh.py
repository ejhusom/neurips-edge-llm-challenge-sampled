# %%
import json
import argparse
from collections import Counter, defaultdict
import re
import sys
import os
import pandas as pd
from pathlib import Path
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

    # Initial cleanup - remove leading/trailing whitespace
    response = response.strip()
    
    # Try to extract the response between <ANS> and </ANS>
    match = re.search(r"<ANS>(.*?)(?:</ANS>)?$", response, re.IGNORECASE | re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        # Handle nested empty tags
        if re.match(r"^<([A-Z0-9]+)>\s*</\1>$", extracted, re.IGNORECASE):
            tag_match = re.match(r"^<([A-Z0-9]+)>", extracted, re.IGNORECASE)
            if tag_match:
                return tag_match.group(1)
                
        # Handle parentheses cases
        paren_match = re.match(r"^\(([A-Z0-9])\)>?$", extracted, re.IGNORECASE)
        if paren_match:
            return paren_match.group(1)
    else:
        # First try to extract content from any HTML-like tags
        content_match = re.search(r"<[^>]+>([^<]+)</[^>]+>", response)
        if content_match:
            return content_match.group(1).strip()
            
        # If no content found, try the original pattern
        match = re.search(r"ANS\s*>?\s*(<([A-Z0-9/]+)>|\(?([A-Z0-9/]+)\)?)", response, re.IGNORECASE)
        extracted = match.group(2) or match.group(3) if match else response.strip()
    
    # Handle standalone parentheses case
    paren_match = re.match(r"^\(([A-Z0-9])\)$", extracted, re.IGNORECASE)
    if paren_match:
        return paren_match.group(1)
    
    # Remove HTML-like tags
    extracted = re.sub(r"<.*?>", "", extracted).strip()
    
    # Preserve multiple words/lines by replacing multiple whitespace with single space
    extracted = re.sub(r"\s+", " ", extracted).strip()
    
    # Handle invalid cases - but allow single letters/numbers
    if extracted.upper() in {"</ANS>", "<ANS>", "ANS >", ">", "", "ANS"}:
        return None
    
    return extracted


def clean_responses_further(response):
    """Clean the response by removing HTML tags, extra characters, and extracting meaningful text."""
    if response is None:
        return None
    
    # Remove HTML-like tags 
    response = re.sub(r"<.*?>", "", response).strip()
    
    # Handle cases where response is just ">", empty, or clearly invalid
    if response in {">", ""}:
        return None
    #response = re.sub(r"^\((.*?)\)$", r"\1", response)  # Remove surrounding parentheses

    # Remove surrounding parentheses if they enclose the whole response
    response = re.sub(r"^\((.*?)\)$", r"\1", response)
    
    return response if response else None  # If cleaned response is empty, return None


def extract_options(text):
    result = []
    text = text[ text.find("Options") + 8 : text.find("Print") ].replace("\n", "").strip()

    tokens = text.split("(")
    if "" in tokens:
        tokens.remove("")
    for token in tokens:
        result.append(f"({token}")

    return result

def find_prompt_type(prompt):
    prompt = prompt.lower()
    if 'options' in prompt: 
        return "option"
    else:
        return "other"

def process_prompt(prompt):
    """there are two prompt types :
            options (exact match of options or answers) 
            true/false, valid/invalid, Yes/No, computation/reasoning  (exact match, ignore case) 
    """
    cleaned_choices = []
    type = find_prompt_type(prompt)

    if type == 'option':
        cleaned_choices = extract_options(prompt)
        #print(f"cleaned_choices: {cleaned_choices}")
    else:
        cleaned_choices = None
    
    return type, cleaned_choices

def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove parentheses and option letters at the start (e.g., "(A) ", "A) ", "A. ")
    text = text.replace('(', '').replace(')', '')
    # Remove option letter if it's at the start (e.g., "a) ", "b. ")
    parts = text.split()
    if len(parts) > 1 and parts[0].rstrip(') .').isalpha() and len(parts[0]) <= 2:
        text = ' '.join(parts[1:])
    # Remove special characters and extra spaces
    text = ''.join(c if c.isalnum() else ' ' for c in text)
    # Replace multiple spaces with single space and strip
    text = ' '.join(text.split())
    return text

def match_response_to_label(cleaned_response, answerKey, choices, prompt, type='other', DEBUG=False):
    result = "None"
    options = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
   
    if cleaned_response is None:
        result = "None"  # discarded response
        #print(f"cleaned_response: {cleaned_response} answerKey: {answerKey}")
    else:
        cleaned_response = cleaned_response.lower() 
        answerKey = answerKey.lower()
        if cleaned_response == answerKey: # this handles type='other' (exact match, ignore case)
            result = "1"
            #print(f"cleaned_response: {cleaned_response} answerKey: {answerKey}")
        elif type == 'option' and "(" in answerKey and ")" in answerKey:  # if prompt type is options, answerKey should has the format of (A). Otherwise, the type is consdiered 'other'
            answerKey = clean_responses_further(answerKey) # if yes, remove () from the answerKey
            if cleaned_response is None:
                result = "None"  # Count discarded response
            elif cleaned_response == answerKey:
                result = "1"
            elif cleaned_response in options: # handle the case where response: c answerKey: b 
                result = "0"
            else: 
                # handle the case where the response does not provide option A, B, C, D
                for index, choice in enumerate(choices):
                    # Normalize both strings for comparison
                    normalized_response = normalize_text(cleaned_response)
                    normalized_choice = normalize_text(choice)
                    
                    if DEBUG:
                        print(f"Normalized response: {normalized_response}")
                        print(f"Normalized choice: {normalized_choice}")
                    
                    if normalized_response in normalized_choice or normalized_response == normalized_choice:
                        #print("in choices")
                        if DEBUG:
                            print("cleaned response matches a choice. Checking if it's correct...")
                            print(f"response: {cleaned_response} choice: {choice} answerKey: {answerKey}")
                            print(options[index], answerKey)
                        if options[index] == answerKey:
                            #print("equal")
                            result = "1"
                        else:
                            result = "0"
                            #print("not equal")
                        break
                    
        else:
            result = "0"
            if "sort" in prompt.lower(): # to handle sorting cases where the response contains "," and extra spaces but asnwerkey does not
                cleaned_response = cleaned_response.replace(",", "").replace(" ", "")
                answerKey = answerKey.replace(",", "").replace(" ", "")
                if cleaned_response == answerKey:
                    result = "1"
            # if the type is not 'option', the response must exact match. Otherwise, consider incorrect
    if DEBUG:
        print(f"cleaned_response: {cleaned_response} answerKey: {answerKey}")
        if result == "1":
            print("correct!")
        elif result == "0":
            print("incorrect!")
        else:
            print("invalid response!")      

    return result, cleaned_response, answerKey
 

# %%
def evaluate_accuracy(df):
    """Evaluate accuracy by counting correct responses."""
    DEBUG = False
    total = 0
    # correct = 0
    valid_responses = 0  # Track how many responses are valid
    removed_responses = 0  # Track how many responses were discarded as invalid
    model_correct_counts = defaultdict(int)
    model_total_counts = defaultdict(int)

    # invalid_responses = []  # To store invalid responses
    df['cleaned_response'] = ""
    df['matched_label'] = "0"
    df['answerKey'] = ""

    for index in range(len(df)):
        # if total >= 80 and total < 90:
        response = df['output.response'].iloc[index]
        answerKey = df['indata.target'].iloc[index]
        prompt = df['formatted_prompt'].iloc[index]
        model = df['output.model'].iloc[index]
        if DEBUG:
            print("\nPrompt: ", prompt)
            print("AnswerKey: ", answerKey)
            print("Response: ", response)
        type, choices = process_prompt(prompt)
        cleaned_response = clean_response(response)
        cleaned_response = clean_responses_further(cleaned_response)
        # save cleaned response into df
        # df.loc[index, "cleaned_response"] = cleaned_response
        
        matched_label, cleaned_response, answerKey = match_response_to_label(cleaned_response, answerKey, choices, prompt, type, DEBUG)
    
        # save stuff into df
        df.loc[index, "cleaned_response"] = cleaned_response
        df.loc[index, "matched_label"] = matched_label
        if type == 'option':
            df.loc[index, "answerKey"] =  answerKey
        
        if cleaned_response is None:
            removed_responses += 1  # If the cleaned response is None, discard it
            #invalid_responses.append((line.strip(), "Empty or invalid response"))  # Capture the raw line as invalid response
        # Only count valid responses that have a matched label
        elif matched_label != "None":
            valid_responses += 1
            model_total_counts[model] += 1  # Count valid responses
            if matched_label == "1":
                model_correct_counts[model] += 1  # Count correct responses
        else:
            removed_responses += 1  # Count discarded responses
            #invalid_responses.append((line.strip(), "No matching label"))  # Capture the raw line as invalid response

        total += 1  # Count every response line, whether valid or invalid
    
    # Compute accuracy per model
    #print(f"Accuracy evaluation in progress...")
    model_accuracies = {
        model: (model_correct_counts[model] / total) * 100 if total > 0 else 0
        for model in model_correct_counts.keys()
    }
    # Print results
    print(f"\nTotal responses: {total}")
    print(f"Valid responses: {valid_responses}")
    print(f"Removed responses: {removed_responses}")
    for model, accuracy in model_accuracies.items():
        print(f"Model: {model} | Correct: {model_correct_counts[model]} / {total} | Accuracy: {accuracy:.2f}%")

    score = round(model_correct_counts[model] / total * 100, 3)

    return model_accuracies
        



# %%
def main(input_file):
    print("\n[Step 1] Extracting necessary fields...")
    df_extracted = extract_fields_from_jsonl(input_file, ["indata.category", "formatted_prompt", "indata.target", "output.model", "output.response"])
    
    print("\n[Step 2] Processing results and evaluating accuracy...")
    model_accuracies = evaluate_accuracy(df_extracted)
    
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
    print(f"\nModel performance breakdown: {model_accuracies}")


