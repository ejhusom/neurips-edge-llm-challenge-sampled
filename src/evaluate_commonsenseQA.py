# %%
import json
import argparse
from collections import Counter, defaultdict
import re
import sys
import os

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
                    "answerKey": data.get("indata", {}).get("answerKey"),
                    "id": data.get("indata", {}).get("id"),
                    "choices": data.get("indata", {}).get("question", {}).get("choices", []),
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
    """Extract the actual response from various ANS formats and handle invalid cases."""
    cleaned_responses = []
    
    for response in responses:
        # Try to extract the response between <ANS> and </ANS>
        match = re.search(r"<ANS>(.*?)</ANS>", response, re.IGNORECASE)
        if not match:
            # Handle cases like "ANS > D", "ANS >C", "ANS > ", "ANS<EM>dauntless</EM>"
            match = re.search(r"ANS\s*>?\s*(\S.*)", response, re.IGNORECASE)

        # Extracted response if found, otherwise set empty string for invalid responses
        extracted = match.group(1).strip() if match else response.strip()

        # Handle invalid cases like "</ANS>", "<ANS>", etc.
        if extracted in {"</ANS>", "<ANS>", "ANS >", ""}:
            extracted = ""

        cleaned_responses.append(extracted)

    return cleaned_responses


# %%
def postprocess_extracted_jsonl_file(input_file):
    """Read, process, and write cleaned JSONL responses."""

    # Generate the output file name by appending "_cleared" to the input file name
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_cleared{ext}"

    cleaned_data = []
    
    # Read the JSONL file
    with open(input_file, 'r') as infile:
        for line in infile:
            data = json.loads(line.strip())

            # Ensure 'response' exists, and process it using postprocess_responses
            if 'response' in data:
                data['response'] = postprocess_responses([data['response']])[0]  # Process single response
            
            cleaned_data.append(data)
    
    # Write the cleaned data back to a new JSONL file
    with open(output_file, 'w') as outfile:
        for entry in cleaned_data:
            outfile.write(json.dumps(entry) + '\n')
            
    print(f"Cleaned responses have been written to {output_file}.")
    return output_file

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
def evaluate_accuracy(input_file):
    """Evaluate accuracy by counting correct responses."""
    total = 0
    correct = 0
    valid_responses = 0  # Track how many responses are valid
    removed_responses = 0  # Track how many responses were discarded as invalid

    model_correct_counts = defaultdict(int)
    model_total_counts = defaultdict(int)

    invalid_responses = []  # To store invalid responses

    with open(input_file, 'r') as infile:
        for line in infile:
            data = json.loads(line.strip())

            # Extract required fields
            answer_key = data.get("answerKey")
            choices = data.get("choices", [])
            raw_response = data.get("response", "")
            model = data.get("model", "unknown_model")  # Model name

            # Clean and match response
            cleaned_response = clean_responses_further(raw_response)

            # If the response is valid (not None), process it
            if cleaned_response is not None:
                matched_label = match_response_to_label(cleaned_response, choices)
                
                # Only count valid responses that have a matched label
                if matched_label is not None:
                    valid_responses += 1
                    model_total_counts[model] += 1  # Count valid responses
                    if matched_label == answer_key:
                        model_correct_counts[model] += 1  # Count correct responses
                else:
                    removed_responses += 1  # Count discarded responses
                    invalid_responses.append((line.strip(), "No matching label"))  # Capture the raw line as invalid response

            else:
                removed_responses += 1  # If the cleaned response is None, discard it
                invalid_responses.append((line.strip(), "Empty or invalid response"))  # Capture the raw line as invalid response

            total += 1  # Count every response line, whether valid or invalid
        
   
    # Compute accuracy per model
    print(f"Accuracy evaluation in progress...")
    model_accuracies = {
        model: (model_correct_counts[model] / total) * 100 if total > 0 else 0
        for model in model_correct_counts.keys()
    }
    # Print results
    print(f"Total responses: {total}")
    print(f"Valid responses: {valid_responses}")
    print(f"Removed responses: {removed_responses}")
    for model, accuracy in model_accuracies.items():
        print(f"Model: {model} | Correct: {model_correct_counts[model]} / {total} | Accuracy: {accuracy:.2f}%")

    # Print invalid responses
    if invalid_responses:
        print("\nInvalid responses detected:")
        for response, reason in invalid_responses:
            print(f"Invalid Response: {response} | Reason: {reason}")
              
    return model_accuracies


# %%
def main(input_file):
    # Step 1: Extract and write required fields to calculate accuracy
    extracted_file = extract_and_write_fields(input_file)
    
    # Step 2: Postprocess the extracted JSONL file, clear responses from <ANS> tags
    cleaned_file = postprocess_extracted_jsonl_file(extracted_file)
    
    # Step 3: Evaluate accuracy
    model_accuracies = evaluate_accuracy(cleaned_file)

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


