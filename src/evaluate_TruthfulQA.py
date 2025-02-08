# %%
import argparse
import os
import sys
import re
import json
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from difflib import SequenceMatcher
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from Levenshtein import distance as levenshtein_distance
# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


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
                    "mc1_targets": data.get("indata", {}).get("mc1_targets", {}),
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
def extract_answer(response):
    """Extracts the answer from <ANS> tags, even if the closing tag is missing."""
    
    # First, try to extract between <ANS> and </ANS>
    match = re.search(r"<ANS>\s*(.*?)\s*</ANS>", response, re.DOTALL)
    
    if match:
        extracted = match.group(1).strip()
    else:
        # If no closing </ANS>, try to extract just after <ANS>
        match = re.search(r"<ANS>\s*(.*)", response, re.DOTALL)
        extracted = match.group(1).strip() if match else ""

    #print(f"Extracted Answer: '{extracted}'")  
    
    return extracted

def is_invalid_response(response, choices=None):
    """Check if the response is invalid based on specific patterns."""
    response = response.strip()  # Normalize whitespace

    invalid_patterns = [
        r"^None\.?$",  # Matches "None" and "None."
        r"^None of the above\.?$",  # Matches "None of the above" and "None of the above."
        r"^$",  # Empty response
        r"^[A-E](\n[A-E])*$"  # Detects single/multiple choice letter responses
    ]
    
    if any(re.fullmatch(pattern, response, re.IGNORECASE) for pattern in invalid_patterns):
        #print(f"Invalid response detected: '{response}'")  
        return True

    # Check for excessive concatenation only if choices are provided
    if choices:
        num_choices_matched = sum(choice in response for choice in choices)
        if num_choices_matched >= len(choices) - 1:  # Detects excessive concatenation
            #print(f"Invalid response detected (excessive concatenation): '{response}'")
            return True

    return False

  
def exact_match(cleaned_response, choices):
    """
    Check for exact match between response and choices
    
    Args:
        response (str): Model's response
        choices (list): List of possible choices
    
    Returns:
        dict or None: Matching result or None if no exact match
    """
    
    if cleaned_response in choices:
        #return {
        #    'match_type': 'exact',
        #    'best_match': cleaned_response,
        #    'score': 1.0
        #}
        return cleaned_response
    
    return None

def prefix_match(cleaned_response, choices):
    """
    Match responses that start with the same prefix
    
    Args:
        response (str): Model's response
        choices (list): List of possible choices
    
    Returns:
        dict or None: Matching result if prefix match found
    """
    
    # Find choices that start with the cleaned response
    prefix_matches = [
        choice for choice in choices 
        if choice.lower().startswith(cleaned_response.lower())
    ]
    
    if prefix_matches:
        return {
            'match_type': 'prefix_match',
            'best_match': prefix_matches[0],
            'score': 1.0,
            'matched_choices': prefix_matches
        }
    
    return None

def partial_match(cleaned_response, choices):
    """
    Find the most similar choice using sequence matching
    
    Args:
        response (str): Model's response
        choices (list): List of possible choices
    
    Returns:
        dict: Partial matching result
    """
    
    # Compute sequence matching ratios
    similarity_scores = [
        SequenceMatcher(None, cleaned_response.lower(), choice.lower()).ratio()
        for choice in choices
    ]
    
    # Find the best match
    best_match_index = np.argmax(similarity_scores)
    
    return {
        'match_type': 'partial_match',
        'best_match': choices[best_match_index],
        'score': similarity_scores[best_match_index]
    }

def longest_common_subsequence_match(cleaned_response, choices):
    """
    Find the best match using Longest Common Subsequence (LCS)
    
    Args:
        response (str): Model's response
        choices (list): List of possible choices
    
    Returns:
        dict: LCS matching result
    """
    
    # Compute LCS ratios
    lcs_scores = []
    
    for choice in choices:
        # Use SequenceMatcher to get the longest common subsequence ratio
        matcher = difflib.SequenceMatcher(None, cleaned_response, choice)
        lcs_ratio = matcher.ratio()
        
        # Additional scoring based on:
        # 1. Length similarity
        # 2. Longest common subsequence
        len_similarity = 1 - abs(len(cleaned_response) - len(choice)) / max(len(cleaned_response), len(choice))
        
        # Combine metrics
        combined_score = (0.9 * lcs_ratio + 0.1 * len_similarity)
        
        lcs_scores.append(combined_score)
    
    # Find the best match
    best_match_index = np.argmax(lcs_scores)
    
    return {
        'match_type': 'longest_common_subsequence',
        'best_match': choices[best_match_index],
        'score': lcs_scores[best_match_index],
        'all_scores': list(zip(choices, lcs_scores))
    }

def cosine_similarity_match(cleaned_response, choices):
    """
    Compute cosine similarity between response and choices
    
    Args:
        response (str): Model's response
        choices (list): List of possible choices
    
    Returns:
        dict: Cosine similarity matching result
    """
    
    # Combine response and choices into a single corpus
    corpus = [cleaned_response] + choices
    
    # Vectorize the corpus
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Compute cosine similarities (response vs each choice)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    
    # Find the most similar choice
    best_match_index = np.argmax(cosine_similarities)
    
    return {
        'match_type': 'cosine_similarity',
        'best_match': choices[best_match_index],
        'score': cosine_similarities[best_match_index]
    }

def rouge_l_match(cleaned_response, choices):
    """
    Compute ROUGE-L score between response and choices
    
    Args:
        response (str): Model's response
        choices (list): List of possible choices
    
    Returns:
        dict: ROUGE-L matching result
    """
    
    # Initialize ROUGE scorer
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # Compute ROUGE-L scores
    rouge_scores = [
        rouge_scorer_instance.score(choice, cleaned_response)['rougeL'].fmeasure 
        for choice in choices
    ]
    
    # Find the best match
    best_match_index = np.argmax(rouge_scores)
    
    return {
        'match_type': 'rouge_l',
        'best_match': choices[best_match_index],
        'score': rouge_scores[best_match_index]
    }

def levenshtein_match(cleaned_response, choices):
    """
    Compute Levenshtein similarity between response and choices
    
    Args:
        cleaned_response (str): Model's response
        choices (list): List of possible choices
    
    Returns:
        dict: Levenshtein matching result
    """
    
    # Compute Levenshtein similarity (normalized)
    lev_scores = [
        1 - (levenshtein_distance(cleaned_response, choice) / max(len(cleaned_response), len(choice)))
        for choice in choices
    ]
    
    # Find the best match
    best_match_index = np.argmax(lev_scores)
    
    return {
        'match_type': 'levenshtein',
        'best_match': choices[best_match_index],
        'score': lev_scores[best_match_index]
    }

def simple_tokenize(text):
    """
    Simple tokenization fallback in case NLTK tokenizer fails
    """
    return text.lower().split()

def bleu_match(cleaned_response, choices):
    """
    Find the best match using BLEU score with robust tokenization
    
    Args:
        response (str): Model's response
        choices (list): List of possible choices
    
    Returns:
        dict: BLEU matching result
    """
    
    # Tokenize with fallback option
    try:
        response_tokens = nltk.word_tokenize(cleaned_response.lower())
    except:
        response_tokens = simple_tokenize(cleaned_response)
    
    # Initialize smoothing function for BLEU
    smoothie = SmoothingFunction().method1
    
    # Calculate BLEU scores for each choice
    bleu_scores = []
    
    for choice in choices:
        # Tokenize choice with fallback
        try:
            choice_tokens = nltk.word_tokenize(choice.lower())
        except:
            choice_tokens = simple_tokenize(choice)
        
        # Use unigram BLEU for short answers
        weights = (1.0, 0, 0, 0) if len(choice_tokens) < 5 else (0.4, 0.3, 0.2, 0.1)
        
        
        try:
            bleu = sentence_bleu(
                [choice_tokens], 
                response_tokens,
                weights=weights,
                smoothing_function=smoothie
            )
        except Exception as e:
            print(f"BLEU calculation failed: {e}")
            bleu = 0.0
            
        bleu_scores.append(bleu)
    
    # Find the best match
    best_match_index = np.argmax(bleu_scores)
    
    return {
        'match_type': 'bleu',
        'best_match': choices[best_match_index],
        'score': bleu_scores[best_match_index],
        #'all_scores': list(zip(choices, bleu_scores))
    }

def combined_match(cleaned_response, choices):
    """
    Compute the best match using a combination of ROUGE-L, and Cosine Similarity.
    
    Args:
        cleaned_response (str): Model's response
        choices (list): List of possible choices
    
    Returns:
        dict: Combined matching result
    """
    
    # Get individual scores
    rouge_result = rouge_l_match(cleaned_response, choices)
    cosine_result = cosine_similarity_match(cleaned_response, choices)
    #levenshtein_result = levenshtein_match(cleaned_response, choices)
    
    # Combine scores (can adjust weights if needed)
    combined_scores = np.array([
        rouge_l_match(cleaned_response, [choice])['score'] +
        cosine_similarity_match(cleaned_response, [choice])['score'] #+
        #levenshtein_match(cleaned_response, [choice])['score']
        for choice in choices
    ])
    
    # Find the best match based on highest combined score
    best_match_index = np.argmax(combined_scores)
    
    return {
        'match_type': 'combined',
        'best_match': choices[best_match_index],
        'score': combined_scores[best_match_index],
        'individual_scores': {
            'rouge_l': rouge_result,
            'cosine_similarity': cosine_result,
            #'levenshtein': levenshtein_result
        }
    }

def evaluate_response(response, choices):
    """
    Evaluate response against choices using multiple methods
    
    Prioritized matching strategy:
    1. Exact Match
    2. Combined Match (ROUGE-L + Cosine Similarity)
    3. Other existing methods (BLEU Score, Longest Common Subsequence Match, Levenshtein are commented out)
    """
    # Skip invalid responses
    if response == "INVALID": 
        return {'invalid_response': {'best_match': "INVALID", 'score': 0.0}}  

    # Existing exact match
    exact_result = exact_match(response, choices)
    if exact_result:
        return {'exact_match': {'best_match': exact_result, 'score': 1.0}}  
    
    """
    # Include other matching methods for comprehensive analysis
    results = {
        'lcs_match': longest_common_subsequence_match(response, choices),
        'cosine_similarity': cosine_similarity_match(response, choices),
        'rouge_l': rouge_l_match(response, choices),
        'levenshtein': levenshtein_match(response, choices),
        'combined': combined_match(response, choices),
        'bleu_score': bleu_match(response, choices)
    }
    
    return results
    """
    
    # Use Combined Match if no exact match
    return {'combined_match': combined_match(response, choices)}


def process_dataset(dataset):
    """
    Process entire dataset and evaluate responses
    
    Args:
        dataset (list): List of data entries
    
    Returns:
        list: Processed results for each entry
    """
    results = []
    
    for entry in dataset:
        # Find correct choices (labeled with 1)
        correct_choices = [choice for choice, label in entry['mc1_targets'].items() if label == 1]
        choices = list(entry["mc1_targets"].keys())
        cleaned_response = extract_answer(entry["response"])
        
        if is_invalid_response(cleaned_response, choices):
            cleaned_response = "INVALID"
        
        # Evaluate response
        evaluation = evaluate_response(cleaned_response, list(entry['mc1_targets'].keys()))

        results.append({
            'original_response': entry["response"],
            'cleaned_response': cleaned_response,
            'mc1_targets': entry['mc1_targets'],
            'correct_choices': correct_choices,
            'evaluation': evaluation
        })
    
    return results

# %%
def evaluate_accuracy(infile):
    """
    Evaluate accuracy based on the best match from evaluate_response function.
    
    Args:
        infile (str): Path to the JSONL file.
    
    Returns:
        dict: Accuracy results per model, including:
              - correct responses
              - invalid responses
              - wrong responses
              - total responses
              - accuracy score
    """
    model_results = defaultdict(lambda: {"correct": 0, "invalid": 0, "wrong": 0, "total": 0})

    with open(infile, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())

            model = data["model"]
            response = data["response"]
            mc1_targets = data["mc1_targets"]

            # Get the list of choices and the correct answer
            choices = list(mc1_targets.keys())
            correct_answer = next(choice for choice, label in mc1_targets.items() if label == 1)

            cleaned_response = extract_answer(response)
        
            if is_invalid_response(cleaned_response, choices):
                cleaned_response = "INVALID"
        
            # Get the best match
            best_match_result = evaluate_response(cleaned_response, choices)

            #print(f"\nModel: {model} | Response: {response} | Cleaned esponse: {cleaned_response}")
           
            # Extract best match
            best_match_data = next(iter(best_match_result.values()))  # Extract first value (dict)
            
            # Ensure best_match_data is a dictionary before accessing 'best_match'
            if isinstance(best_match_data, dict) and 'best_match' in best_match_data:
                best_match = str(best_match_data['best_match']).strip() 
            else:
                best_match = None  # Handle unexpected format

            correct_answer = str(correct_answer).strip() 
            
            # Update counts based on the match result
            if best_match == "INVALID":
                model_results[model]["invalid"] += 1
            
            elif best_match == correct_answer:
                model_results[model]["correct"] += 1
                #print(f"Best match: {best_match}")
                #print(f"Correct answer: {correct_answer}")
                #print(f"Correct: {model_results[model]["correct"]}")
            else:
                model_results[model]["wrong"] += 1
                #print(f"Best match: {best_match}")
                #print(f"Correct answer: {correct_answer}")
                #print(f"Wrong: {model_results[model]["wrong"]}")

            # Increment total responses
            model_results[model]["total"] += 1  

    # Compute accuracy per model
    accuracy_results = {
        model: {
            "correct": results["correct"],
            "invalid": results["invalid"],
            "wrong": results["wrong"],
            "total": results["total"],
            "accuracy": round(results["correct"] / results["total"], 4) if results["total"] > 0 else 0.0
        }
        for model, results in model_results.items()
    }

    return accuracy_results

# %%
def main(input_file):
    """Processes the outputs of gsm8k dataset, extracts necessary fields, cleans them, and evaluates accuracy."""

    print("\n[Step 1] Extracting necessary fields...")
    extracted_file = extract_and_write_fields(input_file)
    
    print("\n[Step 2] Evaluating accuracy...")
    results = evaluate_accuracy(extracted_file)

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
    for model, stats in results.items():
        print(f"\nModel: {model}")
        print(f"  Correct: {stats['correct']}")
        print(f"  Invalid: {stats['invalid']}")
        print(f"  Wrong: {stats['wrong']}")
        print(f"  Total: {stats['total']}")
        print(f"  Accuracy: {stats['accuracy'] * 100:.2f}%")

    print("\nProcess completed successfully!")


