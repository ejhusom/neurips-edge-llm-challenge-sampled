# %%
import argparse
import os
import sys
import re
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from difflib import SequenceMatcher
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from Levenshtein import distance as levenshtein_distance


# %%
def extract_fields_from_jsonl(file_path: str) -> pd.DataFrame:
    """
    Extract specified fields from a JSONL file (for TruthfulQA) and return them as a DataFrame.
    
    :param file_path: Path to the JSONL file.
    :return: DataFrame with extracted columns if available.
    """
    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))  # Load each JSON object
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line[:100]}...\nError: {e}")

    if not data:
        print("No valid data found in the file.")

    # Normalize JSON data
    rows = []
    for entry in data:
        question = entry["indata"]["question"]
        formatted_prompt = entry["formatted_prompt"]
        model_response = entry["output"]["response"]
        model = entry["output"]["model"]

        # Extract choice texts as lists
        mc1_choices = list(entry["indata"]["mc1_targets"].keys())
        mc2_choices = list(entry["indata"]["mc2_targets"].keys())
        #print(mc1_choices)

        # Identify all correct answers (where value == 1)
        correct_mc1 = [choice for choice, label in entry["indata"]["mc1_targets"].items() if label == 1]
        correct_mc2 = [choice for choice, label in entry["indata"]["mc2_targets"].items() if label == 1]

        rows.append({
            "indata.question": question,
            #"formatted_prompt": formatted_prompt,
            "mc1_targets": mc1_choices,  # List of all choices
            "correct_mc1": correct_mc1,  # List of correct answers
            "mc2_targets": mc2_choices,  # List of all choices
            "correct_mc2": correct_mc2,  # List of correct answers
            "output.model": model,
            "output.response": model_response
        })

    # Create DataFrame from extracted fields
    df_ectracted = pd.DataFrame(rows)
    return df_ectracted


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


# %%
def evaluate_accuracy_old(df):
    """
    Evaluate accuracy based on the best match from evaluate_response function.

    Args:
        df (pd.DataFrame): DataFrame containing response data, and choices.

    Returns:
        dict: Accuracy results per model, including:
              - correct responses
              - invalid responses
              - wrong responses
              - total responses
              - accuracy score
    """
    model_results = defaultdict(lambda: {"correct": 0, "invalid": 0, "wrong": 0, "total": 0})

    for _, row in df.iterrows():
        model = row["output.model"]
        response = row["output.response"]
        mc1_targets = row["indata.mc1_targets"]

        # Extract choices and correct answer
        choices = list(mc1_targets.keys())
        correct_answer = next(choice for choice, label in mc1_targets.items() if label == 1)

        # Clean response
        cleaned_response = extract_answer(response)

        # Check for invalid response
        if is_invalid_response(cleaned_response, choices):
            cleaned_response = "INVALID"

        # Get the best match
        best_match_result = evaluate_response(cleaned_response, choices)

        # Extract best match
        best_match_data = next(iter(best_match_result.values()))
        best_match = best_match_data.get("best_match", None) if isinstance(best_match_data, dict) else None

        # Normalize strings for comparison
        correct_answer = str(correct_answer).strip()
        if best_match:
            best_match = str(best_match).strip()

        # Update counts based on match result
        if best_match == "INVALID":
            model_results[model]["invalid"] += 1
        elif best_match == correct_answer:
            model_results[model]["correct"] += 1
        else:
            model_results[model]["wrong"] += 1

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
def evaluate_accuracy(df):
    """
    Evaluate accuracy based on the best match from evaluate_response function.

    Args:
        df (pd.DataFrame): DataFrame containing response data.

    Returns:
        dict: Accuracy results per model, including:
              - correct responses
              - invalid responses
              - wrong responses
              - total responses
              - accuracy score
    """
    model_results = defaultdict(lambda: {"correct": 0, "invalid": 0, "wrong": 0, "total": 0})

    evaluation = []

    for _, row in df.iterrows():
        # Use this variable to check whether response was correct or not at the end of for loop
        correct_response = False

        model = row["output.model"]
        response = row["output.response"]
        mc1_targets = row["mc1_targets"]  # List of all choices (text only)
        correct_mc1 = row["correct_mc1"]  # List of correct answers

        # Clean response
        cleaned_response = extract_answer(response)

        # Check for invalid response
        if is_invalid_response(cleaned_response, mc1_targets):
            cleaned_response = "INVALID"

        # Get the best match
        best_match_result = evaluate_response(cleaned_response, mc1_targets)

        # Extract best match
        best_match_data = next(iter(best_match_result.values()), {})  # Get first value (dict) safely
        best_match = best_match_data.get("best_match", None)

        # Normalize for comparison
        best_match = str(best_match).strip() if best_match else None
        correct_mc1 = {str(ans).strip() for ans in correct_mc1} 

        # Update counts
        if best_match == "INVALID":
            model_results[model]["invalid"] += 1
        elif best_match in correct_mc1:  # Check if best_match is in correct answers list
            model_results[model]["correct"] += 1
            correct_response = True
        else:
            model_results[model]["wrong"] += 1

        # Increment total responses
        model_results[model]["total"] += 1

        if correct_response:
            evaluation.append(1)
        else:
            evaluation.append(0)
    
    df["evaluation"] = evaluation

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

    return accuracy_results, df


# %%
def main(input_file):
    """Processes the outputs of gsm8k dataset, extracts necessary fields, cleans them, and evaluates accuracy."""

    print("\n[Step 1] Extracting necessary fields...")
    df_extracted = extract_fields_from_jsonl(input_file)
    
    print("\n[Step 2] Evaluating accuracy...")
    results, df = evaluate_accuracy(df_extracted)

    print("\n[Process Complete] Accuracy Evaluation Done.\n")
    
    return results, df


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
    results, df = main(args.input_file)

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


