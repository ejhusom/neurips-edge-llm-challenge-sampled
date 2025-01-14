import argparse
import jsonlines
import re

def eval_metric_bigbenchhard(true, prediction):
    """Evaluate the prediction based on the true answer for the BIG-Bench-Hard dataset.

    Source: https://microsoft.github.io/Trace/examples/nlp/bigbench_hard.html

    Args:
        true (str): The true answer.
        prediction (str): The predicted answer.

    Returns:
        bool: Whether the prediction is correct.

    """
    # Check if the true answer contains a single letter in parentheses
    matches = re.findall(r"\([A-Z]\)", true)
    if matches:
        # Parse the prediction to match the format of the true answer
        matches = re.findall(r"\([A-Z]\)", prediction)
        parsed_answer = matches[-1] if matches else ""
        return parsed_answer == true
    else:
        return prediction == true

def main():
    pass
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate performance of a model on a benchmark dataset.")
    parser.add_argument('--input', type=str, help="Path to the input JSONL file.")
    parser.add_argument('--output', type=str, help="Path to the output JSONL file.")
    parser.add_argument('--dataset_type', type=str, help="Dataset type (e.g., BIG-Bench-Hard, CommonSenseQA, GSM8K, HumanEval, TruthfulQA).")

    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    dataset_type = args.dataset_type

    if dataset_type == "BIG-Bench-Hard":

        # Calculate the accuracy of the model on the BIG-Bench-Hard dataset
        n_correct = 0

        with jsonlines.open(input_file, mode='r') as response_reader:
            with jsonlines.open(output_file, mode='a') as writer:
                for response in response_reader:
                    true_answer = response['indata']['target']
                    try:
                        predicted_answer = response['output']['response']
                    except KeyError:
                        predicted_answer = response['output']['message']['content']
                    is_correct = eval_metric_bigbenchhard(true_answer, predicted_answer)
                    response['is_correct'] = is_correct
                    writer.write(response)
                    if is_correct:
                        n_correct += 1
        
        accuracy = n_correct / len(response_reader)
        print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()