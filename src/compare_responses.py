import os
import json
import csv

def extract_model_and_dataset(filename):
    parts = filename.split('_')
    model = parts[1]
    dataset = parts[2]
    return model, dataset

def read_jsonl_content(filepath):
    with open(filepath, 'r') as file:
        data = [json.loads(line) for line in file]
        formatted_prompt = data[0]['formatted_prompt']
        output_content = [entry['output']['message']['content'].strip() for entry in data]
    return formatted_prompt, output_content

def compare_responses(directories, output_csv, exclude_without=True, include_prompt=True):
    files = os.listdir(directories[0])

    if exclude_without:
        files = [f for f in files if 'without' not in f and f.endswith('.jsonl')]
    else:
        files = [f for f in files if f.endswith('.jsonl')]

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if include_prompt:
            writer.writerow(['model', 'dataset', 'formatted_prompt', 'experiment_1', 'experiment_2'])
        else:
            writer.writerow(['model', 'dataset', 'without_instruct', 'experiment_1', 'experiment_2'])

        for filename in files:
            model, dataset = extract_model_and_dataset(filename)
            formatted_prompt = None
            contents = []
            
            for directory in directories:
                filepath = os.path.join(directory, filename)
                prompt, output_content = read_jsonl_content(filepath)
                if formatted_prompt is None:
                    formatted_prompt = prompt
                contents.append(output_content)

            for lines in zip(*contents):
                if include_prompt:
                    writer.writerow([model, dataset, formatted_prompt] + list(lines))
                else:
                    writer.writerow([model, dataset] + list(lines))

if __name__ == "__main__":
    directories = [
        "experiment-instructions-without",
        "experiment-instructions-v1",
        #"experiment-instructions-v2",
        "experiment-instructions-v3"
    ]
    output_csv = "comparison_results.csv"
    compare_responses(directories, output_csv, exclude_without=True, include_prompt=False)
