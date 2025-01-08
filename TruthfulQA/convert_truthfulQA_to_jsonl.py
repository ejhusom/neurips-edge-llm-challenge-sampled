import json

# Load the JSON data
with open('truthfulQA_MC_200_samples.json', 'r') as f:
    data = json.load(f)

# Open a new JSONL file for writing
with open('truthfulQA_MC_200_samples.jsonl', 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')
