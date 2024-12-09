import json
import random

def sample_jsonl_files(train_file, test_file, output_file, sample_size=200, seed=42):
    # Set the random seed for reproducibility
    random.seed(seed)
    
    # Read data from both JSONL files
    all_data = []
    
    # Read train.jsonl
    with open(train_file, 'r') as f:
        all_data.extend(json.loads(line.strip()) for line in f)
    
    # Read test.jsonl
    with open(test_file, 'r') as f:
        all_data.extend(json.loads(line.strip()) for line in f)
    
    # Perform uniform sampling
    sampled_data = random.sample(all_data, min(sample_size, len(all_data)))
    
    # Write sampled data to output file
    with open(output_file, 'w') as f:
        for item in sampled_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Sampled {len(sampled_data)} data points from {len(all_data)} total entries using seed {seed}.")

# Usage
sample_jsonl_files('train_rand_split.jsonl', 'dev_rand_split.jsonl', 'commonsenseqa_200_samples.jsonl', sample_size=200, seed=42)
