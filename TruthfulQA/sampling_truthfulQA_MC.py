# %%
import json
import random

def sample_json_files(dataset_file, output_file, sample_size=200, seed=42):
    # Set the random seed for reproducibility
    random.seed(seed)

    # Load the dataset 
    with open(dataset_file, "r") as f:
        all_data = json.load(f)

    # Perform uniform sampling
    sampled_data = random.sample(all_data, sample_size)

    # Save the sampled dataset
    with open(output_file, "w") as f:
        json.dump(sampled_data, f, indent=2)

    print(f"Sampled {len(sampled_data)} data points from {len(all_data)} total entries using seed {seed}.")

# Usage
sample_json_files('mc_task.json', 'truthfulQA_MC_200_samples.json', sample_size=200, seed=42)


