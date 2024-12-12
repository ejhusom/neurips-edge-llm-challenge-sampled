import pandas as pd
import numpy as np

def sample_csv_proportional(dataset_file, output_file, category_column="Category", sample_size=200, seed=42):
    """
    Perform proportional stratified sampling on a dataset and save the sampled dataset.

    Parameters:
    - dataset_file (str): Path to the input dataset file (CSV format).
    - output_file (str): Path to save the sampled dataset (CSV format).
    - category_column (str): Column name for the category to stratify by.
    - sample_size (int): Total number of items to sample (default: 42).
    - seed (int): Seed for random number generator (default: 42).

    Returns:
    - None
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Load the dataset 
    all_data = pd.read_csv(dataset_file)

    # Calculate category distribution
    category_counts = all_data[category_column].value_counts()
    category_proportions = category_counts / category_counts.sum()

    # Determine the number of samples for each category
    category_samples = (category_proportions * sample_size).round().astype(int)
    
    # Adjust for rounding discrepancies
    total_samples = category_samples.sum()
    print(total_samples)
    while total_samples != sample_size:
        diff = sample_size - total_samples
        adjustment = np.sign(diff)  # +1 if deficit, -1 if surplus
        # Adjust one category at a time, prioritizing the largest proportions
        category_samples.iloc[0] += adjustment
        total_samples = category_samples.sum()

    # Perform proportional sampling
    sampled_data = pd.DataFrame()
    for category, n_samples in category_samples.items():
        category_data = all_data[all_data[category_column] == category]
        sampled_data = pd.concat([sampled_data, category_data.sample(n=min(n_samples, len(category_data)), random_state=seed)])

    # Shuffle the final sampled dataset
    sampled_data = sampled_data.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Save the sampled dataset to a new CSV file
    sampled_data.to_csv(output_file, index=False)

    print(f"Sampled {len(sampled_data)} data points from {len(all_data)} total entries across {len(category_counts)} categories using seed {seed}.")


# Usage
sample_csv_proportional('TruthfulQA.csv', 'truthfulQA_gen_200_samples.csv', category_column="Category", sample_size=200, seed=42)