import json
import pandas as pd
import numpy as np

def sample_hierarchical_jsonl(dataset_file, output_file, sample_size = 200, seed = 42):
    """
    Sample entries from a JSONL dataset in proportion to their original main category distribution,
    then sample subcategories proportionally within their main category.

    Args:
        dataset_file (str): Path to the input JSONL file.
        output_file (str): Path to save the sampled JSONL file.
        sample_size (int): Total number of entries to sample.
        seed (int): Random seed for reproducibility.

    Returns:
        None
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Read the JSONL file into a DataFrame
    with open(dataset_file, 'r') as f:
        all_data = [json.loads(line) for line in f]
    df = pd.DataFrame(all_data)
    
    # Define the mapping of subcategories to main categories
    sub_to_main_category = {
        'logical_deduction_three_objects': 'logical_deduction',
        'logical_deduction_five_objects': 'logical_deduction',
        'logical_deduction_seven_objects': 'logical_deduction',
        'tracking_shuffled_objects_three_objects': 'tracking_shuffled_objects',
        'tracking_shuffled_objects_five_objects': 'tracking_shuffled_objects',
        'tracking_shuffled_objects_seven_objects': 'tracking_shuffled_objects'
    }
    
    # Add a 'main_category' column to the DataFrame
    df['main_category'] = df['category'].map(sub_to_main_category).fillna(df['category'])
    
    # Calculate main category distribution
    main_category_counts = df['main_category'].value_counts()
    main_category_proportions = main_category_counts / main_category_counts.sum()
    
    # Determine initial number of samples for each main category
    main_category_samples = (main_category_proportions * sample_size).round().astype(int)
    
    # Adjust for rounding discrepancies
    total_samples = main_category_samples.sum()
    while total_samples != sample_size:
        diff = sample_size - total_samples
        adjustment = np.sign(diff)  # +1 if deficit, -1 if surplus
        
        # Prioritize main categories based on relative rounding error
        proportionate_diffs = (main_category_proportions * sample_size) - main_category_samples
        prioritized_main_category = proportionate_diffs.idxmax() if adjustment > 0 else proportionate_diffs.idxmin()
        
        # Perform the adjustment
        main_category_samples[prioritized_main_category] += adjustment
        total_samples = main_category_samples.sum()
    
    # Perform proportional sampling within each main category
    sampled_data = pd.DataFrame()
    for main_category, n_main_samples in main_category_samples.items():
        main_category_data = df[df['main_category'] == main_category]
        
        # Calculate subcategory distribution within the main category
        subcategory_counts = main_category_data['category'].value_counts()
        subcategory_proportions = subcategory_counts / subcategory_counts.sum() 
        
        # Determine number of samples for each subcategory
        subcategory_samples = (subcategory_proportions * n_main_samples).round().astype(int)
        
        # Adjust for rounding discrepancies at the subcategory level
        sub_total_samples = subcategory_samples.sum()
        while sub_total_samples != n_main_samples:
            diff = n_main_samples - sub_total_samples
            adjustment = np.sign(diff)  # +1 if deficit, -1 if surplus
            
            # Prioritize subcategories based on relative rounding error
            proportionate_diffs = (subcategory_proportions * n_main_samples) - subcategory_samples
            prioritized_subcategory = proportionate_diffs.idxmax() if adjustment > 0 else proportionate_diffs.idxmin()
            
            # Perform the adjustment
            subcategory_samples[prioritized_subcategory] += adjustment
            sub_total_samples = subcategory_samples.sum()
        
        # Sample data from each subcategory
        for subcategory, n_sub_samples in subcategory_samples.items():
            subcategory_data = main_category_data[main_category_data['category'] == subcategory]
            sampled_data = pd.concat([sampled_data, subcategory_data.sample(n=min(n_sub_samples, len(subcategory_data)), random_state=seed)])
    
    # Shuffle the final sampled dataset
    sampled_data = sampled_data.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Save the sampled dataset to a JSONL file
    with open(output_file, 'w') as f:
        for record in sampled_data.to_dict(orient='records'):
            f.write(json.dumps(record) + '\n')
    
    print(f"Sampled {len(sampled_data)} data points from {len(all_data)} total entries across {len(main_category_counts)} main categories using seed {seed}.")


# Usage example
input_file_path = "collated_bbh_dataset.jsonl"
output_file_path = "collated_bbh_200_samples.jsonl"
sample_hierarchical_jsonl(dataset_file=input_file_path, output_file=output_file_path, sample_size=200)

