import jsonlines
import argparse
import matplotlib.pyplot as plt
import os

def analyze_file(file_path):
    eval_counts = []
    eval_durations = []
    dataset_model = ""

    with jsonlines.open(file_path, mode='r') as reader:
        for i, obj in enumerate(reader):
            if i == 0:
                dataset_model = get_dataset_name(file_path) + " - " + get_model_name(file_path)
            eval_count = obj['output'].get('eval_count')
            eval_duration = obj['output'].get('eval_duration')
            if eval_count is not None and eval_duration is not None:
                eval_counts.append(eval_count)
                eval_durations.append(eval_duration)

    return eval_counts, eval_durations, dataset_model

def get_dataset_name(file_name):
    parts = file_name.split('_')
    return parts[2]

def get_model_name(file_name):
    parts = file_name.split('_')
    return parts[3]

def plot_analysis(file_data, output_dir):
    eval_counts_data = []
    eval_durations_data = []
    labels = []

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    for i, (file_name, (eval_counts, eval_durations, dataset_model)) in enumerate(file_data.items()):
        color = 'blue' if 'without_instruction' in file_name else 'red'
        plt.boxplot(eval_counts, positions=[i], patch_artist=True, boxprops=dict(facecolor=color))
        labels.append(f"{dataset_model}")
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.xlabel('Dataset - Model')
    plt.ylabel('Eval Count')
    plt.title('Eval Count Comparison')
    plt.legend(handles=[plt.Line2D([0], [0], color='blue', lw=4, label='Without instruction'),
                        plt.Line2D([0], [0], color='red', lw=4, label='With instruction')])

    plt.subplot(1, 2, 2)
    for i, (file_name, (eval_counts, eval_durations, dataset_model)) in enumerate(file_data.items()):
        color = 'blue' if 'without_instruction' in file_name else 'red'
        plt.boxplot(eval_durations, positions=[i], patch_artist=True, boxprops=dict(facecolor=color))
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.xlabel('Dataset - Model')
    plt.ylabel('Eval Duration (ns)')
    plt.title('Eval Duration Comparison')
    plt.legend(handles=[plt.Line2D([0], [0], color='blue', lw=4, label='Without instruction'),
                        plt.Line2D([0], [0], color='red', lw=4, label='With instruction')])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_analysis.png'))
    plt.close()

def print_average_eval_counts(file_data):
    for file_name, (eval_counts, _, dataset_model) in file_data.items():
        average_eval_count = sum(eval_counts) / len(eval_counts) if eval_counts else 0
        print(f"Avg response length for {file_name} ({dataset_model}): {average_eval_count}")

def compute_average_reduction(file_data):
    paired_files = {}
    for file_name in file_data.keys():
        base_name = file_name.replace("_with_instruction", "").replace("_without_instruction", "")
        if base_name not in paired_files:
            paired_files[base_name] = {}
        if "with_instruction" in file_name:
            paired_files[base_name]["with"] = file_name
        else:
            paired_files[base_name]["without"] = file_name

    for base_name, files in paired_files.items():
        if "with" in files and "without" in files:
            with_counts = file_data[files["with"]][0]
            without_counts = file_data[files["without"]][0]
            if without_counts:
                avg_with = sum(with_counts) / len(with_counts)
                avg_without = sum(without_counts) / len(without_counts)
                reduction = ((avg_without - avg_with) / avg_without) * 100
                print(f"Average reduction in eval_count for {base_name}: {reduction:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Analyze JSONL files and create plots.")
    parser.add_argument('--files', type=str, nargs='+', required=True, help="Paths to the input JSONL files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output plots.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    file_data = {}
    for file_path in args.files:
        file_name = os.path.basename(file_path)
        dataset_name = get_dataset_name(file_name)
        eval_counts, eval_durations, dataset_model = analyze_file(file_path)
        file_data[file_name] = (eval_counts, eval_durations, dataset_model)

    plot_analysis(file_data, args.output_dir)
    print_average_eval_counts(file_data)
    compute_average_reduction(file_data)

if __name__ == "__main__":
    main()
