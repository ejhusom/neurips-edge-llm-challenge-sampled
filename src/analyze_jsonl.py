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
                dataset_model = obj['indata']['dataset'] + " - " + obj['output']['model']
            eval_counts.append(obj['output']['eval_count'])
            eval_durations.append(obj['output']['eval_duration'])

    return eval_counts, eval_durations, dataset_model

def plot_analysis(file_data, output_dir):
    eval_counts_data = []
    eval_durations_data = []
    labels = []

    for file_name, (eval_counts, eval_durations, dataset_model) in file_data.items():
        eval_counts_data.append(eval_counts)
        eval_durations_data.append(eval_durations)
        labels.append(f"{file_name}\n({dataset_model})")

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.boxplot(eval_counts_data, labels=labels)
    plt.xlabel('Files (Dataset - Model)')
    plt.ylabel('Eval Count')
    plt.title('Eval Count Comparison')

    plt.subplot(1, 2, 2)
    plt.boxplot(eval_durations_data, labels=labels)
    plt.xlabel('Files (Dataset - Model)')
    plt.ylabel('Eval Duration (ns)')
    plt.title('Eval Duration Comparison')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_analysis.png'))
    plt.close()

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
        eval_counts, eval_durations, dataset_model = analyze_file(file_path)
        file_data[file_name] = (eval_counts, eval_durations, dataset_model)

    plot_analysis(file_data, args.output_dir)

if __name__ == "__main__":
    main()
