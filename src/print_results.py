import json
import sys

def print_assistant_output(jsonl_file):
    with open(jsonl_file, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            output_content = data['output']['message']['content'].strip()
            print(output_content)
            print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python print_results.py <input_jsonl_file>")
        sys.exit(1)

    jsonl_file = sys.argv[1]
    print_assistant_output(jsonl_file)
