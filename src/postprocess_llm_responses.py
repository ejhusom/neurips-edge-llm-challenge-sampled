#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

import json
import pandas as pd
import numpy as np
import jsonlines

import evaluate_commonsenseQA
import evaluate_gsm8k
import evaluate_HumanEval
import evaluate_TruthfulQA
import evaluate_bbh

def read_llm_response_file(filepath):
    """Read LLM response data.

    Will perform evaluation of the response. The resulting dataframe will include the correctness of each response, and the start and stop timestamps for matching up against the energy consumption data.

    """

    df = pd.DataFrame(columns=[
        "dataset",
        "model",
        "formatted_prompt",
        "created_at",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
        "response",
        "energy_consumption_kwh",
    ])

    filename = os.path.basename(filepath)
    dataset = get_dataset_name(filename)
    model = get_model_name(filename)

    print(f"Evaluating {filepath}")

    if dataset.lower() == "commonsenseqa":
        model_accuracies, evaluation_df = evaluate_commonsenseQA.main(filepath)
    elif dataset.lower() == "bigbenchhard":
        model_accuracies, evaluation_df = evaluate_bbh.main(filepath)
    elif dataset.lower() == "gsm8k":
        model_accuracies, evaluation_df = evaluate_gsm8k.main(filepath)
    elif dataset.lower() == "truthfulqa":
        model_accuracies, evaluation_df = evaluate_TruthfulQA.main(filepath)
    elif dataset.lower() == "humaneval":
        performance, evaluation_df = evaluate_HumanEval.main(filepath)
    else:
        print("Dataset not recognized")
        sys.exit(1)

    print(f"Saving to dataframe")

    # Read JSONL file and load data
    with jsonlines.open(filepath, mode='r') as reader:
        for i, obj in enumerate(reader):
            df.loc[len(df)] = [
                dataset,
                model,
                obj["formatted_prompt"],
                obj["output"]["created_at"],
                obj["output"]["total_duration"],
                obj["output"]["load_duration"],
                obj["output"]["prompt_eval_count"],
                obj["output"]["prompt_eval_duration"],
                obj["output"]["eval_count"],
                obj["output"]["eval_duration"],
                obj["output"]["response"],
                None,
            ]

    print("Processing timestamps")

    df["created_at"] = pd.to_datetime(df["created_at"], format="ISO8601")
    df["total_duration"] = pd.to_timedelta(df["total_duration"], unit="ns")
    df["stopped_at"] = df["created_at"] + df["total_duration"]
    # breakpoint()
    df["evaluation"] = evaluation_df["evaluation"].astype(bool)

    return df

def get_dataset_name(file_name):
    parts = file_name.split('_')
    return parts[0]

def get_model_name(file_name):
    parts = file_name.split(".")[0].split('_')
    return "_".join(parts[1:])

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: postprocess_llm_responses.py <file1.jsonl> <file2.jsonl> ...")
        sys.exit(1)

    filepaths = sys.argv[1:]

    for filepath in filepaths:
        print(f"Processing {filepath}.")
        # Read LLM response data. 
        df = read_llm_response_file(filepath)
        print("Saving to file")
        df.to_csv(filepath.replace("jsonl", "csv"))
        print("Done!")
        print("=====================")
