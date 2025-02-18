#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

import json
import pandas as pd

from utils import get_dataset_name, get_model_name

def calculate_accuracy_for_df(df):
    """Calculate accuracy of LLM responses.

    Assumes df contains rows with LLM responses, including a column
    "evaluation", with True/False values based on whether the response
    was correct or not wrt. certain evaluation. This script assumes that
    the evaluation already is performed, with the result put in the
    "evaluation" column.

    """

    return df["evaluation"].sum() / len(df["evaluation"])

def calculate_energy_consumption_for_df(df):

    energy_consumption = {
            "mean": None,
            "std": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "Q1": None,
            "Q3": None,
            "IQR": None,
    }

    return energy_consumption

if __name__ == '__main__':

    # Read filepaths from command line.
    # The filepaths are assumed to have the following format:
    #   [dataset]_[modelname]_[modelsize]_[tags].jsonl
    # 
    # Example:
    #   commonsenseqa_qwen25_15b_instruct_q4_0.jsonl

    filepaths = sys.argv[1:]

    # Data to be extracted:
    #   Accuracy:
    #       model -> dataset -> accuracy
    #   Energy consumption:
    #       model -> dataset -> energy consumption -> mean, std, IQR, etc
    accuracy = {}
    energy_consumption = {}

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        dataset = get_dataset_name(filename)
        model = get_model_name(filename)

        print(f"Processing {model} on {dataset}...")
        
        df = pd.read_csv(filepath, index_col=0)

        # Calculate performance
        if not model in accuracy:
            accuracy[model] = {}
            energy_consumption[model] = {}

        accuracy[model][dataset] = calculate_accuracy_for_df(df)
        print(f"Accuracy: {accuracy[model][dataset]}")

        energy_consumption[model][dataset] = calculate_energy_consumption_for_df(df)
        print(f"Energy consumption (mean): {energy_consumption[model][dataset]['mean']}")

        print("========================")

    accuracy_filepath = "accuracy_all_models.json"

    with open(filepath, "w+") as outfile:
        json.dump(accuracy, outfile, indent=4)

    print("Final results:")
    print(accuracy)
