#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze results from LLM inference at the edge experiments.

Plots:
    - Average accuracy per model per dataset
    - Average accuracy vs model size (in bytes)


"""
import os
import sys

import json
import pandas as pd
import matplotlib.pyplot as plt

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

def get_color_for_model(model_name):

    model_base_name = "_".join(model_name.split("_")[:2])

    colors = {
        "qwen25_05b": "skyblue",
        "qwen25_15b": "blue",
        "gemma2_2b": "orange",
        "llama32_1b": "green",
    }
    return colors.get(model_base_name, "gray")

def get_model_info(filepath="ollama_models.json"):

    with open(filepath, "r") as infile:
        model_info = json.load(infile)

    return model_info

def plot_accuracy_subplots_vertical_bars(accuracy):
    # Plot the accuracy results
    datasets = set()
    for model in accuracy:
        datasets.update(accuracy[model].keys())

    datasets = sorted(datasets)
    num_datasets = len(datasets)

    fig, axes = plt.subplots(num_datasets, 1, figsize=(8, 2 * num_datasets), sharex=True)
    if num_datasets == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        ax.set_title(f"Accuracy for {dataset}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)

        models = []
        accuracies = []
        colors = []

        for model in accuracy:
            if dataset in accuracy[model]:
                models.append(model)
                accuracies.append(accuracy[model][dataset])
                colors.append(get_color_for_model(model))

        ax.bar(models, accuracies, color=colors)
        ax.set_xticklabels(models, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("accuracy_comparison.png")
    plt.show()

def plot_accuracy_subplots_horizontal_bars(accuracy):
    # Plot the accuracy results in a horizontal bar plot
    fig, axes = plt.subplots(num_datasets, 1, figsize=(12, 6 * num_datasets), sharey=True)
    if num_datasets == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        ax.set_title(f"Accuracy for {dataset}")
        ax.set_ylabel("Model")
        ax.set_xlabel("Accuracy")
        ax.set_xlim(0, 1)

        models = []
        accuracies = []
        colors = []

        for model in accuracy:
            if dataset in accuracy[model]:
                models.append(model)
                accuracies.append(accuracy[model][dataset])
                colors.append(get_color_for_model(model))

        ax.barh(models, accuracies, color=colors)
        ax.set_yticklabels(models, rotation=0, ha='right')

    plt.tight_layout()
    plt.savefig("accuracy_comparison_horizontal.png")
    plt.show()

def plot_accuracy_separate_datasets(accuracy):
    # Plot the accuracy results
    datasets = set()
    for model in accuracy:
        datasets.update(accuracy[model].keys())

    datasets = sorted(datasets)

    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(f"Accuracy for {dataset}")
        ax.set_ylabel("Model")
        ax.set_xlabel("Accuracy")
        ax.set_xlim(0, 1)

        models = []
        accuracies = []
        colors = []

        for model in accuracy:
            if dataset in accuracy[model]:
                models.append(model)
                accuracies.append(accuracy[model][dataset])
                colors.append(get_color_for_model(model))

        ax.barh(models, accuracies, color=colors)
        ax.set_yticklabels(models, rotation=0, ha='right')

        plt.tight_layout()
        plt.savefig(f"accuracy_comparison_{dataset}.pdf")
        plt.show()

def plot_accuracy_vs_model_size(accuracy):

    model_info = get_model_info()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Accuracy vs model size")
    ax.set_xlabel("Model size (bytes)")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)

    for model in accuracy:
        breakpoint()
        model_size = model_info[model.replace("_", ":", 1).replace("_", "-", 2)]["size"]
        ax.scatter(model_size, accuracy[model], color=get_color_for_model(model))

    plt.tight_layout()
    plt.savefig("accuracy_vs_model_size.png")
    plt.show()

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

    with open(accuracy_filepath, "w+") as outfile:
        json.dump(accuracy, outfile, indent=4)

    print("Final results:")
    print(accuracy)

    # Plot the results
    # plot_accuracy_subplots_vertical_bars(accuracy)
    # plot_accuracy_subplots_horizontal_bars(accuracy)
    plot_accuracy_separate_datasets(accuracy)
    # plot_accuracy_vs_model_size(accuracy)
