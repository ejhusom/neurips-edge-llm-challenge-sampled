#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze results from LLM inference at the edge experiments.

Plots:
    - Average accuracy per model per dataset
    - Average accuracy vs model size (in bytes)


"""
import os
import re
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

def calculate_statistics_for_df(df, column="energy_consumption_joules"):

    statistics = {
            "mean": None,
            "std": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "Q1": None,
            "Q3": None,
            "IQR": None,
    }

    # Column is named "energy_consumption_joules" in the dataframe
    statistics["mean"] = df[column].mean()
    statistics["std"] = df[column].std()
    statistics["median"] = df[column].median()
    statistics["minimum"] = df[column].min()
    statistics["maximum"] = df[column].max()
    statistics["Q1"] = df[column].quantile(0.25)
    statistics["Q3"] = df[column].quantile(0.75)
    statistics["IQR"] = statistics["Q3"] - statistics["Q1"]

    return statistics

def get_color_for_model(model_name):

    model_base_name = "_".join(model_name.split("_")[:2])

    colors = {
        "qwen25_05b": "skyblue",
        "qwen25_15b": "blue",
        "gemma2_2b": "orange",
        "llama32_1b": "green",
    }
    return colors.get(model_base_name, "gray")

def plot_legend(ax):
    colors = {
        "qwen25_05b": "skyblue",
        "qwen25_15b": "blue",
        "gemma2_2b": "orange",
        "llama32_1b": "green",
    }

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors.values()]
    labels = colors.keys()
    ax.legend(handles, labels, title="Base model", loc='best')

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

def plot_energy_consumption_subplots_vertical_bars(energy_consumption):
    # Plot the energy consumption results
    datasets = set()
    for model in energy_consumption:
        datasets.update(energy_consumption[model].keys())

    datasets = sorted(datasets)
    num_datasets = len(datasets)

    fig, axes = plt.subplots(num_datasets, 1, figsize=(8, 2 * num_datasets), sharex=True)
    if num_datasets == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        ax.set_title(f"Energy consumption for {dataset}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Energy consumption (J)")

        models = []
        energy_consumptions = []
        colors = []

        for model in energy_consumption:
            if dataset in energy_consumption[model]:
                models.append(model)
                energy_consumptions.append(energy_consumption[model][dataset]["mean"])
                colors.append(get_color_for_model(model))

        ax.bar(models, energy_consumptions, color=colors)
        ax.set_xticklabels(models, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("energy_consumption_comparison.png")
    plt.show()

def plot_metric_vs_model_size(metric, name=""):

    datasets = set()
    for model in metric:
        datasets.update(metric[model].keys())

    datasets = sorted(datasets)

    model_info = get_model_info()

    for dataset in datasets:

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(f"{name} vs model size – {dataset}")
        ax.set_xlabel("Model size (bytes)")
        ax.set_ylabel("{name}")

        models = []
        metrics = []
        model_sizes = []
        colors = []

        for model in metric:
            if dataset in metric[model]:
                models.append(model)
                metrics.append(metric[model][dataset])
                colors.append(get_color_for_model(model))

                # Translate model to model key. Need to handle cases like:
                #   qwen25_05b_instruct_q4_0 -> qwen2.5:0.5b-instruct-q4_0
                #   qwen25_15b_instruct_q4_0 -> qwen2.5:1.5b-instruct-q4_0
                #   llama32_1b_instruct_q3_K_L -> llama3.2:1b-instruct-q3_K_L
                #   gemma2_2b_instruct_q3_K_L -> gemma2:2b-instruct-q3_K_L
                model_key = model.replace("_", ":", 1).replace("_", "-", 2)
                # Use regex to put a dot between cases where there are two numbers.
                # FIXME: This will break for models with >=10b parameters. Need to change the filename format to handle this.
                model_key = re.sub(r"(\d)(\d)", r"\1.\2", model_key, count=2)

                model_key = next((item['model'] for item in model_info['models'] if item['name'] == model_key), None)

                if model_key:
                    model_size = next(item['size'] for item in model_info['models'] if item['model'] == model_key)
                    model_sizes.append(model_size)

        ax.scatter(model_sizes, metrics, color=colors)
        plot_legend(ax)

        plt.tight_layout()
        plt.savefig(f"{name}_vs_model_size_{dataset}.pdf")
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
    all_data = {}
    accuracy = {}
    energy_consumption = {}
    energy_consumption_mean = {}
    energy_consumption_per_token = {}
    energy_consumption_mean_per_token = {}

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        dataset = get_dataset_name(filename)
        model = get_model_name(filename)

        print(f"Processing {model} on {dataset}...")
        
        df = pd.read_csv(filepath, index_col=0)
        df["energy_consumption_joules_per_token"] = df["energy_consumption_joules"] / df["eval_count"]

        # Calculate performance
        if not model in accuracy:
            all_data[model] = {}
            accuracy[model] = {}
            energy_consumption[model] = {}
            energy_consumption_mean[model] = {}
            energy_consumption_per_token[model] = {}
            energy_consumption_mean_per_token[model] = {}

        all_data[model][dataset] = df

        accuracy[model][dataset] = calculate_accuracy_for_df(df)
        print(f"Accuracy: {accuracy[model][dataset]}")

        energy_consumption[model][dataset] = calculate_statistics_for_df(df, "energy_consumption_joules")
        energy_consumption_per_token[model][dataset] = calculate_statistics_for_df(df, "energy_consumption_joules_per_token")
        energy_consumption_mean[model][dataset] = energy_consumption[model][dataset]["mean"]
        energy_consumption_mean_per_token[model][dataset] = energy_consumption_per_token[model][dataset]["mean"]
        print(f"Energy consumption (mean): {energy_consumption[model][dataset]['mean']}")

        print("========================")

    breakpoint()

    accuracy_filepath = "accuracy_all_models.json"
    energy_consumption_filepath = "energy_consumption_all_models.json"

    with open(accuracy_filepath, "w+") as outfile:
        json.dump(accuracy, outfile, indent=4)

    with open(energy_consumption_filepath, "w+") as outfile:
        json.dump(energy_consumption, outfile, indent=4)

    print("Final results:")
    print(accuracy)

    # Plot the results
    # plot_accuracy_subplots_vertical_bars(accuracy)
    # plot_accuracy_subplots_horizontal_bars(accuracy)
    # plot_accuracy_separate_datasets(accuracy)
    # plot_metric_vs_model_size(accuracy, "Accuracy")

    # plot_energy_consumption_subplots_vertical_bars(energy_consumption)
    # plot_metric_vs_model_size(energy_consumption_mean, "Energy consumption (J)")
    # plot_metric_vs_model_size(energy_consumption_mean_per_token, "Energy consumption per token (J)")
