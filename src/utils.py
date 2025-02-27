#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import json

def get_dataset_name(file_name):
    parts = file_name.split('_')
    return parts[0].lower()

def get_model_name(file_name):
    parts = file_name.split(".")[0].split('_')
    return "_".join(parts[1:])

def get_color_for_model(model_name):

    model_base_name = "_".join(model_name.split("_")[:2])

    colors = {
        "qwen25_05b": "skyblue",
        "qwen25_15b": "blue",
        "gemma2_2b": "darkblue",
        "llama32_1b": "turquoise",
    }
    return colors.get(model_base_name, "gray")

def get_markers():
    """Return a dictionary of model markers."""
    return {
        "fp16": "o",
        "q8_0": "X",
        "q4_1": "s",
        "q4_K_M": "P",
        "q4_0": "D",
        "q4_K_S": (4, 1, 0),
        "q3_K_L": "^",
        "q3_K_M": (4, 1, 45),
        "q3_K_S": "v",
    }

def get_markers_for_dataset():
    """Return a dictionary of dataset markers."""
    return {
        "bigbenchhard": (8, 0, 0),
        "commonsenseqa": "*",
        "gsm8k": (8, 1, 0),
        "humaneval": "d",
        "truthfulqa": "p"
    }

def get_colors():
    """Return a dictionary of model colors."""
    return {
        "bigbenchhard": "#88CCEE",
        "commonsenseqa": "#882255",
        "gsm8k": "#44AA99",
        "humaneval": "#999933",
        "truthfulqa": "#AA4499",
        "qwen25_05b": "#CC6677",
        "qwen25_05b_instruct": "#CC6677",
        "qwen25_05b_instruct_fp16": "#CC6677",
        "qwen25_05b_instruct_q8_0": "#CC6677",
        "qwen25_05b_instruct_q4_1": "#CC6677",
        "qwen25_05b_instruct_q4_K_M": "#CC6677",
        "qwen25_05b_instruct_q4_0": "#CC6677",
        "qwen25_05b_instruct_q4_K_S": "#CC6677",
        "qwen25_05b_instruct_q3_K_L": "#CC6677",
        "qwen25_05b_instruct_q3_K_M": "#CC6677",
        "qwen25_05b_instruct_q3_K_S": "#CC6677",
        "qwen25_15b": "#332288",
        "qwen25_15b_instruct": "#332288",
        "qwen25_15b_instruct_fp16": "#332288",
        "qwen25_15b_instruct_q8_0": "#332288",
        "qwen25_15b_instruct_q4_1": "#332288",
        "qwen25_15b_instruct_q4_K_M": "#332288",
        "qwen25_15b_instruct_q4_0": "#332288",
        "qwen25_15b_instruct_q4_K_S": "#332288",
        "qwen25_15b_instruct_q3_K_L": "#332288",
        "qwen25_15b_instruct_q3_K_M": "#332288",
        "qwen25_15b_instruct_q3_K_S": "#332288",
        "gemma2_2b": "#DDCC77",
        "gemma2_2b_instruct": "#DDCC77",
        "gemma2_2b_instruct_fp16": "#DDCC77",
        "gemma2_2b_instruct_q8_0": "#DDCC77",
        "gemma2_2b_instruct_q4_1": "#DDCC77",
        "gemma2_2b_instruct_q4_K_M": "#DDCC77",
        "gemma2_2b_instruct_q4_0": "#DDCC77",
        "gemma2_2b_instruct_q4_K_S": "#DDCC77",
        "gemma2_2b_instruct_q3_K_L": "#DDCC77",
        "gemma2_2b_instruct_q3_K_M": "#DDCC77",
        "gemma2_2b_instruct_q3_K_S": "#DDCC77",
        "llama32_1b": "#117733",
        "llama32_1b_instruct": "#117733",
        "llama32_1b_instruct_fp16": "#117733",
        "llama32_1b_instruct_q8_0": "#117733",
        "llama32_1b_instruct_q4_1": "#117733",
        "llama32_1b_instruct_q4_K_M": "#117733",
        "llama32_1b_instruct_q4_0": "#117733",
        "llama32_1b_instruct_q4_K_S": "#117733",
        "llama32_1b_instruct_q3_K_L": "#117733",
        "llama32_1b_instruct_q3_K_M": "#117733",
        "llama32_1b_instruct_q3_K_S": "#117733",
    }

def get_model_colors():
    """Return a dictionary of model colors."""
    return {
        "qwen25_05b": "skyblue",
        "qwen25_15b": "blue",
        "gemma2_2b": "orange",
        "llama32_1b": "green",
        "bigbenchhard": "olive",
        "commonsenseqa": "pink",
        "gsm8k": "brown",
        "humaneval": "red",
        "truthfulqa": "purple",
    }

def get_correctness_colors():
    """Return a dictionary of correctness colors."""
    return {
        "correct": "green",
        "Correct": "green",
        "incorrect": "red",
        "Incorrect": "red",
    }

def plot_legend(ax, location="best"):
    colors = {
        "qwen25_05b": "skyblue",
        "qwen25_15b": "blue",
        "gemma2_2b": "orange",
        "llama32_1b": "green",
    }

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors.values()]
    labels = colors.keys()

    if location == "outside":
        ax.legend(handles, labels, title="Base model", loc='upper center', bbox_to_anchor=(0.5, 1.9), ncol=4)
    else:
        ax.legend(handles, labels, title="Base model", loc='best')

def get_model_info(filepath="ollama_models.json"):

    with open(filepath, "r") as infile:
        model_info = json.load(infile)

    return model_info
