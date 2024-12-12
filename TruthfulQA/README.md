# TruthfulQA: Measuring How Models Mimic Human Falsehoods

- Description: A dataset of 817 diverse questions across 38 categories, including health, law, finance and politics, to assess language model's truthfulness in answering.[^1].
- Link to source: [https://github.com/sylinrl/TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- No. of tasks for full dataset: 817
- No. of tasks for reduced dataset: 200 

#### Generation task: Stratified Sampling
In this version, the dataset contains 817 generation tasks categorized into 38 different categories. To reflect the same category distribution as the original, proportional stratified sampling was performed.
The proportional sampling ensures that each category in the reduced dataset has a number of entries proportional to its prevalence in the original dataset.

The script `sampling_truthfulQA_gen.py` was used for this purpose, and the output file was named `truthfulQA_gen_200_samples.csv`.

---
#### Multiple-choice task: Uniform sampling
In this version, the dataset consists of 817 multiple-choice questions without any categorization. Since no particular distribution or sorting was indicated in the original dataset, the reduced dataset was created using uniform sampling. 

The script `sampling_truthfulQA_MC.py` was used for this purpose, and the output file was named `truthfulQA_MC_200_samples.json`.

---

Both scripts ensure reproducibility by setting a fixed random seed (42).

[^1]: Stephanie Lin, Jacob Hilton, and Owain Evans. 2021. [TruthfulQA: Measuring How Models Mimic Human Falsehoods.](https://arxiv.org/abs/2109.07958) arXiv preprint arXiv:2109.07958 (2021).

