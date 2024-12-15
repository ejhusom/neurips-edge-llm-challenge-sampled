# BIG-Bench Hard Tasks

- Description: A dataset consisting of 6511 tasks across 23 main categories, including logic, language understanding, and arithmetic reasoning[^1].
- Link to Source: [https://github.com/suzgunmirac/BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard)
- No. of tasks in full dataset: 6511
- No. of tasks for reduced dataset: 200

---
The original dataset consists of separate JSON files, each corresponding to a specific task category. These files were collated into a single dataset before sampling.

The collated dataset contains 23 challenging BIG-Bench tasks where two tasks (Logical deduction and Tracking shuffled objects) have three subcategories. To maintain the distribution of main categories and their subcategories, hierarchical stratified sampling was performed. This ensures that the sampled dataset represents both the diversity of the tasks and their prevalence in the original dataset.

The script `sample_bbh_collated.py` was used for this process, and the output file was named `collated_bbh_200_samples.jsonl`.

[^1]: Mirac Suzgun, et al. 2022. [Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them.](https://arxiv.org/abs/2210.09261) arXiv preprint arXiv:2210.09261 (2022).



