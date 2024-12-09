# CommonSenseQA – Question Answering Challenge Targeting Commonsense Knowledge

- Description: Multiple choice question answering, targeting commonsense knowledge.
- Link to source: [https://www.tau-nlp.sites.tau.ac.il/commonsenseqa](https://www.tau-nlp.sites.tau.ac.il/commonsenseqa)
- No. of tasks for full dataset: 12 102.
    - 1140 of these samples are test samples where no solution is provided in the public dataset.
- No. of tasks with solution: 10 962.
- No. of tasks for reduced dataset: 200 

No particular distribution or sorting was found or indicated in the original dataset. The reduced dataset was obtained using a uniform sampling from the original data (`train_rand_split.jsonl` and `dev_rand_split.jsonl` combined). The script `sampling_commonsenseqa.py` was used for producing `commonsenseqa_200_samples.jsonl`.
