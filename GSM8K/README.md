# GSM8K – Grade School Math

- Description: A dataset of "high quality linguistically diverse grade school math word problems"[^1].
- Link to source: [https://github.com/openai/grade-school-math](https://github.com/openai/grade-school-math)
- No. of tasks for full dataset: 8.5K
- No. of tasks for reduced dataset: 200 

No particular distribution or sorting was found or indicated in the original dataset. The reduced dataset was obtained using a uniform sampling from the original data (`train.jsonl` and `test.jsonl` combined). The script `sampling_gsm8k.py` was used for producing `gsm8k_200_samples.jsonl`.

[^1]: [https://github.com/openai/grade-school-math](https://github.com/openai/grade-school-math)
