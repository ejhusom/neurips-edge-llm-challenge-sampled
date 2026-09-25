# Website: Sustainable LLM Inference for Edge AI

An interactive companion to Husom et al. (2025), *Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency, Output Accuracy, and Inference Latency*, ACM Transactions on Internet of Things 6(4):28, [doi:10.1145/3767742](https://doi.org/10.1145/3767742).

The published page is a single file, [`docs/index.html`](../docs/index.html). It has no external dependencies: the data, D3 and the fonts are all inlined, so it also works when opened straight from disk without a network connection.

## Layout

```
website/
  src/            page template (index.html), styles.css, app.js
  build/
    build_data.py   raw measurements -> data.json (needs pandas)
    data.json       prepared data, committed so the page can be rebuilt without the raw data
    build_site.py   src + data.json + vendor -> docs/index.html (standard library only)
  vendor/         D3 v7.9.0 and the three web fonts, with their licenses
docs/
  index.html      the built page served by GitHub Pages
  .nojekyll
```

## Rebuilding

After editing anything in `website/src/`:

```bash
python3 website/build/build_site.py
```

To regenerate `data.json` from the raw measurements (expects `../llm-edge-experiments-data/main` next to this repository, or set `EDGE_LLM_DATA`):

```bash
venv/bin/python website/build/build_data.py
```

`build_data.py` also checks that each Joulescope power trace integrates to roughly the logged energy for that answer.

## Publishing with GitHub Pages

In the repository settings on GitHub, open **Pages**, choose **Deploy from a branch**, and select `main` with the `/docs` folder. The site is then served at `https://ejhusom.github.io/neurips-edge-llm-challenge-sampled/`.

## Where the numbers come from

- Headline numbers in the text are quoted from the paper's tables.
- Charts use the idle-subtracted per-answer measurements (`llm_responses_with_energy_consumption_idle_subtracted_with_dots`), which reproduce Tables 6, 7, 8, 9 and 10 exactly.
- The all-tasks Pareto frontier averages both energy and accuracy over the five tasks, which gives 8 frontier models. Fig. 6 in the paper pairs CommonsenseQA energy with average accuracy.
- The split between reading and writing time comes from Ollama's timing fields (`prompt_eval_duration`, `eval_duration`) for the same runs.
- The 60 questions in the inspector are a seeded random sample (12 per task, mostly questions with mixed results across models).
