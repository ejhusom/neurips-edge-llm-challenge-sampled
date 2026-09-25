"""Build data.json for the Edge LLM site.

Sources
- Paper tables (Husom et al. 2025, ACM TIoT 6(4):28) for published headline numbers.
- Raw per-inference CSVs (idle-subtracted, as used for the paper) for distributions,
  per-question inspection and Pareto frontiers.
- Joulescope 2 Hz statistics files for power traces of the curated inspector questions.

Run with the experiment repo's venv (needs pandas):
  ~/Documents/neurips-edge-llm-challenge-sampled/venv/bin/python build/build_data.py
"""
import glob, json, os, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data.json")
DATA_ROOT = "/Users/erikjohanneshusom/Documents/llm-edge-experiments-data/main"
RAW = os.path.join(DATA_ROOT, "llm_responses_with_energy_consumption_idle_subtracted_with_dots")
JS_STATS = os.path.join(DATA_ROOT, "joulescope_statistics")
IDLE_W = 2.85
PER_DATASET = 12
SEED = 7

DATASETS = ["commonsenseqa", "truthfulqa", "bigbenchhard", "gsm8k", "humaneval"]
DS_ALIASES = {"commonsenseqa": "commonsenseqa", "truthfulqa": "truthfulqa", "big-bench-hard": "bigbenchhard",
              "bigbenchhard": "bigbenchhard", "bbh": "bigbenchhard", "gsm8k": "gsm8k", "humaneval": "humaneval"}
FAMILIES = ["qwen2.5_0.5b", "llama3.2_1b", "qwen2.5_1.5b", "gemma2_2b"]
QUANT_ORDER = ["fp16", "q8_0", "q4_1", "q4_K_M", "q4_0", "q4_K_S", "q3_K_L", "q3_K_M", "q3_K_S"]

# Table 2 (size in MB, parameter count)
SIZES = {
    "llama3.2_1b": {"fp16": 2364.74, "q8_0": 1259.90, "q4_1": 793.23, "q4_K_M": 770.29, "q4_0": 735.23,
                    "q4_K_S": 739.73, "q3_K_L": 698.60, "q3_K_M": 658.85, "q3_K_S": 611.98},
    "qwen2.5_1.5b": {"q4_1": 969.75, "q4_K_M": 940.38, "q4_0": 891.66, "q4_K_S": 896.76, "q3_K_L": 839.40,
                     "q3_K_M": 786.01, "q3_K_S": 725.71},
    "qwen2.5_0.5b": {"fp16": 948.11, "q8_0": 506.48, "q4_1": 357.18, "q4_K_M": 379.39, "q4_0": 335.85,
                     "q4_K_S": 367.63, "q3_K_L": 352.26, "q3_K_M": 339.01, "q3_K_S": 322.61},
    "gemma2_2b": {"q3_K_S": 1297.64, "q3_K_M": 1393.96, "q3_K_L": 1478.62},
}
PARAMS = {"llama3.2_1b": "1.2B", "qwen2.5_1.5b": "1.5B", "qwen2.5_0.5b": "494M", "gemma2_2b": "2.6B"}
FAMILY_LABEL = {"llama3.2_1b": "Llama 3.2 1B", "qwen2.5_1.5b": "Qwen 2.5 1.5B",
                "qwen2.5_0.5b": "Qwen 2.5 0.5B", "gemma2_2b": "Gemma 2 2B"}


def load_raw():
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
        base = os.path.basename(f)[:-4]
        ds, model = base.split("_", 1)
        ds = DS_ALIASES[ds.lower()]
        df = pd.read_csv(f, index_col=0)
        df["ds"] = ds
        df["mdl"] = model
        df["row"] = np.arange(len(df))
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["fam"] = raw.mdl.str.extract(r"^(gemma2_2b|llama3\.2_1b|qwen2\.5_0\.5b|qwen2\.5_1\.5b)")[0]
    raw["q"] = raw.mdl.str.extract(r"instruct_(.*)$")[0]
    raw["latency_s"] = pd.to_timedelta(raw.total_duration).dt.total_seconds()
    for c in ["load_duration", "prompt_eval_duration", "eval_duration"]:
        raw[c + "_s"] = pd.to_numeric(raw[c], errors="coerce") / 1e9
    raw["E"] = pd.to_numeric(raw.energy_consumption_joules, errors="coerce")
    raw["tok"] = raw.eval_count.astype(float)
    raw["jpt"] = raw.E / raw.tok
    raw["correct"] = raw.evaluation.astype(str).str.lower().isin(["true", "1", "1.0"])
    raw["start"] = pd.to_datetime(raw.created_at, format="mixed", utc=True).dt.tz_localize(None)
    raw["stop"] = pd.to_datetime(raw.stopped_at, format="mixed", utc=True).dt.tz_localize(None)
    return raw


def model_list():
    models = []
    for fam in FAMILIES:
        for q in QUANT_ORDER:
            if q in SIZES[fam]:
                name, size = fam.split("_")  # Ollama tag, e.g. llama3.2:1b-instruct-q4_K_M
                models.append({"id": f"{fam}_instruct_{q}", "fam": fam, "q": q, "size": SIZES[fam][q],
                               "ollama": f"{name}:{size}-instruct-{q}"})
    return models


def quartiles(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    lo = x[x >= q1 - 1.5 * iqr].min()
    hi = x[x <= q3 + 1.5 * iqr].max()
    return [round(float(v), 4) for v in (lo, q1, med, q3, hi)]


def r4(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 4)


def pareto(points):
    """points: list of (id, energy, acc). Lower energy, higher accuracy is better."""
    pts = sorted(points, key=lambda p: (p[1], -p[2]))
    best, keep = -1, []
    for p in pts:
        if p[2] > best:
            keep.append(p[0])
            best = p[2]
    return keep


def load_joulescope_index():
    idx = []
    for f in sorted(glob.glob(os.path.join(JS_STATS, "*.csv"))):
        ts = pd.to_datetime(os.path.basename(f).split("-")[0], format="%Y%m%d_%H%M%S")
        idx.append((ts, f))
    return idx


_js_cache = {}


def js_frame(path):
    if path not in _js_cache:
        df = pd.read_csv(path, usecols=["#time", "power", "energy"])
        _js_cache[path] = df
    return _js_cache[path]


def trace_for(row, js_index, pad=3.0):
    """Power trace (W) at 2 Hz from pad s before start to pad s after stop."""
    cands = [(ts, f) for ts, f in js_index if ts < row.start]
    if not cands:
        return None
    ts0, path = max(cands, key=lambda t: t[0])
    df = js_frame(path)
    t_start = (row.start - ts0).total_seconds()
    t_stop = (row.stop - ts0).total_seconds()
    if t_stop > df["#time"].iloc[-1]:
        return None
    sel = df[(df["#time"] >= t_start - pad) & (df["#time"] <= t_stop + pad)]
    if len(sel) < 4:
        return None
    t0 = float(sel["#time"].iloc[0]) - t_start  # seconds relative to inference start (negative)
    power = [int(round(p * 100)) for p in sel["power"].tolist()]
    # sanity check: integrated energy above idle within the window should match the logged value
    inside = df[(df["#time"] >= t_start) & (df["#time"] <= t_stop)]
    e_meas = float(inside["energy"].iloc[-1] - inside["energy"].iloc[0]) - IDLE_W * (t_stop - t_start) if len(inside) > 1 else None
    return {"t0": round(t0, 2), "dt": 0.5, "p": power, "check": e_meas}


def main():
    raw = load_raw()
    models = model_list()
    mid = {m["id"]: i for i, m in enumerate(models)}
    assert set(raw.mdl.unique()) == set(mid), set(raw.mdl.unique()) ^ set(mid)

    # ---------- aggregates per model x dataset (raw; reproduces paper Tables 6-8) ----------
    agg = {}
    for (mdl, ds), d in raw.groupby(["mdl", "ds"]):
        d = d[np.isfinite(d.E)]
        agg.setdefault(mdl, {})[ds] = {
            "n": int(len(d)),
            "acc": r4(d.correct.mean()),
            "E": r4(d.E.mean()),                 # J per response
            "jpt": r4(d.jpt.mean()),             # J per token (mean of per-response ratios, as in paper)
            "jptSd": r4(d.jpt.std()),
            "tok": r4(d.tok.mean()),
            "lat": r4(d.latency_s.mean()),
            "tps": r4(d.tokens_per_second.mean()),
            "prefill": r4(d.prompt_eval_duration_s.mean()),
            "gen": r4(d.eval_duration_s.mean()),
            "load": r4(d.load_duration_s.mean()),
            "r": r4(d.tok.corr(d.E)),            # Fig 4
            "box": {"jpt": quartiles(d.jpt), "E": quartiles(d.E), "lat": quartiles(d.latency_s),
                    "tps": quartiles(d.tokens_per_second)},
        }
    for mdl in agg:
        vals = [agg[mdl][ds] for ds in DATASETS]
        agg[mdl]["all"] = {k: r4(np.mean([v[k] for v in vals])) for k in
                           ["acc", "E", "jpt", "tok", "lat", "tps", "prefill", "gen", "load", "r"]}
        agg[mdl]["all"]["n"] = int(sum(v["n"] for v in vals))

    # ---------- Pareto frontiers (energy per response vs accuracy) ----------
    fronts = {}
    for ds in DATASETS + ["all"]:
        fronts[ds] = pareto([(m, agg[m][ds]["E"], agg[m][ds]["acc"]) for m in agg])

    # ---------- per-inference compact arrays (for distributions & scatter) ----------
    points = {}
    for (mdl, ds), d in raw.groupby(["mdl", "ds"]):
        d = d[np.isfinite(d.E)]
        points.setdefault(mdl, {})[ds] = {
            "E": [int(round(v * 10)) for v in d.E],                 # 0.1 J
            "tok": [int(v) for v in d.tok],
            "lat": [int(round(v * 10)) for v in d.latency_s],       # 0.1 s
            "pre": [int(round(v * 10)) for v in d.prompt_eval_duration_s],
            "ok": "".join("1" if v else "0" for v in d.correct),
        }

    # ---------- curated inspector questions ----------
    rng = np.random.default_rng(SEED)
    js_index = load_joulescope_index()
    questions = []
    for ds in DATASETS:
        d = raw[raw.ds == ds]
        by_prompt = d.groupby("formatted_prompt")
        stats = by_prompt.agg(n=("mdl", "nunique"), acc=("correct", "mean"), minrow=("row", "min"),
                              plen=("formatted_prompt", lambda s: len(s.iloc[0])))
        ok = stats[(stats.n == 28) & (stats.minrow > 0) & (stats.plen < 1100)]
        mixed = ok[(ok.acc >= 0.2) & (ok.acc <= 0.8)]
        rest = ok.drop(mixed.index)
        n_mixed = min(len(mixed), int(PER_DATASET * 0.75))
        pick = list(rng.choice(mixed.index, n_mixed, replace=False))
        pick += list(rng.choice(rest.index, min(len(rest), PER_DATASET - n_mixed), replace=False))
        for prompt in pick:
            rows = by_prompt.get_group(prompt)
            answers = []
            for _, r in rows.iterrows():
                tr = trace_for(r, js_index)
                answers.append({
                    "m": mid[r.mdl],
                    "resp": str(r.response) if isinstance(r.response, str) else "",
                    "ok": bool(r.correct),
                    "E": r4(r.E), "tok": int(r.tok), "ptok": int(r.prompt_eval_count), "lat": r4(r.latency_s),
                    "load": r4(r.load_duration_s), "pre": r4(r.prompt_eval_duration_s), "gen": r4(r.eval_duration_s),
                    "tr": None if tr is None else {k: tr[k] for k in ("t0", "dt", "p")},
                    "_check": None if tr is None else tr["check"],
                })
            answers.sort(key=lambda a: a["m"])
            questions.append({"ds": ds, "prompt": prompt, "answers": answers})

    # sanity-check traces against logged energy
    diffs = [(a["_check"] - a["E"]) for q in questions for a in q["answers"]
             if a["_check"] is not None and a["E"] is not None]
    missing = sum(1 for q in questions for a in q["answers"] if a["tr"] is None)
    print(f"traces: {sum(len(q['answers']) for q in questions) - missing} ok, {missing} missing; "
          f"median |trace-logged| = {np.median(np.abs(diffs)):.2f} J, p95 = {np.percentile(np.abs(diffs), 95):.2f} J")
    for q in questions:
        for a in q["answers"]:
            a.pop("_check", None)

    data = {
        "meta": {"idleW": IDLE_W, "datasets": DATASETS, "families": FAMILIES, "familyLabel": FAMILY_LABEL,
                 "params": PARAMS, "nInferences": int(np.isfinite(raw.E).sum())},
        "models": models,
        "agg": agg,
        "fronts": fronts,
        "points": points,
        "questions": questions,
        "paper": paper_tables(),
    }
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
    print(f"wrote {OUT}: {os.path.getsize(OUT) / 1e6:.2f} MB; {len(questions)} questions")
    sizes = {k: len(json.dumps(v, separators=(',', ':'))) / 1e6 for k, v in data.items()}
    print({k: round(v, 2) for k, v in sizes.items()})


def paper_tables():
    """Published numbers quoted directly on the page (Husom et al. 2025)."""
    return {
        # Table 3: idle power of RPi 4 over 104 minutes
        "idle": {"mean": 2.85, "std": 0.17, "min": 2.81, "max": 5.63},
        # Table 4: mean energy per token per base model (J/token)
        "t4": {"qwen2.5_0.5b": [2.61, 1.49], "qwen2.5_1.5b": [7.57, 4.54], "llama3.2_1b": [8.40, 5.36],
               "gemma2_2b": [9.35, 3.30]},
        # Table 5: mean energy per response (J), RPi 4
        "t5": {"fp16": {"llama3.2_1b": 159.42, "qwen2.5_0.5b": 91.59, "avg": 125.51},
               "q8_0": {"llama3.2_1b": 75.85, "qwen2.5_0.5b": 42.34, "avg": 59.10},
               "q4": {"llama3.2_1b": 83.95, "qwen2.5_0.5b": 49.75, "qwen2.5_1.5b": 107.00, "avg": 80.23},
               "q3": {"gemma2_2b": 255.79, "llama3.2_1b": 101.02, "qwen2.5_0.5b": 51.65, "qwen2.5_1.5b": 113.72, "avg": 130.54},
               "avg": {"gemma2_2b": 255.79, "llama3.2_1b": 105.06, "qwen2.5_0.5b": 58.83, "qwen2.5_1.5b": 110.36}},
        # Table 9: mean inference latency (s)
        "t9": {"fp16": {"llama3.2_1b": 59.66, "qwen2.5_0.5b": 31.40, "avg": 45.53},
               "q8_0": {"llama3.2_1b": 26.32, "qwen2.5_0.5b": 9.84, "avg": 18.08},
               "q4": {"llama3.2_1b": 27.97, "qwen2.5_0.5b": 14.09, "qwen2.5_1.5b": 29.42, "avg": 23.83},
               "q3": {"gemma2_2b": 86.43, "llama3.2_1b": 31.39, "qwen2.5_0.5b": 16.19, "qwen2.5_1.5b": 38.28, "avg": 43.07},
               "avg": {"gemma2_2b": 86.43, "llama3.2_1b": 36.33, "qwen2.5_0.5b": 17.88, "qwen2.5_1.5b": 33.85}},
        # Table 10: mean latency by dataset (s)
        "t10": {"bigbenchhard": 33.19, "commonsenseqa": 13.96, "gsm8k": 22.52, "humaneval": 94.84, "truthfulqa": 21.07},
        # Table 11: mean energy per response (J), RPi 5 (8 GB)
        "t11": {"fp16": {"llama3.2_1b": 141.34, "qwen2.5_0.5b": 69.66, "avg": 105.50},
                "q8_0": {"llama3.2_1b": 63.42, "qwen2.5_0.5b": 30.70, "avg": 47.06},
                "q4": {"llama3.2_1b": 70.17, "qwen2.5_0.5b": 60.97, "qwen2.5_1.5b": 220.15, "avg": 117.10},
                "q3": {"gemma2_2b": 202.87, "llama3.2_1b": 102.29, "qwen2.5_0.5b": 53.89, "qwen2.5_1.5b": 267.23, "avg": 156.57},
                "avg": {"gemma2_2b": 202.87, "llama3.2_1b": 94.30, "qwen2.5_0.5b": 53.80, "qwen2.5_1.5b": 243.69}},
    }


if __name__ == "__main__":
    main()
