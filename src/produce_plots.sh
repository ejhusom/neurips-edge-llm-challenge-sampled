#!/bin/bash
 python3 src/analysis.py ../llm-edge-experiments-data/main/llm_responses_with_energy_consumption/*.csv
 python3 src/analysis.py ../llm-edge-experiments-data/main/llm_responses_with_energy_consumption/*.csv  --dataset bigbenchhard
 python3 src/analysis.py ../llm-edge-experiments-data/main/llm_responses_with_energy_consumption/*.csv  --dataset commonsenseqa
 python3 src/analysis.py ../llm-edge-experiments-data/main/llm_responses_with_energy_consumption/*.csv  --dataset gsm8k
 python3 src/analysis.py ../llm-edge-experiments-data/main/llm_responses_with_energy_consumption/*.csv  --dataset humaneval
 python3 src/analysis.py ../llm-edge-experiments-data/main/llm_responses_with_energy_consumption/*.csv  --dataset truthfulqa
