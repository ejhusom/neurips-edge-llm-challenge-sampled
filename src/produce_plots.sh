#!/bin/bash
 # python3 src/analysis.py ../llm-edge-experiments-data/main/llm_responses_with_energy_consumption_idle_subtracted/*.csv
 # python3 src/analysis.py ../llm-edge-experiments-data/main/llm_responses_with_energy_consumption_idle_subtracted/*.csv  --dataset bigbenchhard
 # python3 src/analysis.py ../llm-edge-experiments-data/main/llm_responses_with_energy_consumption_idle_subtracted/*.csv  --dataset commonsenseqa
 # python3 src/analysis.py ../llm-edge-experiments-data/main/llm_responses_with_energy_consumption_idle_subtracted/*.csv  --dataset gsm8k
 # python3 src/analysis.py ../llm-edge-experiments-data/main/llm_responses_with_energy_consumption_idle_subtracted/*.csv  --dataset humaneval
 # python3 src/analysis.py ../llm-edge-experiments-data/main/llm_responses_with_energy_consumption_idle_subtracted/*.csv  --dataset truthfulqa

 python3 src/analysis.py ../llm-edge-experiments-data/main/llm_responses_with_energy_consumption_idle_subtracted_with_dots/*.csv
 python3 src/analysis.py ../llm-edge-experiments-data/main/llm_responses_with_energy_consumption_idle_subtracted_with_dots/*.csv  --dataset bigbenchhard
 python3 src/analysis.py ../llm-edge-experiments-data/main/llm_responses_with_energy_consumption_idle_subtracted_with_dots/*.csv  --dataset commonsenseqa
 python3 src/analysis.py ../llm-edge-experiments-data/main/llm_responses_with_energy_consumption_idle_subtracted_with_dots/*.csv  --dataset gsm8k
 python3 src/analysis.py ../llm-edge-experiments-data/main/llm_responses_with_energy_consumption_idle_subtracted_with_dots/*.csv  --dataset humaneval
 python3 src/analysis.py ../llm-edge-experiments-data/main/llm_responses_with_energy_consumption_idle_subtracted_with_dots/*.csv  --dataset truthfulqa