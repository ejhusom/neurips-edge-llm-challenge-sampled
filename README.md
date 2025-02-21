# neurips-edge-llm-challenge-sampled

A subselection of prompts from five of the dataset of the [NeurIps 2024 Edge LLM Challenge](https://edge-llms-challenge.github.io/edge-llm-challenge.github.io/challenge).


| Dataset 	        | Dimension 	| Source                                                    |
| ----------------- | ------------- | --------------------------------------------------------- |
| CommonsenseQA 	| Knowledge 	| [Link](https://www.tau-nlp.sites.tau.ac.il/commonsenseqa) |
| BIG-Bench Hard 	| Reasoning 	| [Link](https://github.com/suzgunmirac/BIG-Bench-Hard)     |
| GSM8K 	        | Math 	        | [Link](https://github.com/openai/grade-school-math)       |
| HumanEval 	    | Programming   | [Link](https://github.com/openai/human-eval)              |
| TruthfulQA 	    | Knowledge 	| [Link](https://github.com/sylinrl/TruthfulQA)             |
<!-- | LongBench 	    | Long-Context  | [Link](https://github.com/THUDM/LongBench)                | -->
<!-- | CHID 	            | Language 	    | [Link](https://github.com/chujiezheng/ChID-Dataset)       | -->


## Experiment procedure

Devices:

- Joulescope
- Computer recording Joulescope-data (referred to as simply "computer")
- Raspberry Pi running LLM inference (referred to as RPi)

Procedure (assuming that all cables are correctly connected):

1. Start up computer.
2. Start Joulescope-UI software on the computer. This should make the Joulescope device start up, which again will give power to the RPi.
3. When RPi has booted, ensure that the Ollama server is running on the RPi, and that all the needed models are there. It is also good to verify that the models you plan to run actually fit into memory on the RPi, otherwise it will simply loop through the prompts without saving responses, which will waste time.
4. On the RPi (assuming that this repo is cloned), adjust the script `exp.sh` to run the models and datasets you plan to run continuously. This is done manually because how many and which experiments you run in one go is dependent on your available disk space on the computer.
5. Check that the clocks on the RPi and the computer are synchronized, for example by running `date +"%H:%M:%S.%N" on both.
6. The `inference.py` script has one command line argument for timeout, which decides when to skip a prompt if it takes too much time. This is to avoid that many hours are wasted on a hangup. Currently the default is 20 minutes, but consider expanding it.
7. When everything is ready on the RPi, start recording values in the Joulescope-UI on the computer:
    - Check that the sampling frequency is set to 100 kHz.
    - Start both "signal sample recording" and "statistics recording"!
8. When you are sure that the computer is recording Joulescope-data, run the script `exp.sh` (or run commands directly in the command line) to start the inference process.

Comments on how to set up `exp.sh`:

- The first round of experiments were conducted by running all models on a single dataset in one round (e.g., running all models on CommonSenseQA). When that is done:
    - Stop recording Joulescope-data.
    - Move Joulescope-data to an external location (since it takes a lot of disk space on the computer).
    - Backup/move LLM response data from the RPi.
    - Restart the RPi.
    - Adjust `exp.sh` to run all models on the next dataset (e.g., TruthfulQA).
    - Perform step 4-8 again.

Since the Joulescope data takes so much disk space, it can be difficult to run all models on HumanEval in one go, since these responses are longer and take more time.
