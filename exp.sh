# Done
#
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_gemma2_2b_instruct_q3_K_L.jsonl --instruction --dataset_type BIG-Bench-Hard --model gemma2:2b-instruct-q3_K_L 
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_gemma2_2b_instruct_q3_K_M.jsonl --instruction --dataset_type BIG-Bench-Hard --model gemma2:2b-instruct-q3_K_M 
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_gemma2_2b_instruct_q3_K_S.jsonl --instruction --dataset_type BIG-Bench-Hard --model gemma2:2b-instruct-q3_K_S 
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_qwen25_05b_instruct_q3_K_L.jsonl --instruction --dataset_type BIG-Bench-Hard --model qwen2.5:0.5b-instruct-q3_K_L
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_qwen25_05b_instruct_q3_K_M.jsonl --instruction --dataset_type BIG-Bench-Hard --model qwen2.5:0.5b-instruct-q3_K_M
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_qwen25_05b_instruct_q3_K_S.jsonl --instruction --dataset_type BIG-Bench-Hard --model qwen2.5:0.5b-instruct-q3_K_S
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_qwen25_05b_instruct_q4_0.jsonl --instruction --dataset_type BIG-Bench-Hard --model qwen2.5:0.5b-instruct-q4_0
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_qwen25_05b_instruct_q4_1.jsonl --instruction --dataset_type BIG-Bench-Hard --model qwen2.5:0.5b-instruct-q4_1
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_qwen25_05b_instruct_fp16.jsonl --instruction --dataset_type BIG-Bench-Hard --model qwen2.5:0.5b-instruct-fp16
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_qwen25_15b_instruct_q3_K_L.jsonl --instruction --dataset_type BIG-Bench-Hard --model qwen2.5:1.5b-instruct-q3_K_L
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_qwen25_15b_instruct_q3_K_M.jsonl --instruction --dataset_type BIG-Bench-Hard --model qwen2.5:1.5b-instruct-q3_K_M
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_qwen25_15b_instruct_q3_K_S.jsonl --instruction --dataset_type BIG-Bench-Hard --model qwen2.5:1.5b-instruct-q3_K_S
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_qwen25_15b_instruct_q4_0.jsonl --instruction --dataset_type BIG-Bench-Hard --model qwen2.5:1.5b-instruct-q4_0
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_qwen25_15b_instruct_q4_1.jsonl --instruction --dataset_type BIG-Bench-Hard --model qwen2.5:1.5b-instruct-q4_1
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_llama32_1b_instruct_q3_K_L.jsonl --instruction --dataset_type BIG-Bench-Hard --model llama3.2:1b-instruct-q3_K_L
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_llama32_1b_instruct_q3_K_M.jsonl --instruction --dataset_type BIG-Bench-Hard --model llama3.2:1b-instruct-q3_K_M
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_llama32_1b_instruct_q3_K_S.jsonl --instruction --dataset_type BIG-Bench-Hard --model llama3.2:1b-instruct-q3_K_S
#
# Need more RAM:
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_gemma2_2b_instruct_q4_K_M.jsonl --instruction --dataset_type BIG-Bench-Hard --model gemma2:2b-instruct-q4_K_M 
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_gemma2_2b_instruct_q4_K_S.jsonl --instruction --dataset_type BIG-Bench-Hard --model gemma2:2b-instruct-q4_K_S 
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_gemma2_2b_instruct_q8_0.jsonl --instruction --dataset_type BIG-Bench-Hard --model gemma2:2b-instruct-q8_0 
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_qwen25_15b_instruct_fp16.jsonl --instruction --dataset_type BIG-Bench-Hard --model qwen2.5:1.5b-instruct-fp16
# python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_llama32_1b_instruct_fp16.jsonl --instruction --dataset_type BIG-Bench-Hard --model llama3.2:1b-instruct-fp16



# Llama3.2:1b
python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_llama32_1b_instruct_q4_0.jsonl --instruction --dataset_type BIG-Bench-Hard --model llama3.2:1b-instruct-q4_0
python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_llama32_1b_instruct_q4_1.jsonl --instruction --dataset_type BIG-Bench-Hard --model llama3.2:1b-instruct-q4_1
python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_llama32_1b_instruct_q4_K_M.jsonl --instruction --dataset_type BIG-Bench-Hard --model llama3.2:1b-instruct-q4_K_M
python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_llama32_1b_instruct_q4_K_S.jsonl --instruction --dataset_type BIG-Bench-Hard --model llama3.2:1b-instruct-q4_K_S
python3 src/inference.py --input BIG-Bench-Hard/collated_bbh_200_samples.jsonl --output bigbenchhard_llama32_1b_instruct_q8_0.jsonl --instruction --dataset_type BIG-Bench-Hard --model llama3.2:1b-instruct-q8_0
