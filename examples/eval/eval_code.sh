#!/bin/bash

# python scripts/model_merger.py --local_dir ~/checkpoints/code-r1-13k-leetcode2k-taco-grpo/global_step_2048/actor
# python -m sglang_router.launch_server --model-path $HOME/checkpoints/code-r1-13k-leetcode2k-taco-grpo/global_step_1536/actor/huggingface/ --dp 4

step=1536
temperature=0.0
exp_name=code-r1-13k-leetcode2k-taco-grpo
SYSTEM_PROMPT="You are a helpful programming assistant. \
The user will ask you a question and you as the assistant solve it. \
The assistant first thinks how to solve the task through reasoning \
and then provides the user with the final answer. \
The reasoning process and answer are enclosed within \
<think>...</think> and <answer>...</answer> tags, respectively."

rm -rf "$HOME/metrics/$exp_name/global_step_$step/humaneval_eval_results.json"
rm -rf "$HOME/metrics/$exp_name/global_step_$step/mbpp_eval_results.json"
rm -rf "$HOME/metrics/$exp_name/global_step_$step/livecodebench_results.json"

evalhub run --model Qwen2.5-7B-Instruct --tasks humaneval --output-dir "$HOME/metrics/$exp_name/global_step_$step/" -p temperature="$temperature" # --system-prompt "$SYSTEM_PROMPT"
evalplus.evaluate --dataset humaneval --samples "$HOME/metrics/$exp_name/global_step_$step/humaneval.jsonl"

evalhub run --model Qwen2.5-7B-Instruct --tasks mbpp --output-dir "$HOME/metrics/$exp_name/global_step_$step/" -p temperature="$temperature" # --system-prompt "$SYSTEM_PROMPT"
evalplus.evaluate --dataset mbpp --samples "$HOME/metrics/$exp_name/global_step_$step/mbpp.jsonl"

evalhub run --model Qwen2.5-7B-Instruct --tasks livecodebench --output-dir "$HOME/metrics/$exp_name/global_step_$step/" -p temperature="$temperature" # --system-prompt "$SYSTEM_PROMPT"
evalhub eval --tasks livecodebench --solutions "$HOME/metrics/$exp_name/global_step_$step/livecodebench.jsonl" --output-dir "$HOME/metrics/$exp_name/global_step_$step/"

# evalhub eval --tasks livecodebench --solutions "$HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_768/livecodebench.jsonl" --output-dir "$HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_768/"
