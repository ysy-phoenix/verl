# python scripts/model_merger.py --local_dir ~/models/code-r1-13k-leetcode2k-taco-grpo/global_step_256/actor
# python -m sglang_router.launch_server --model-path $HOME/models/code-r1-13k-leetcode2k-taco-grpo/global_step_256/actor/huggingface/ --dp 4

step=256
temperature=0.0
rm -rf $HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_$step/humaneval_eval_results.json
rm -rf $HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_$step/mbpp_eval_results.json
rm -rf $HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_$step/livecodebench_results.json

evalhub run --model Qwen2.5-7B-Instruct --tasks humaneval --output-dir $HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_$step/ -p temperature=$temperature
evalplus.evaluate --dataset humaneval --samples $HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_$step/humaneval.jsonl

evalhub run --model Qwen2.5-7B-Instruct --tasks mbpp --output-dir $HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_$step/ -p temperature=$temperature
evalplus.evaluate --dataset mbpp --samples $HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_$step/mbpp.jsonl

evalhub run --model Qwen2.5-7B-Instruct --tasks livecodebench --output-dir $HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_$step/ -p temperature=$temperature
evalhub eval --tasks livecodebench --solutions $HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_$step/livecodebench.jsonl --output-dir $HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_$step/