## Installation

```bash
conda create -n verl python==3.10 -y
conda activate verl
pip install -U pip
pip install uv
uv pip install -e .
uv pip install vllm==0.8.3
uv pip install flash-attn --no-build-isolation
uv pip install wandb IPython matplotlib gpustat hf_transfer
```

## Quick Start

### math example
```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $HOME/models/Qwen2.5-7B-Instruct
python examples/data_preprocess/gsm8k.py
python examples/data_preprocess/math_dataset.py
bash examples/mini/run_qwen2_5-7b_math.sh
```

### code-r1
```bash
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir $HOME/models/Qwen2.5-Coder-7B-Instruct
python examples/data_preprocess/code/coder1.py --root_dir $HOME/data/ # 3 minutes
# Train set: Dataset({
#     features: ['prompt', 'data_source', 'ability', 'reward_model', 'extra_info'],
#     num_rows: 12677
# })
# Test set: Dataset({
#     features: ['prompt', 'data_source', 'ability', 'reward_model', 'extra_info'],
#     num_rows: 750
# })
python recipe/r1/data_process.py --local_dir $HOME/data/r1/ --tasks livecodebench
bash examples/mini/run_coder1.sh

python scripts/model_merger.py --local_dir ~/models/code-r1-13k-leetcode2k-taco-grpo/global_step_2048/actor
python -m sglang_router.launch_server --model-path $HOME/models/code-r1-13k-leetcode2k-taco-grpo/global_step_2048/actor/huggingface/ --dp 4

# Fix evalplus maximum_memory_bytes = min(resource.getrlimit(resource.RLIMIT_STACK)[1], maximum_memory_bytes)

evalhub run --model Qwen2.5-7B-Instruct --tasks humaneval --output-dir $HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_2048/ -p temperature=0.6
evalhub run --model Qwen2.5-7B-Instruct --tasks mbpp --output-dir $HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_2048/ -p temperature=0.6
evalplus.evaluate --dataset humaneval --samples $HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_2048/humaneval.jsonl
evalplus.evaluate --dataset mbpp --samples $HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_2048/mbpp.jsonl
evalhub run --model Qwen2.5-7B-Instruct --tasks livecodebench --output-dir $HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_2048/ -p temperature=0.6
evalhub eval --tasks livecodebench --solutions $HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_2048/livecodebench.jsonl --output-dir $HOME/metrics/code-r1-13k-leetcode2k-taco-grpo/global_step_2048/

# or bash examples/eval/eval_code.sh
```
