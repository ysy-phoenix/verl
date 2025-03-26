## Installation

```bash
conda create -n verl python==3.10 -y
conda activate verl
pip install -U pip
pip install uv
uv pip install -e .
uv pip install vllm==0.7.3
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
python examples/data_preprocess/code/coder1.py --root_dir $HOME/data/ --hdfs_dir $HOME/data/ # 3 minutes
# Train set: Dataset({
#     features: ['prompt', 'data_source', 'ability', 'reward_model', 'extra_info'],
#     num_rows: 12677
# })
# Test set: Dataset({
#     features: ['prompt', 'data_source', 'ability', 'reward_model', 'extra_info'],
#     num_rows: 750
# })
bash examples/mini/run_coder1.sh
```
