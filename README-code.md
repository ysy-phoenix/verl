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

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $HOME/models/Qwen2.5-7B-Instruct
bash examples/mini/run_qwen2_5-7b_math.sh
```

