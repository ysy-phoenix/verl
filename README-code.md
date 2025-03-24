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
git clone https://github.com/newfacade/LeetCodeDataset.git
cd LeetCodeDataset && git checkout f3519e6
mkdir -p $HOME/data/LeetCodeDataset
for split in test rl sft; do
    cp data/LeetCodeDataset-v2-${split}-problems.jsonl $HOME/data/LeetCodeDataset/
done
python examples/data_preprocess/code/coder1.py --root_dir $HOME/data/ --hdfs_dir $HOME/data/
# train.parquet 12287
# test.parquet 712
```
