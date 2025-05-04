"""
Preprocess LeetCode problems (newfacade/LeetCodeDataset) to parquet format.
"""

import os
import json
import sys

from datasets import load_dataset, concatenate_datasets
from rich.rule import Rule
import rich

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.code.code_exec import code_exec
from examples.data_preprocess.code.utils import check_code_with_ast

MAX_PROMPT_LENGTH = 8192
MAX_CODE_LENGTH = 10000
N_TESTSET_PER_DATASET = 512  # per dataset
_EMPTY_RETURN_ = {
    "data_source": None,
    "prompt": None,
    "ability": None,
    "reward_model": None,
    "extra_info": None,
}

DEBUG = False


def get_solution(solutions: list[str]) -> str | None:
    for solution in solutions:
        if len(solution) > MAX_CODE_LENGTH:
            continue
        if check_code_with_ast(solution):
            return solution
    return None


def minimize_stdio(inputs, outputs, max_n_tests=64):
    stdin_list = []
    stdout_list = []
    for stdin, stdout in zip(inputs, outputs):
        if isinstance(stdin, list):
            stdin = "\n".join(stdin)
        if isinstance(stdout, list):
            stdout = "\n".join(stdout)
        if sys.getsizeof(stdin) > 4 * 1024:
            continue
        stdout.replace("\r\n", "\n")
        stdin_list.append(stdin)
        stdout_list.append(stdout)

    zipped = sorted(zip(stdin_list, stdout_list), key=lambda x: sys.getsizeof(x[0]))

    if not zipped:
        if DEBUG:
            print("No tests found!")
        return [], []

    sorted_stdin, sorted_stdout = zip(*zipped)
    return list(sorted_stdin[:max_n_tests]), list(sorted_stdout[:max_n_tests])


SYSTEM_PROMPT = """You are a helpful programming assistant. \
The user will ask you a question and you as the assistant solve it. \
The assistant first thinks how to solve the task through reasoning and then provides the user with the final answer. \
The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively."""

TEST_CODE = """
_inputs = {inputs}
_outputs = {outputs}
import math
def _deep_eq(a, b, tol=1e-5):
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b): return False
        return all(_deep_eq(x, y, tol) for x, y in zip(a, b))
    return a == b

for i, o in zip(_inputs, _outputs):
"""


# this dataset is super noisy and needs code execution to verify the tasks
def taco():
    rich.print(Rule("Loading likaixin/TACO-verified..."))
    dataset = load_dataset("likaixin/TACO-verified")["train"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            oracle = json.loads(example["input_output"])
            source = example["source"]

            # skip poorly formatted examples
            if source in ["geeksforgeeks", "leetcode"]:
                return _EMPTY_RETURN_

            # too short description
            if len("".join([c for c in example["question"] if c.isalnum()])) < 100:
                return _EMPTY_RETURN_

            # no image
            if "image" in example["question"].lower() or "\n![" in example["question"]:
                return _EMPTY_RETURN_

            prompt_pieces = [
                "Solve the programming task below in a Python markdown code block.",
                example["question"].strip(),
            ]
            if example["starter_code"].strip():
                prompt_pieces.append(
                    "Also feel free to reuse/extend the following starter code:"
                )
                prompt_pieces.append(
                    f"```python\n{example['starter_code'].strip()}\n```"
                )
            code = get_solution(example["solutions"])
            if code is None:
                return _EMPTY_RETURN_

            if "fn_name" in oracle:  # the dataset is too noisy
                fn_name = oracle["fn_name"]

                test_code = TEST_CODE.format(
                    inputs=oracle["inputs"], outputs=oracle["outputs"]
                )

                if source in ["hackerrank"]:
                    test_code += f"    assert _deep_eq({fn_name}(*i), o)\n"
                elif source == "codewars":
                    test_code += f"    assert _deep_eq({fn_name}(*i), o[0])\n"
                else:
                    raise ValueError(f"Unknown source: {source}")

                _check_test = code + "\n" + test_code

                result = code_exec(_check_test)
                if result["status"] != "accepted":
                    if DEBUG:
                        rich.print(f"[bold red]Test code failed for {source}")
                        print(_check_test)
                        print(result["error_message"])
                    return _EMPTY_RETURN_
                oracle = json.dumps({"functional": test_code})
                assert example["starter_code"].strip() != ""
            elif "inputs" in oracle and "outputs" in oracle:  # ACM mode
                stdin_list, stdout_list = minimize_stdio(
                    oracle["inputs"], oracle["outputs"]
                )
                if len(stdin_list) == 0:
                    return _EMPTY_RETURN_

                result = code_exec(code, stdin_list, stdout_list)
                if result["status"] != "accepted":
                    if DEBUG:
                        rich.print(f"[bold red]Test code failed for {source}")
                        print(code)
                        print(result["error_message"])
                    return _EMPTY_RETURN_

                oracle = json.dumps({"inputs": stdin_list, "outputs": stdout_list})
            else:
                raise ValueError(f"Unknown ground truth format: {oracle}")

            prompt = "\n".join(prompt_pieces)

            # Check prompt length, filter out if too long
            if len(prompt) > MAX_PROMPT_LENGTH:
                return _EMPTY_RETURN_

            return {
                "data_source": "code-r1",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": oracle,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "prompt": prompt,
                    "reference": code,
                    "dataset": "likaixin/TACO-verified",
                },
            }

        return process_fn

    dataset = dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        num_proc=64,
        remove_columns=dataset.column_names,
    ).filter(lambda x: x != _EMPTY_RETURN_)
    # splits = dataset.train_test_split(
    #     test_size=max(1, min(N_TESTSET_PER_DATASET, int(len(dataset) * 0.1))), seed=666
    # )
    # train_dataset = splits["train"]
    # test_dataset = splits["test"]

    # for t in test_dataset:
    #     t["extra_info"]["split"] = "test"

    return dataset, None


def leetcode2k():
    rich.print(Rule("Loading LeetCodeDataset..."))
    dataset = load_dataset(
        "newfacade/LeetCodeDataset",
        split="train",
        revision="34803eb64eab1979069ba1f80e7ea474282e28f3",
        trust_remote_code=True,
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = f"Please solve the programming task below using a self-contained code snippet in a markdown code block.\n\n{example['query'].strip()}"

            # Filter out examples with prompts that are too long
            if len(prompt) > MAX_PROMPT_LENGTH:
                return _EMPTY_RETURN_

            return {
                "data_source": "code-r1",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(
                        {
                            "functional": f"{example['test']}\n\ncheck({example['entry_point'].strip()})\n"
                        }
                    ),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "reference": example["completion"],  # C++?
                    "prompt": prompt,
                    "starter_code": example["prompt"],
                    "dataset": "LeetCodeDataset",
                },
            }

        return process_fn

    dataset = dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        num_proc=64,
        remove_columns=dataset.column_names,
    ).filter(lambda x: x != _EMPTY_RETURN_)
    # splits = dataset.train_test_split(
    #     test_size=max(1, min(N_TESTSET_PER_DATASET, int(len(dataset) * 0.1))), seed=666
    # )
    # train_dataset = splits["train"]
    # test_dataset = splits["test"]

    # for t in test_dataset:
    #     t["extra_info"]["split"] = "test"

    return dataset, None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="~/data/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    root_dir = args.root_dir
    hdfs_dir = args.hdfs_dir

    train_datasets = []
    test_datasets = []

    dataset_makes = [leetcode2k, taco]
    names = "-".join([make.__name__ for make in dataset_makes])

    for train, test in [make() for make in dataset_makes]:
        train_datasets.append(train)
        test_datasets.append(test)

    train_dataset = concatenate_datasets(train_datasets).shuffle(seed=666)
    # test_dataset = concatenate_datasets(test_datasets)

    rich.print(Rule("Saving the final dataset"))
    print("Train set:", train_dataset)
    # print("Test set:", test_dataset)

    local_dir = os.path.join(root_dir, f"code-r1")
    rich.print(f"[bold green]Saving to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    # test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=root_dir, dst=hdfs_dir)
