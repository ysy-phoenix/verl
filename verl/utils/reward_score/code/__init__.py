import re
from typing import Tuple, Optional
import time
import json
import time

import numpy as np

from .code_exec import code_exec


def validate_response_structure(processed_str: str) -> bool:
    pattern = re.compile(r"<think>.*</think>.*<answer>.*</answer>$", re.DOTALL)
    return bool(pattern.match(processed_str.strip()))


# https://github.com/Unakar/Logic-RL/blob/main/verl/utils/reward_score/kk.py
def try_extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if matches:
        final_answer = matches[-1].group(1).strip()
        return final_answer

    return solution_str


CODE_PATTERN = re.compile(r"```(?:\w+)?\n(.*?)\n```", re.DOTALL)


def extract_code_from_string(solution_str: str) -> str:
    solution_str = try_extract_solution(solution_str)
    code_blocks = CODE_PATTERN.findall(solution_str)
    return "\n".join(code_blocks).strip()


def _compute_score(
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    format_reward: float = 0.1,
    answer_reward: float = 1.0,
) -> Tuple[float, str]:
    reward_log = []

    # ground_truth is not code, but tests
    pass_fmt = validate_response_structure(solution_str)
    solution_code = extract_code_from_string(solution_str)

    if (
        not pass_fmt or len(solution_code) == 0
    ):  # only print full output when there is an error
        reward_log.append("-" * 16 + "Bad format detected!" + "-" * 16)
        reward_log.append("-" * 16 + "Original Model Output" + "-" * 16)
        reward_log.append(solution_str)
        return -answer_reward - format_reward, "\n".join(reward_log)

    reward_log.append("-" * 16 + "Extracted Code to Execute" + "-" * 16)
    ground_truth = json.loads(ground_truth)

    t_start = time.time()

    # log code
    if "functional" in ground_truth:
        reward_log.append(solution_code + "\n" + ground_truth["functional"])
    else:
        reward_log.append(solution_code)

    if "functional" in ground_truth:
        result = code_exec(solution_code + "\n" + ground_truth["functional"])
        if result["status"] != "accepted":
            reward_log.append(
                "!" * 16
                + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s"
                + "!" * 16
            )
            reward_log.append(result["error_message"])
            reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
            reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))
            return format_reward, "\n".join(reward_log)
    elif "inputs" in ground_truth and "outputs" in ground_truth:
        stdin_list: str = ground_truth["inputs"]
        stdout_list: str = ground_truth["outputs"]
        result = code_exec(solution_code, inputs=stdin_list, outputs=stdout_list)
        if result["status"] != "accepted":
            reward_log.append(
                "!" * 16
                + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s"
                + "!" * 16
            )
            reward_log.append(result["error_message"])
            reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
            reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))
            return format_reward, "\n".join(reward_log)
    else:
        raise ValueError(
            f"Current supports for ground-truth are ['functional', 'inputs/outputs'] -- No idea what's: {ground_truth = }"
        )

    return format_reward + answer_reward, "\n".join(reward_log)


def compute_score(
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    format_reward: float = 0.1,
    answer_reward: float = 1.0,
):
    if isinstance(extra_info, np.ndarray):
        extra_info = extra_info.item()
    score, reward_log = _compute_score(
        solution_str,
        ground_truth,
        extra_info=extra_info,
        format_reward=format_reward,
        answer_reward=answer_reward,
    )
    marker = "✅" if score == (format_reward + answer_reward) else "❌"
    reward_log = (
        marker * 16
        + "Reward Calculation"
        + marker * 16
        + "\n"
        + reward_log
        + "\n"
        + marker * 16
        + f"Final Rward = {score}"
        + marker * 16
    )
    print(reward_log + "\n\n")
    return score
