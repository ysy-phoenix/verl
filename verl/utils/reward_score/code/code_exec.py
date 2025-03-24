import requests
from typing import Any
from verl.utils.reward_score.code.utils import (
    BASE_URL,
    DEFAULT_TIME_LIMIT,
    DEFAULT_MEMORY_LIMIT,
    EMPTY_TEST_CASES,
)


def code_exec(
    code: str,
    inputs: list[Any] | None = None,
    outputs: list[Any] | None = None,
    time_limit: int = DEFAULT_TIME_LIMIT,  # seconds
    memory_limit: int = DEFAULT_MEMORY_LIMIT,  # MB
) -> dict[str, Any]:
    mode = "fullcode" if inputs is None else "acm"
    test_cases = (
        EMPTY_TEST_CASES
        if inputs is None
        else [{"input": inp, "expected": out} for inp, out in zip(inputs, outputs)]
    )
    submission = {
        "code": code,
        "language": "python",  # TODO: support other languages
        "mode": mode,
        "test_cases": test_cases,
        "time_limit": time_limit,
        "memory_limit": memory_limit,
    }
    response = requests.post(f"{BASE_URL}/judge", json=submission)
    return response.json()
