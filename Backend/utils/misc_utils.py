from typing import List, Tuple, Any
import numpy as np
import math
import re


def flatten(data: List[List[Any]]) -> Tuple[List[Any], List[int]]:
    num_each = [len(x) for x in data]
    split_idxs: List[int] = list(np.cumsum(num_each)[:-1])

    data_flat = [item for sublist in data for item in sublist]

    return data_flat, split_idxs


def unflatten(data: List[Any], split_idxs: List[int]) -> List[List[Any]]:
    data_split: List[List[Any]] = []

    start_idx: int = 0
    end_idx: int
    for end_idx in split_idxs:
        data_split.append(data[start_idx:end_idx])

        start_idx = end_idx

    data_split.append(data[start_idx:])

    return data_split


def split_evenly(num_total: int, num_splits: int) -> List[int]:
    num_per: List[int] = [math.floor(num_total / num_splits) for _ in range(num_splits)]
    left_over: int = num_total % num_splits
    for idx in range(left_over):
        num_per[idx] += 1

    return num_per


def cum_min(data: List) -> List:
    data_cum_min: List = []
    prev_min = float('inf')
    for data_i in data:
        prev_min = min(prev_min, data_i)
        data_cum_min.append(prev_min)

    return data_cum_min


def remove_all_whitespace(val: str) -> str:
    pattern = re.compile(r'\s+')
    val = re.sub(pattern, '', val)

    return val
