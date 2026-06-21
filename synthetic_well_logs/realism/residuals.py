from __future__ import annotations

import numpy as np


def high_pass(values: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    kernel_size = min(kernel_size, values.shape[-1])
    if kernel_size % 2 == 0:
        kernel_size -= 1
    kernel_size = max(kernel_size, 3)
    kernel = np.ones(kernel_size) / kernel_size
    smooth = np.vstack([np.convolve(channel, kernel, mode="same") for channel in values])
    return values - smooth


def overlap_weights(window_size: int) -> np.ndarray:
    weights = np.hanning(window_size)
    return np.maximum(weights, 1e-3)
