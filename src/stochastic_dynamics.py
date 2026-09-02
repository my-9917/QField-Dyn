import numpy as np


def bootstrap_residuals(values, correlation, seed, length):
    innovations = values[1:] - correlation * values[:-1]
    innovations -= innovations.mean(0)
    indices = np.random.default_rng(seed).integers(
        0, len(innovations), size=length
    )
    residual = np.zeros(3, dtype=np.float64)
    residuals = []
    for index in indices:
        residual = correlation * residual + innovations[index]
        residuals.append(residual.copy())
    residuals = np.asarray(residuals)
    residuals -= residuals.mean(0)
    return residuals
