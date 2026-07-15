"""Utils for model evaluation

Weighted Interval Score (used by FluSight)

RMSE
"""

import numpy as np

# FluSight standard levels
# use as reference
FLUSIGHT_QUANTILES = np.array([
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99,
])
FLUSIGHT_MEDIAN_IDX = 11
FLUSIGHT_LOWER = FLUSIGHT_QUANTILES[FLUSIGHT_QUANTILES < 0.5]
FLUSIGHT_ALPHAS = 2 * FLUSIGHT_LOWER
FLUSIGHT_K = len(FLUSIGHT_ALPHAS)
FLUSIGHT_INTERVAL_INDICES = [
    (
        int(np.where(np.isclose(FLUSIGHT_QUANTILES, alpha / 2))[0][0]),
        int(np.where(np.isclose(FLUSIGHT_QUANTILES, 1 - alpha / 2))[0][0]),
    )
    for alpha in FLUSIGHT_ALPHAS
]


def calc_flusight_quantiles(y, y_pred) -> np.ndarray:
    # make sure y and y_pred are log scaled
    y_raw = np.expm1(y)
    y_pred_raw = np.expm1(y_pred)
    residuals_raw = y_raw - y_pred_raw
    return np.quantile(residuals_raw, FLUSIGHT_QUANTILES)


def calc_weighted_internal_score(y, quantile_preds) -> np.ndarray:
    median_pred = quantile_preds[:, FLUSIGHT_MEDIAN_IDX]
    N = y.shape[0]
    scores = np.zeros(N)

    for alpha, (lower_idx, upper_idx) in zip(FLUSIGHT_ALPHAS, FLUSIGHT_INTERVAL_INDICES):
        lower = quantile_preds[:, lower_idx]
        upper = quantile_preds[:, upper_idx]

        sharpness = upper - lower
        under_penalty = (2 / alpha) * np.maximum(lower - y, 0)
        over_penalty = (2 / alpha) * np.maximum(y - upper, 0)

        interval_score = sharpness + under_penalty + over_penalty
        interval_score_sum += (alpha / 2) * interval_score

    wis = (1 / (_K + 0.5)) * (0.5 * np.abs(y - median_pred) + interval_score_sum)
    return wis