"""Business cost for delivery-delay decisions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def expected_cost(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    c_int: float = 2,
    c_late: float = 15,
) -> float:
    """Total expected cost of an intervention policy.

    cost = n_interventions * c_int + n_false_negatives * c_late

    - ``n_interventions`` = TP + FP = number of predicted positives.
    - ``n_false_negatives`` = FN = late orders missed by the policy.
    - TN cost 0.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, "
            f"got {y_true_arr.shape} vs {y_pred_arr.shape}"
        )

    n_interventions = int(np.sum(y_pred_arr == 1))
    n_false_negatives = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))

    return float(n_interventions * c_int + n_false_negatives * c_late)
