from olist_delivery_risk.cost import expected_cost


def test_mixed_predictions():
    # y_true:  [0, 1, 1, 0]  (2 late)
    # y_pred:  [0, 1, 0, 1]  (2 interventions: 1 TP + 1 FP, 1 FN)
    # cost = 2 * 2 + 1 * 15 = 19
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 1]

    assert expected_cost(y_true, y_pred, c_int=2, c_late=15) == 19.0


def test_all_on_time_no_intervention():
    # No late orders, no interventions -> TN only -> cost 0.
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]

    assert expected_cost(y_true, y_pred, c_int=2, c_late=15) == 0.0
