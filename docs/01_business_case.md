# 01 — Business Case: Delivery Delay Risk

## Context
Olist is a Brazilian marketplace (~100k orders, 9 relational CSVs).
A late order generates support tickets, low reviews, and lost repeat purchases.
Today operations only notices delays after they happen.

## Who Uses the Model
Olist operations / customer care team, at purchase time.

## Decision That Changes
If `P(late)` is high → light preventive intervention:

- picking/packing priority,
- more reliable carrier / warehouse choice,
- proactive message to the customer.

Without the model: either never intervene (absorb all delays) or intervene
on everyone (unsustainable cost).

## Prediction Time
Purchase moment (`order_purchase_timestamp`).
Only features known at that time may be used: amount, weight, customer state,
category, day/hour of purchase, seller history up to that moment.
Forbidden: `delivered_date`, `review`, tracking data. This avoids leakage.

## Target
Only `delivered` orders:

```text
is_late = order_delivered_customer_date > order_estimated_delivery_date
```

`1` = late, `0` = on time. Canceled / undelivered orders are excluded
from training.

## Business Metric: Total Expected Cost
Accuracy is useless with ~8% delays: always predicting "on time" gives 92%
accuracy but zero value.

We use total expected cost (see `src/olist_delivery_risk/cost.py`):

```text
cost = n_interventions * C_int + n_false_negatives * C_late
```

Where, from the confusion matrix at a given threshold:

- `n_interventions` = TP + FP = all orders predicted positive (late).
  Every predicted positive triggers an intervention and costs `C_int`.
- False negatives (FN) / `n_false_negatives` = late orders predicted on time
  (missed delays). Each one costs `C_late`. This is the worst case.
- True positives (TP) = late orders correctly flagged. They cost `C_int`
  (the intervention) but avoid `C_late`.
- False positives (FP) = on-time orders flagged unnecessarily. They cost
  only `C_int` (wasted intervention).
- True negatives (TN) = on-time orders correctly left
  alone. They cost `0` (no intervention, no delay).

Costs:

- `C_int = 2 EUR`: cost of a preventive intervention (made-up, configurable).
- `C_late = 15 EUR`: cost of an unmanaged delay (shipping refund, ticket, review).

The intervention threshold (day 8) will be the one minimizing this cost on
validation, not 0.5. Comparison baselines: never intervene
(`cost = n_late * C_late`) and intervene on everyone.

## Assumptions and Limits
Temporal split only (train=past, val/test=recent). Costs are made up,
intervention assumed 100% effective. To be recalibrated with real data.
