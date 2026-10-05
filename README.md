# Kestrel returns-risk service

Scores an order **at dispatch** with the probability it will be returned, says whether a
Rs 45 confirmation call pays for itself, and shows the reasons. No paid API, no API key, no network.

## Data Setup & Outputs
Before running the training script, please ensure the following files are placed in the `data/` directory at the root of the project:
- `customers.csv`
- `products.csv`
- `train.csv`
- `test_unlabelled.csv`
- `sample_submission.csv`

When the training script is run, it will generate the predictive model and its outputs in the following locations:
- `out/predictions.csv` (Predictions for the unlabelled test data)
- `out/validation.json` (Validation metrics and policy simulation results)
- `model/model.pkl` (The serialized trained model)

## Run (clean machine, Python 3.10+)
```
pip install -r requirements.txt
python train.py --data data --out out      # ~10 s: validation, model/model.pkl, out/predictions.csv, out/validation.json
python app/server.py                       # open http://localhost:8000
python tests/test_all.py
```
Without a trained model the page still opens and says to run `train.py` (no crash).

## Endpoint
`POST /predict` with one JSON order, e.g.
```
{"sales_channel":"marketplace","payment_mode":"cod","sku":"KH-RH-01","discount_pct":25,"qty":1,
 "order_value_inr":1499,"promised_delivery_days":9,"customer_prior_orders":4,"customer_prior_returns":2,
 "is_gift":"Y","delivery_pincode":0,"customer_id":"KC104433"}
```
returns `return_probability`, `action` (CALL BEFORE DISPATCH / DISPATCH NORMALLY), `expected_net_gain_rs_if_called`, `reasons[]`, `notes[]`.
Bad input gets HTTP 400 with a plain-English message. `last_service_event_type` and `pickup_scheduled_at` are ignored if sent (see DECISIONS).

## Files
- `kestrel/features.py` features known at dispatch; fixes the October x100 order values
- `kestrel/model.py` logistic regression, reasons, call/hold economics from the ops policy
- `train.py` de-dupes, rolling-origin validation, rupee policy simulation, final fit, `predictions.csv`
- `app/server.py`, `app/index.html` stdlib server + one-screen UI

Probabilities for extreme inputs (e.g. >90% or <1%) are extrapolation, not measurement; the model was validated on the middle of the range.
