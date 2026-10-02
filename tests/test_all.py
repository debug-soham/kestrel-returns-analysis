"""python tests/test_all.py   (needs data/ and a trained model for the last two checks)"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from kestrel.features import FEATURES, LEAKY, fix_order_value
from kestrel.model import CALL_BREAKEVEN


def test_no_leaky_columns():
    assert not set(FEATURES) & set(LEAKY)


def test_october_paise_fix():
    # list 5199, 10% off -> 4679.1 expected; stored x100 by the new gateway
    v = fix_order_value(np.array([467910.0, 4679.1]), np.array([10.0, 10.0]), np.array([1.0, 1.0]), np.array([5199.0, 5199.0]))
    assert abs(v[0] - 4679.1) < 1e-6 and abs(v[1] - 4679.1) < 1e-6


def test_breakeven_matches_policy():
    assert abs(CALL_BREAKEVEN - 45 / (0.35 * 1150)) < 1e-9


def test_predictions_shape():
    p, s = os.path.join(ROOT, "out", "predictions.csv"), os.path.join(ROOT, "data", "sample_submission.csv")
    if not (os.path.exists(p) and os.path.exists(s)):
        return
    a, b = pd.read_csv(p), pd.read_csv(s)
    assert list(a.columns) == list(b.columns) and len(a) == len(b) and set(a.order_id) == set(b.order_id)
    assert a.order_id.is_unique and a.score.between(0, 1).all() and a.score.nunique() > 100


def test_service_refuses_bad_input_politely():
    if not os.path.exists(os.path.join(ROOT, "model", "model.pkl")):
        return
    sys.path.insert(0, os.path.join(ROOT, "app"))
    import server
    try:
        server.predict({"sku": "KH-AF-01"}); assert False
    except ValueError as e:
        assert "Missing" in str(e)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
