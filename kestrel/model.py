import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .features import CAT, NUM, FEATURES

RETURN_COST = 1150.0   # policy s4
CALL_COST = 45.0       # policy s4
CALL_EFFECT = 0.35     # policy s7 (pilot; see caveats)
HOLD_CANCEL = 0.12     # policy s7
CALL_BREAKEVEN = CALL_COST / (CALL_EFFECT * RETURN_COST)   # = 0.1118


def make_model():
    pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), CAT), ("n", StandardScaler(), NUM)])
    return make_pipeline(pre, LogisticRegression(C=0.3, max_iter=3000))


def fit(X, y):
    m = make_model().fit(X[FEATURES], y)
    z = m[0].transform(X[FEATURES])
    z = z.toarray() if hasattr(z, "toarray") else z
    m._train_mean = np.asarray(z.mean(axis=0)).ravel()
    return m


def score(model, X):
    return model.predict_proba(X[FEATURES])[:, 1]


PRETTY = {
    "payment_mode": {"cod": "Cash on delivery", "prepaid_upi": "Prepaid by UPI", "prepaid_card": "Prepaid by card", "emi": "EMI"},
    "sales_channel": {"app": "Ordered on the app", "web": "Ordered on the website", "marketplace": "Marketplace order", "partner_outlet": "Partner-outlet order"},
}


def explain(model, row, top=4):
    """Top reasons for ONE record (1-row feature frame): coef * (x - training mean), in log-odds.
    Positive = pushes risk above a typical order."""
    pre, lr = model[0], model[-1]
    names = pre.get_feature_names_out()
    x = pre.transform(row[FEATURES])
    x = x.toarray()[0] if hasattr(x, "toarray") else x[0]
    contrib = lr.coef_[0] * (x - model._train_mean)
    out = [(n, float(c)) for n, c in zip(names, contrib) if abs(c) >= 0.03]
    out.sort(key=lambda t: -abs(t[1]))
    return out[:top]


def label(name, row):
    r = row.iloc[0]
    if name.startswith("c__payment_mode_"):
        return PRETTY["payment_mode"].get(name.split("c__payment_mode_")[1], name)
    if name.startswith("c__sales_channel_"):
        return PRETTY["sales_channel"].get(name.split("c__sales_channel_")[1], name)
    if name.startswith("c__sku_"):
        return "Product " + name.split("c__sku_")[1]
    return {
        "n__customer_prior_returns": f"{int(r.customer_prior_returns)} earlier return(s) by this customer",
        "n__customer_prior_orders": f"{int(r.customer_prior_orders)} earlier order(s) by this customer",
        "n__prior_return_rate": f"Customer has returned {r.prior_return_rate:.0%} of earlier orders",
        "n__promised_delivery_days": f"Delivery promised in {int(r.promised_delivery_days)} days",
        "n__discount_pct": f"{int(r.discount_pct)}% discount",
        "n__log_value": f"Order value Rs {np.expm1(r.log_value):,.0f}",
        "n__shield": "Kestrel Shield member (free returns)" if r.shield else "Not a Shield member",
        "n__is_gift": "Marked as a gift" if r.is_gift else "Not a gift",
        "n__no_address": "No delivery address captured" if r.no_address else "Address captured",
        "n__qty": f"Quantity {int(r.qty)}",
    }.get(name, name)


def action(p):
    """A confirmation call pays when p * 35% * Rs1150 > Rs45, i.e. p > 11.2%."""
    gain = p * CALL_EFFECT * RETURN_COST - CALL_COST
    return ("CALL BEFORE DISPATCH" if p >= CALL_BREAKEVEN else "DISPATCH NORMALLY"), round(gain, 0)
