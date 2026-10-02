"""Features available AT DISPATCH. Deliberately excludes last_service_event_type and
pickup_scheduled_at (they are written AFTER a return is approved: leakage)."""
import numpy as np
import pandas as pd

CAT = ["sales_channel", "payment_mode", "sku"]
NUM = ["discount_pct", "qty", "log_value", "promised_delivery_days", "customer_prior_orders",
       "customer_prior_returns", "prior_return_rate", "no_address", "is_gift", "shield"]
FEATURES = CAT + NUM
LEAKY = ["last_service_event_type", "pickup_scheduled_at"]


def fix_order_value(value, discount_pct, qty, list_price):
    """Oct-2025 orders were stored x100 by the new gateway. Detect by comparing with
    list_price*qty*(1-discount); a ratio above 50 means paise."""
    expected = list_price * qty * (1 - discount_pct / 100.0)
    ratio = value / np.where(expected > 0, expected, np.nan)
    return np.where(ratio > 50, value / 100.0, value)


def build(df, products, customers=None):
    """df: raw order rows (train/test/API). Returns feature frame aligned to df.index."""
    p = products.set_index("sku")
    d = df.copy()
    lp = d["sku"].map(p["list_price_inr"])
    value = fix_order_value(d["order_value_inr"].astype(float), d["discount_pct"].astype(float),
                            d["qty"].astype(float), lp.fillna(d["order_value_inr"]).astype(float))
    shield = None
    if customers is not None and "customer_id" in d:
        shield = d["customer_id"].map(customers.set_index("customer_id")["shield_member"])
    if "shield_member" in d:  # an explicit value on the record wins
        shield = d["shield_member"] if shield is None else d["shield_member"].fillna(shield)
    if shield is None:
        shield = pd.Series("N", index=d.index)
    shield = shield.fillna("N")
    f = pd.DataFrame(index=d.index)
    f["sales_channel"] = d["sales_channel"].astype(str)
    f["payment_mode"] = d["payment_mode"].astype(str)
    f["sku"] = d["sku"].astype(str)
    f["discount_pct"] = d["discount_pct"].astype(float)
    f["qty"] = d["qty"].astype(float)
    f["log_value"] = np.log1p(value)
    f["promised_delivery_days"] = d["promised_delivery_days"].astype(float)
    f["customer_prior_orders"] = d["customer_prior_orders"].astype(float)
    f["customer_prior_returns"] = d["customer_prior_returns"].astype(float)
    f["prior_return_rate"] = f["customer_prior_returns"] / f["customer_prior_orders"].clip(lower=1)
    f["no_address"] = (pd.to_numeric(d["delivery_pincode"], errors="coerce").fillna(0).astype(int) == 0).astype(float)
    f["is_gift"] = (d["is_gift"].astype(str) == "Y").astype(float)
    f["shield"] = (shield.astype(str) == "Y").astype(float)
    return f[FEATURES]
