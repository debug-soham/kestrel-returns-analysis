"""Stdlib-only web service (no paid API, no extra dependencies).
  POST /predict  JSON order record -> score, action, reasons
  GET  /         one-screen form that calls /predict
Run: python app/server.py   (needs model/model.pkl from `python train.py`)"""
import json, os, pickle, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from kestrel.features import build, LEAKY
from kestrel.model import score, explain, label, action, CALL_BREAKEVEN

REQUIRED = ["sales_channel", "payment_mode", "sku", "discount_pct", "qty", "order_value_inr", "promised_delivery_days",
            "customer_prior_orders", "customer_prior_returns"]
OPTIONAL_DEFAULT = {"delivery_pincode": 0, "is_gift": "N"}
ALLOWED = {"sales_channel": {"app", "web", "marketplace", "partner_outlet"}, "payment_mode": {"prepaid_upi", "prepaid_card", "cod", "emi"}}


def load():
    path = os.path.join(ROOT, "model", "model.pkl")
    if not os.path.exists(path):
        return None, None, None, None
    bundle = pickle.load(open(path, "rb"))
    data = os.path.join(ROOT, "data")
    pr = pd.read_csv(os.path.join(data, "products.csv")) if os.path.exists(os.path.join(data, "products.csv")) else None
    cu = pd.read_csv(os.path.join(data, "customers.csv")) if os.path.exists(os.path.join(data, "customers.csv")) else None
    return bundle["model"], bundle["report"], pr, cu


MODEL, REPORT, PRODUCTS, CUSTOMERS = load()


def predict(rec):
    if MODEL is None:
        raise ValueError("Model not trained yet. Run: python train.py --data data")
    if PRODUCTS is None:
        raise ValueError("data/products.csv is missing (needed to check order value).")
    missing = [k for k in REQUIRED if rec.get(k) in (None, "")]
    if missing:
        raise ValueError("Missing field(s): " + ", ".join(missing))
    for k, ok in ALLOWED.items():
        if rec[k] not in ok:
            raise ValueError(f"{k} must be one of {sorted(ok)}")
    if rec["sku"] not in set(PRODUCTS.sku):
        raise ValueError("Unknown sku " + str(rec["sku"]))
    notes = [f"Ignored {k}: it is only filled after a return starts, so it cannot be used at dispatch." for k in LEAKY if rec.get(k) not in (None, "", "NONE")]
    rec = {**OPTIONAL_DEFAULT, **{k: v for k, v in rec.items() if k not in LEAKY}}
    if "customer_id" in rec and CUSTOMERS is not None and rec["customer_id"] not in set(CUSTOMERS.customer_id) and "shield_member" not in rec:
        notes.append("Customer not found; assumed not a Shield member.")
    row = pd.DataFrame([rec])
    f = build(row, PRODUCTS, CUSTOMERS)
    p = float(score(MODEL, f)[0])
    act, gain = action(p)
    reasons = [{"text": label(n, f), "effect": "raises risk" if c > 0 else "lowers risk", "strength": round(abs(c), 2)} for n, c in explain(MODEL, f)]
    return {"order_id": rec.get("order_id"), "return_probability": round(p, 4), "action": act,
            "expected_net_gain_rs_if_called": gain, "call_threshold": round(CALL_BREAKEVEN, 4),
            "reasons": reasons, "notes": notes,
            "how_to_read": "Probability a typical order like this is returned. Calling pays off above %.1f%%; holding does not (see memo)." % (CALL_BREAKEVEN * 100),
            "model_quality": {"auc_out_of_time": round(REPORT["mean_auc"], 3), "accuracy_claim": "None. Do not read this as a yes/no verdict: even at the 40% risk level most flagged orders are not returned."}}


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, open(os.path.join(ROOT, "app", "index.html"), "rb").read(), "text/html; charset=utf-8")
        elif self.path == "/health":
            self._send(200, json.dumps({"ok": MODEL is not None}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/predict":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            rec = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(rec, dict):
                raise ValueError("Send one JSON object.")
            self._send(200, json.dumps(predict(rec)))
        except (ValueError, KeyError, TypeError) as e:
            self._send(400, json.dumps({"error": str(e)}))
        except Exception as e:  # never leak a stack trace to the screen
            self._send(500, json.dumps({"error": "Could not score this record: " + type(e).__name__}))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Kestrel returns-risk service on http://localhost:{port}  (model loaded: {MODEL is not None})")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
