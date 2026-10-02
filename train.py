"""python train.py --data data --out out   -> model/model.pkl, out/predictions.csv, out/validation.json"""
import argparse, json, os, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from kestrel.features import build, FEATURES
from kestrel.model import fit, score, CALL_BREAKEVEN, RETURN_COST, CALL_COST, CALL_EFFECT, HOLD_CANCEL


def load(data):
    tr = pd.read_csv(os.path.join(data, "train.csv"))
    n_raw = len(tr)
    tr = tr.sort_values("source").drop_duplicates("order_id", keep="first").copy()  # 'crm' sorts before 'partner_feed'
    tr["ts"] = pd.to_datetime(tr["order_placed_at"])
    return tr, n_raw


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="out")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs("model", exist_ok=True)
    pr = pd.read_csv(os.path.join(a.data, "products.csv")); cu = pd.read_csv(os.path.join(a.data, "customers.csv"))
    te = pd.read_csv(os.path.join(a.data, "test_unlabelled.csv"))
    tr, n_raw = load(a.data)
    X = build(tr, pr, cu); y = tr["returned"].values
    rep = {"rows_raw": n_raw, "rows_after_dedupe": len(tr), "return_rate": float(y.mean())}

    # --- rolling-origin validation: always train on the past, score the next quarter
    folds = [("2025-10-01", "2026-01-01"), ("2026-01-01", "2026-04-01"), ("2026-04-01", "2026-07-01")]
    rep["folds"] = []
    last = None
    for s, e in folds:
        m = (tr.ts < s).values; v = ((tr.ts >= s) & (tr.ts < e)).values
        mod = fit(X[m], y[m]); p = score(mod, X[v]); yv = y[v]
        k = int(len(yv) * 0.10); top = np.argsort(-p)[:k]
        rep["folds"].append({"train_until": s, "test_quarter_start": s, "n_test": int(v.sum()), "return_rate": float(yv.mean()),
                             "auc": float(roc_auc_score(yv, p)), "pr_auc": float(average_precision_score(yv, p)),
                             "precision_top10pct": float(yv[top].mean()), "lift_top10pct": float(yv[top].mean() / yv.mean()),
                             "accuracy_at_0.5": float(accuracy_score(yv, p > 0.5)), "accuracy_predict_none": float(1 - yv.mean()),
                             "best_possible_accuracy_any_threshold": float(max(accuracy_score(yv, p > t) for t in np.linspace(.2, .9, 36)))})
        last = (p, yv, tr.loc[v, "value"] if "value" in tr else None)
    rep["mean_auc"] = float(np.mean([f["auc"] for f in rep["folds"]]))
    rep["mean_pr_auc"] = float(np.mean([f["pr_auc"] for f in rep["folds"]]))
    rng = np.random.default_rng(0); p, yv, _ = last
    boots = [roc_auc_score(yv[i], p[i]) for i in (rng.integers(0, len(yv), len(yv)) for _ in range(500))]
    rep["last_fold_auc_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    # --- rupee policy on the last out-of-time quarter (p is calibrated, checked below)
    n = len(yv); ret = yv.sum()
    def call_net(mask): return float((yv[mask] * CALL_EFFECT * RETURN_COST).sum() - CALL_COST * mask.sum())
    pol = {"orders": int(n), "returns": int(ret), "cost_of_returns_rs": float(ret * RETURN_COST)}
    for name, mask in (("call_all", np.ones(n, bool)), ("call_model_ge_breakeven", p >= CALL_BREAKEVEN), ("call_top10pct", p >= np.sort(p)[-int(n * .1)])):
        pol[name] = {"calls": int(mask.sum()), "returns_in_called": int(yv[mask].sum()), "net_saving_rs": call_net(mask)}
    rep["policy_last_quarter"] = pol
    # hold: saves 1150 only when a held order that WOULD be returned is cancelled; loses the margin on held non-returners that cancel
    hold = {}
    for name, mask in (("top10pct", p >= np.sort(p)[-int(n * .1)]),):
        sav = float(yv[mask].sum() * HOLD_CANCEL * RETURN_COST); nonret = int((1 - yv[mask]).sum()); lost_orders = nonret * HOLD_CANCEL
        hold[name] = {"held": int(mask.sum()), "saved_rs": sav, "good_orders_lost": lost_orders,
                      "break_even_margin_per_lost_order_rs": float(sav / lost_orders)}
    rep["hold_last_quarter"] = hold
    d = pd.DataFrame({"p": p, "y": yv}); d["bin"] = pd.qcut(d.p, 10, duplicates="drop")
    rep["calibration"] = d.groupby("bin", observed=True).agg(mean_p=("p", "mean"), actual=("y", "mean"), n=("y", "size")).round(4).reset_index(drop=True).to_dict("records")

    # --- final model on everything, predict test
    model = fit(X, y)
    pre = model[0]; model._train_mean = np.asarray(pre.transform(X[FEATURES]).mean(axis=0)).ravel()
    pickle.dump({"model": model, "report": rep}, open("model/model.pkl", "wb"))
    Xt = build(te, pr, cu); sc = score(model, Xt)
    pd.DataFrame({"order_id": te["order_id"], "score": sc}).to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    rep["test_rows"] = len(te); rep["test_score_mean"] = float(sc.mean()); rep["test_share_ge_breakeven"] = float((sc >= CALL_BREAKEVEN).mean())
    json.dump(rep, open(os.path.join(a.out, "validation.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("calibration", "folds", "policy_last_quarter", "hold_last_quarter")}, indent=1))
    for f in rep["folds"]: print({k: round(v, 3) if isinstance(v, float) else v for k, v in f.items()})
    print(json.dumps(rep["policy_last_quarter"], indent=1)); print(json.dumps(rep["hold_last_quarter"], indent=1))


if __name__ == "__main__":
    main()
