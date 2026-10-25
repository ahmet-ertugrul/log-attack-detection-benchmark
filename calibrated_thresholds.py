"""Calibrated decision thresholds (Table 4, calibrated rows).

The threshold is chosen on benign traffic held out from model fitting, for a target false-positive rate.
E2: per held-out campaign, the most recent 20% of benign training traffic is the calibration set;
    the same model is evaluated at the default threshold (0.5) and at targets of 0.1% and 1%.
    An OR combination of the calibrated random forest, the Isolation Forest and the rule is also reported.
E3: calibration on 50,000 benign UWF-ZeekData22 connections not used for training; test on UWF-ZeekData24.
Output: results/calibrated_thresholds.csv
"""
import numpy as np
import pandas as pd

import experiments as d

z22 = d.add_window_features(d.load_z22())
z22["y"] = (z22.tactic != "none").astype(int)
feats = d.build_features(z22)
conn_cols = [c for c in feats.columns if not c.startswith("w_")]
all_cols = list(feats.columns)
rows = []


def calibrated(model_name, Xfit, yfit, Xcal, Xte, yte, targets, label, extra=None):
    m = d.models()[model_name].fit(Xfit, yfit)
    cal = m.predict_proba(Xcal)[:, 1]
    sc = m.predict_proba(Xte)[:, 1]
    # Same model at the default threshold (0.5), to separate the threshold effect from the model change
    out = [{"experiment": label, "model": model_name, "target_fpr": "default_0.5",
            **d.metrics(yte, (sc >= 0.5).astype(int), sc)}]
    for t in targets:
        thr = np.quantile(cal, 1 - t)
        pred = (sc > thr).astype(int)
        out.append({"experiment": label, "model": model_name, "target_fpr": t, **d.metrics(yte, pred, sc)})
        if extra is not None and model_name == "RandomForest":
            comb = np.maximum.reduce([pred] + extra)
            out.append({"experiment": label, "model": "OR(RF_cal, IF, rule)", "target_fpr": t, **d.metrics(yte, comb)})
    return out


# E2
ben = z22[z22.y == 0]
cut = ben.ts.quantile(0.7)
ben_tr, ben_te = ben[ben.ts <= cut], ben[ben.ts > cut]
counts = z22[z22.y == 1].campaign.value_counts()
for held in sorted(counts[counts > 100].index):
    att_tr = z22[(z22.y == 1) & (z22.campaign != held)]
    att_te = z22[(z22.y == 1) & (z22.campaign == held)]
    tr_df = pd.concat([d.sample(ben_tr, 150_000), d.sample(att_tr, 60_000)])
    te_df = pd.concat([d.sample(ben_te, 60_000), d.sample(att_te, 30_000)])
    # The most recent 20% of benign training records (by time) form the calibration set
    tb = tr_df[tr_df.y == 0].sort_values("ts")
    cal_idx = tb.index[int(len(tb) * 0.8):]
    fit_df = tr_df.drop(cal_idx)
    Xfit, yfit = feats.loc[fit_df.index, all_cols], fit_df.y.values
    Xcal, Xte, yte = feats.loc[cal_idx, all_cols], feats.loc[te_df.index, all_cols], te_df.y.values
    iso = d.IsolationForest(n_estimators=200, random_state=d.SEED, n_jobs=-1).fit(Xfit[yfit == 0])
    thr_iso = np.quantile(-iso.score_samples(Xfit[yfit == 0]), 0.99)
    extra = [(-iso.score_samples(Xte) > thr_iso).astype(int), d.rule_baseline(te_df)]
    for mn in ("RandomForest", "HistGradientBoosting"):
        rows += calibrated(mn, Xfit, yfit, Xcal, Xte, yte, (0.001, 0.01), f"E2_holdout_{held}", extra)

# E3: calibration on UWF-ZeekData22 benign records not used for training; test on UWF-ZeekData24
z24 = d.load_z24(["Benign", "Reconnaissance"])
z24["y"] = (z24.tactic != "none").astype(int)
X24 = d.build_features(z24)[conn_cols]
tr_df = pd.concat([d.sample(z22[z22.y == 0], 200_000), d.sample(z22[z22.y == 1], 60_000)])
rest = z22[(z22.y == 0) & (~z22.index.isin(tr_df.index))]
cal_idx = d.sample(rest, 50_000).index
for mn in ("RandomForest", "HistGradientBoosting"):
    rows += calibrated(mn, feats.loc[tr_df.index, conn_cols], tr_df.y.values, feats.loc[cal_idx, conn_cols],
                       X24, z24.y.values, (0.001, 0.01), "E3_cross_dataset")

df = pd.DataFrame(rows)
df.to_csv(d.OUT / "calibrated_thresholds.csv", index=False)
pd.set_option("display.width", 200)
print(df[["experiment", "model", "target_fpr", "precision", "recall", "f1", "fpr"]].round(4).to_string())
