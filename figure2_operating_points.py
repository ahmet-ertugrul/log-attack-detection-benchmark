"""Figure 2: ROC curves of the random forest in the low-FPR region and operating points of all detectors (E2).

Uses the same splits as calibrated_thresholds.py. In each panel the curve and all random forest markers come
from one model, fitted without the calibration set. Output: figures/figure2_operating_points.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import experiments as d

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif"]

z22 = d.add_window_features(d.load_z22())
z22["y"] = (z22.tactic != "none").astype(int)
feats = d.build_features(z22)
cols = list(feats.columns)
ben = z22[z22.y == 0]
cut = ben.ts.quantile(0.7)
ben_tr, ben_te = ben[ben.ts <= cut], ben[ben.ts > cut]
panels = [("z22_part2", "(a) Discovery (unseen tactic)"), ("z22_part4", "(b) Reconnaissance 1"),
          ("z22_part5", "(c) Reconnaissance 2")]

fig, axes = plt.subplots(1, 3, figsize=(6.7, 3.5), dpi=300, sharey=True)
for ax, (held, title) in zip(axes, panels):
    att_tr = z22[(z22.y == 1) & (z22.campaign != held)]
    att_te = z22[(z22.y == 1) & (z22.campaign == held)]
    tr_df = pd.concat([d.sample(ben_tr, 150_000), d.sample(att_tr, 60_000)])
    te_df = pd.concat([d.sample(ben_te, 60_000), d.sample(att_te, 30_000)])
    tb = tr_df[tr_df.y == 0].sort_values("ts")
    cal_idx = tb.index[int(len(tb) * 0.8):]
    fit_df = tr_df.drop(cal_idx)
    rf = d.models()["RandomForest"].fit(feats.loc[fit_df.index, cols], fit_df.y.values)
    sc = rf.predict_proba(feats.loc[te_df.index, cols])[:, 1]
    yte = te_df.y.values
    fpr, tpr, _ = roc_curve(yte, sc)
    ax.plot(np.clip(fpr, 1e-5, None), tpr, color="#1f4e79", linewidth=1.2, label="RF (ROC curve)")
    cal = rf.predict_proba(feats.loc[cal_idx, cols])[:, 1]
    for t, mk in ((0.001, "v"), (0.01, "^")):
        thr = np.quantile(cal, 1 - t)
        p = sc > thr
        ax.scatter(max(p[yte == 0].mean(), 1e-5), p[yte == 1].mean(), marker=mk, s=28, color="#1f4e79", zorder=4,
                   label=f"RF, calibrated for {t:.1%} FPR")
    p = sc >= 0.5
    ax.scatter(max(p[yte == 0].mean(), 1e-5), p[yte == 1].mean(), marker="o", s=28, facecolors="white",
               edgecolors="#1f4e79", zorder=4, label="RF, default threshold (0.5)")
    ben_fit = feats.loc[fit_df.index[fit_df.y == 0], cols]
    iso = d.IsolationForest(n_estimators=200, random_state=d.SEED, n_jobs=-1).fit(ben_fit)
    thr_iso = np.quantile(-iso.score_samples(ben_fit), 0.99)
    pi = -iso.score_samples(feats.loc[te_df.index, cols]) > thr_iso
    ax.scatter(pi[yte == 0].mean(), pi[yte == 1].mean(), marker="D", s=26, color="#c55a11", zorder=4,
               label="Isolation Forest")
    pr = d.rule_baseline(te_df).astype(bool)
    ax.scatter(pr[yte == 0].mean(), pr[yte == 1].mean(), marker="X", s=34, color="#595959", zorder=4,
               label="Threshold rule")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1e-1)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("False-positive rate", fontsize=10)
    ax.tick_params(labelsize=9.5)
    ax.grid(linewidth=0.3, color="#dddddd")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Recall", fontsize=10)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.16, 1, 1))
fig.savefig(FIG / "figure2_operating_points.png", dpi=300, bbox_inches="tight", facecolor="white")
print("saved", FIG / "figure2_operating_points.png")
