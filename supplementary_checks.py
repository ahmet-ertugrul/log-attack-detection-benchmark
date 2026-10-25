"""Supplementary checks reported in the chapter (same sampling and seed as experiments.py).

1. Whether rows sharing a Zeek uid are identical in every field (exact duplicates).
2. Start time of each attack campaign in UWF-ZeekData22.
3. Training and test counts for E1, E2 and E3.
4. Share of E1 test attacks whose source-minute window also occurs in the training set.
Output: results/supplementary_checks.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import experiments as d

res = {}

# 1. Exact-duplicate check across all raw files
parts = []
for i in range(1, 6):
    p = pd.read_csv(d.DATA / "UWF-ZeekData22" / f"part{i}.csv", low_memory=False)
    p["file"] = i
    parts.append(p)
raw = pd.concat(parts, ignore_index=True)
content_cols = [c for c in raw.columns if c != "file"]
dup_uid = raw[raw.duplicated("uid", keep=False)]
res["duplicate_uid_groups"] = int(dup_uid.uid.nunique())
res["duplicate_uid_groups_with_differing_fields"] = int(
    dup_uid.groupby("uid")[content_cols].nunique(dropna=False).gt(1).any(axis=1).sum())
del raw, parts, dup_uid

z22 = d.add_window_features(d.load_z22())
z22["y"] = (z22.tactic != "none").astype(int)

# 2. Campaign start times
res["attack_campaign_start"] = {c: str(pd.to_datetime(g.ts.min(), unit="s"))
                                for c, g in z22[z22.y == 1].groupby("campaign")}

# 3 and 4. E1 counts and window sharing between training and test
s = pd.concat([d.sample(z22[z22.y == 0], 200_000), d.sample(z22[z22.y == 1], 60_000)])
tr, te = train_test_split(s.index, test_size=0.3, stratify=s.y, random_state=d.SEED)
res["E1_counts"] = {"train_benign": int((z22.loc[tr, "y"] == 0).sum()), "train_attack": int(z22.loc[tr, "y"].sum()),
                    "test_benign": int((z22.loc[te, "y"] == 0).sum()), "test_attack": int(z22.loc[te, "y"].sum())}
key = z22["src_ip"].astype(str) + "|" + (z22["ts"] // 60).astype(int).astype(str)
tr_keys = set(key.loc[tr])
att_te = [i for i in te if z22.at[i, "y"] == 1]
res["E1_test_attack_share_with_window_in_train"] = float(np.mean([key.at[i] in tr_keys for i in att_te]))

# 3. E2 counts
ben = z22[z22.y == 0]
cut = ben.ts.quantile(0.7)
ben_tr, ben_te = ben[ben.ts <= cut], ben[ben.ts > cut]
counts = z22[z22.y == 1].campaign.value_counts()
res["E2_counts"] = {}
for held in sorted(counts[counts > 100].index):
    att_tr = z22[(z22.y == 1) & (z22.campaign != held)]
    att_te = z22[(z22.y == 1) & (z22.campaign == held)]
    tr_df = pd.concat([d.sample(ben_tr, 150_000), d.sample(att_tr, 60_000)])
    te_df = pd.concat([d.sample(ben_te, 60_000), d.sample(att_te, 30_000)])
    res["E2_counts"][held] = {"train_benign": int((tr_df.y == 0).sum()), "train_attack": int(tr_df.y.sum()),
                              "test_benign": int((te_df.y == 0).sum()), "test_attack": int(te_df.y.sum())}

# 3. E3 counts
z24 = d.load_z24(["Benign", "Reconnaissance"])
z24["y"] = (z24.tactic != "none").astype(int)
tr_df = pd.concat([d.sample(z22[z22.y == 0], 200_000), d.sample(z22[z22.y == 1], 60_000)])
res["E3_counts"] = {"train_benign": int((tr_df.y == 0).sum()), "train_attack": int(tr_df.y.sum()),
                    "test_benign": int((z24.y == 0).sum()), "test_attack": int(z24.y.sum())}

with open(d.OUT / "supplementary_checks.json", "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
print(json.dumps(res, indent=2))
