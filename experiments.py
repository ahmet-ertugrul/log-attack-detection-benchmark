"""Log-based attack detection experiments on UWF-ZeekData22 / UWF-ZeekData24.

E0: naive stratified random split of the raw (non-deduplicated) CSV files
E1: stratified random split after removing duplicate connections (optimistic upper bound)
E2: leave-one-campaign-out on UWF-ZeekData22 (unseen attack campaign; benign traffic split by time)
E3: train on UWF-ZeekData22, test on UWF-ZeekData24 (different network, two years later)
E4: supplementary random split within UWF-ZeekData24 (benign and attack records come from different months)

IP addresses, timestamps, uid and community_id are never used as model inputs.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

CONN_STATES = ["S0", "S1", "SF", "REJ", "S2", "S3", "RSTO", "RSTR", "RSTOS0", "RSTRH", "SH", "SHR", "OTH"]
HISTORY_FLAGS = list("ShADadFfRrTtWw^")
COMMON_PORTS = [21, 22, 23, 25, 53, 80, 88, 110, 135, 139, 143, 389, 443, 445, 3389, 8080]
FAIL_STATES = {"REJ", "S0", "RSTO", "RSTOS0", "RSTR", "SH", "OTH"}
# Failed TCP attempts for the rule baseline: rejected or unanswered TCP connections.
# Unanswered UDP (S0/OTH) is routine in benign traffic and is therefore excluded from the rule.
FAIL_STATES_TCP = {"REJ", "S0", "RSTOS0", "RSTO", "SH"}
NUMERIC = ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts",
           "orig_ip_bytes", "resp_ip_bytes", "missed_bytes"]
SERVICES = ["dns", "http", "ssl", "ssh", "smb", "dce_rpc", "krb", "ntp", "ldap", "ftp", "dhcp"]
WINDOW = ["w_conn", "w_dports", "w_dips", "w_fail_frac"]

# Rule baseline thresholds: heuristic values chosen by the authors before running the experiments.
RULE_DPORTS = 15
RULE_DIPS = 15
RULE_MIN_CONN = 20
RULE_FAIL_FRAC = 0.5


# ---------------------------------------------------------------- data loading

def load_z22(dedupe=True):
    cols = ["ts", "uid", "src_ip", "dest_ip", "dest_port", "protocol", "service", "duration",
            "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes",
            "missed_bytes", "conn_state", "history", "local_orig", "local_resp", "mitre_attack_tactics"]
    parts = []
    for i in range(1, 6):
        d = pd.read_csv(DATA / "UWF-ZeekData22" / f"part{i}.csv", usecols=cols, low_memory=False)
        d["campaign"] = f"z22_part{i}"
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df = df.rename(columns={"protocol": "proto", "mitre_attack_tactics": "tactic"})
    if dedupe:
        stats = []
        for camp, g in df.groupby("campaign"):
            att = g[g.tactic.fillna("none") != "none"]
            ben = g[g.tactic.fillna("none") == "none"]
            stats.append({"file": camp, "rows": len(g), "unique_uid": g.uid.nunique(),
                          "attack_rows": len(att), "attack_unique": att.uid.nunique(),
                          "benign_rows": len(ben), "benign_unique": ben.uid.nunique()})
        pd.DataFrame(stats).to_csv(OUT / "z22_duplicate_stats.csv", index=False)
        before = len(df)
        df = df.drop_duplicates("uid", keep="first")
        print(f"[z22] {before} rows, {len(df)} after removing duplicate uids")
    df["tactic"] = df["tactic"].fillna("none")
    df.loc[df.tactic == "none", "campaign"] = "benign"
    return df


def load_z24(tactics):
    rename = {"src_ip_zeek": "src_ip", "dest_ip_zeek": "dest_ip", "dest_port_zeek": "dest_port",
              "src_port_zeek": "src_port", "label_tactic": "tactic"}
    parts = []
    for t in tactics:
        d = pd.read_csv(DATA / "UWF-ZeekData24" / f"{t}.csv", low_memory=False).rename(columns=rename)
        d["file"] = t
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates("uid", keep="first")
    print(f"[z24] {before} rows, {len(df)} after removing duplicate uids")
    df["tactic"] = df["tactic"].fillna("none")
    # Persistence and Privilege Escalation records are copies of the Defense Evasion (T1078) records.
    df.loc[df.file.isin(["Defense_Evasion", "Persistence", "Privilege_Escalation"]), "tactic"] = "Valid Accounts (T1078)"
    return df


# ---------------------------------------------------------------- features

def add_window_features(df):
    """Per-source summaries over calendar-minute buckets. IPs are used only for grouping, never as inputs."""
    bucket = (df["ts"] // 60).astype("int64")
    fail = df["conn_state"].isin(FAIL_STATES)
    g = pd.DataFrame({"src": df["src_ip"].values, "b": bucket.values, "dport": df["dest_port"].values,
                      "dip": df["dest_ip"].values, "fail": fail.values})
    agg = g.groupby(["src", "b"]).agg(w_conn=("dport", "size"), w_dports=("dport", "nunique"),
                                      w_dips=("dip", "nunique"), w_fail_frac=("fail", "mean"))
    joined = g[["src", "b"]].merge(agg, left_on=["src", "b"], right_index=True, how="left")
    for c in WINDOW:
        df[c] = joined[c].values
    # Used only by the rule baseline (not a model input): TCP connections and share of failed TCP attempts
    tcp = (df["proto"].fillna("").str.lower() == "tcp").values
    g["tcp"] = tcp
    g["tcp_fail"] = tcp & df["conn_state"].isin(FAIL_STATES_TCP).values
    agg2 = g.groupby(["src", "b"]).agg(r_tcp_conn=("tcp", "sum"), r_tcp_fail=("tcp_fail", "sum"))
    j2 = g[["src", "b"]].merge(agg2, left_on=["src", "b"], right_index=True, how="left")
    df["r_tcp_conn"] = j2["r_tcp_conn"].values
    df["r_tcp_fail_frac"] = np.where(j2["r_tcp_conn"].values > 0,
                                     j2["r_tcp_fail"].values / np.maximum(j2["r_tcp_conn"].values, 1), 0.0)
    return df


def build_features(df):
    X = pd.DataFrame(index=df.index)
    for c in NUMERIC:
        X[c] = np.log1p(pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0))
    for s in CONN_STATES:
        X[f"cs_{s}"] = (df["conn_state"] == s).astype(np.int8)
    hist = df["history"].fillna("").astype(str)
    for ch in HISTORY_FLAGS:
        X[f"h_{ch}"] = hist.str.count(ch.replace("^", r"\^")).astype(np.int16)
    port = pd.to_numeric(df["dest_port"], errors="coerce").fillna(-1)
    X["port_wellknown"] = (port.between(0, 1023)).astype(np.int8)
    X["port_registered"] = (port.between(1024, 49151)).astype(np.int8)
    X["port_dynamic"] = (port >= 49152).astype(np.int8)
    for p in COMMON_PORTS:
        X[f"port_{p}"] = (port == p).astype(np.int8)
    proto = df["proto"].fillna("").str.lower()
    for p in ["tcp", "udp", "icmp"]:
        X[f"proto_{p}"] = (proto == p).astype(np.int8)
    svc = df["service"].fillna("-").astype(str)
    X["svc_none"] = svc.isin(["-", "", "nan"]).astype(np.int8)
    for s in SERVICES:
        X[f"svc_{s}"] = svc.str.contains(s, regex=False).astype(np.int8)
    for c in ["local_orig", "local_resp"]:
        X[c] = df[c].astype(str).str.lower().isin(["true", "t", "1"]).astype(np.int8)
    if "w_conn" in df:
        X["w_conn"] = np.log1p(df["w_conn"])
        X["w_dports"] = np.log1p(df["w_dports"])
        X["w_dips"] = np.log1p(df["w_dips"])
        X["w_fail_frac"] = df["w_fail_frac"]
    return X


def rule_baseline(df):
    """Final rule: scan threshold or failed-TCP-attempt threshold."""
    scan = (df["w_dports"] >= RULE_DPORTS) | (df["w_dips"] >= RULE_DIPS)
    fail = (df["r_tcp_conn"] >= RULE_MIN_CONN) & (df["r_tcp_fail_frac"] >= RULE_FAIL_FRAC)
    return (scan | fail).astype(int).values


def rule_baseline_naive(df):
    """First version: unanswered UDP also counts as failed (kept to show threshold sensitivity)."""
    scan = (df["w_dports"] >= RULE_DPORTS) | (df["w_dips"] >= RULE_DIPS)
    fail = (df["w_conn"] >= RULE_MIN_CONN) & (df["w_fail_frac"] >= RULE_FAIL_FRAC)
    return (scan | fail).astype(int).values


def rule_scan_only(df):
    return ((df["w_dports"] >= RULE_DPORTS) | (df["w_dips"] >= RULE_DIPS)).astype(int).values


# ---------------------------------------------------------------- models and metrics

def models():
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced_subsample",
                                               min_samples_leaf=2, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced"),
    }


def metrics(y, pred, score=None):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {"n": int(len(y)), "n_attack": int(y.sum()), "precision": precision_score(y, pred, zero_division=0),
         "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
         "fpr": fp / (fp + tn) if (fp + tn) else np.nan, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
    if score is not None and len(np.unique(y)) == 2:
        m["roc_auc"] = roc_auc_score(y, score)
        m["pr_auc"] = average_precision_score(y, score)
    else:
        m["roc_auc"] = m["pr_auc"] = np.nan
    return m


def rule_rows(name, te_df):
    y = te_df["y"].values
    return [{"experiment": name, "features": "window (rule)", "model": m, **metrics(y, f(te_df))}
            for m, f in [("RuleBaseline", rule_baseline), ("RuleBaseline_naive", rule_baseline_naive),
                         ("RuleScanOnly", rule_scan_only)]]


def evaluate(name, Xtr, ytr, Xte, yte, rule_pred=None, feature_set="conn+window"):
    rows, fitted = [], {}
    for mname, model in models().items():
        model.fit(Xtr, ytr)
        score = model.predict_proba(Xte)[:, 1]
        rows.append({"experiment": name, "features": feature_set, "model": mname,
                     **metrics(yte, (score >= 0.5).astype(int), score)})
        fitted[mname] = model
    iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Xtr[ytr == 0])
    thr = np.quantile(-iso.score_samples(Xtr[ytr == 0]), 0.99)
    iso_score = -iso.score_samples(Xte)
    rows.append({"experiment": name, "features": feature_set, "model": "IsolationForest",
                 **metrics(yte, (iso_score > thr).astype(int), iso_score)})
    fitted["IsolationForest"] = (iso, thr)
    return rows, fitted


def sample(df, n, seed=SEED):
    return df.sample(n=min(n, len(df)), random_state=seed)


# ---------------------------------------------------------------- experiments

def main():
    results = []

    # E0: random split without deduplication (the "naive" protocol common in the literature)
    raw = add_window_features(load_z22(dedupe=False))
    raw["tactic"] = raw["tactic"].fillna("none")
    raw["y"] = (raw.tactic != "none").astype(int)
    feats_raw = build_features(raw)
    s = pd.concat([sample(raw[raw.y == 0], 200_000), sample(raw[raw.y == 1], 60_000)])
    tr, te = train_test_split(s.index, test_size=0.3, stratify=s.y, random_state=SEED)
    rows, _ = evaluate("E0_raw_random", feats_raw.loc[tr], raw.loc[tr, "y"].values,
                       feats_raw.loc[te], raw.loc[te, "y"].values, None, "conn+window")
    results += rows + rule_rows("E0_raw_random", raw.loc[te])
    del raw, feats_raw

    z22 = add_window_features(load_z22())
    z22["y"] = (z22.tactic != "none").astype(int)
    print("[z22] tactics:", z22.tactic.value_counts().to_dict())
    print("[z22] campaigns:", z22.campaign.value_counts().to_dict())

    feats_all = build_features(z22)
    conn_cols = [c for c in feats_all.columns if not c.startswith("w_")]

    # E1: stratified random split
    s = pd.concat([sample(z22[z22.y == 0], 200_000), sample(z22[z22.y == 1], 60_000)])
    tr, te = train_test_split(s.index, test_size=0.3, stratify=s.y, random_state=SEED)
    for fs, cols in [("conn", conn_cols), ("conn+window", list(feats_all.columns))]:
        rows, fitted = evaluate("E1_random", feats_all.loc[tr, cols], z22.loc[tr, "y"].values,
                                feats_all.loc[te, cols], z22.loc[te, "y"].values, None, fs)
        results += rows
    results += rule_rows("E1_random", z22.loc[te])

    # E2: leave one campaign out; benign traffic split by time (earliest 70% for training)
    ben = z22[z22.y == 0]
    cut = ben.ts.quantile(0.7)
    ben_tr, ben_te = ben[ben.ts <= cut], ben[ben.ts > cut]
    counts = z22[z22.y == 1].campaign.value_counts()
    campaigns = sorted(counts[counts > 100].index)
    # Distribution of window features on benign traffic (explains rule false alarms)
    ben_w = z22[z22.y == 0][WINDOW].describe(percentiles=[0.5, 0.9, 0.99]).T
    ben_w.to_csv(OUT / "z22_benign_window_distribution.csv")
    perm_done = False
    for held in campaigns:
        att_tr = z22[(z22.y == 1) & (z22.campaign != held)]
        att_te = z22[(z22.y == 1) & (z22.campaign == held)]
        tr_df = pd.concat([sample(ben_tr, 150_000), sample(att_tr, 60_000)])
        te_df = pd.concat([sample(ben_te, 60_000), sample(att_te, 30_000)])
        for fs, cols in [("conn", conn_cols), ("conn+window", list(feats_all.columns))]:
            rows, fitted = evaluate(f"E2_holdout_{held}", feats_all.loc[tr_df.index, cols], tr_df.y.values,
                                    feats_all.loc[te_df.index, cols], te_df.y.values, None, fs)
            if fs == "conn+window":
                rows += rule_rows(f"E2_holdout_{held}", te_df)
            for r in rows:
                r["held_tactic"] = att_te.tactic.iloc[0]
            results += rows
            if not perm_done and fs == "conn+window" and held == "z22_part4":
                # Held-out reconnaissance campaign, where the supervised model detects meaningfully.
                sub = te_df.sample(n=min(20_000, len(te_df)), random_state=SEED)
                pi = permutation_importance(fitted["RandomForest"], feats_all.loc[sub.index, cols], sub.y.values,
                                            scoring="f1", n_repeats=5, random_state=SEED, n_jobs=-1)
                pd.DataFrame({"feature": cols, "importance_mean": pi.importances_mean,
                              "importance_std": pi.importances_std}).sort_values(
                    "importance_mean", ascending=False).to_csv(OUT / "perm_importance_E2_part4_RF.csv", index=False)
                perm_done = True

    # E3: train on UWF-ZeekData22, test on UWF-ZeekData24 (Reconnaissance + benign). Per-connection features only:
    # the UWF-ZeekData24 CSV files are sampled subsets, so window summaries would not be comparable.
    z24 = load_z24(["Benign", "Reconnaissance"])
    z24["y"] = (z24.tactic != "none").astype(int)
    X24 = build_features(z24)[conn_cols]
    tr_df = pd.concat([sample(z22[z22.y == 0], 200_000), sample(z22[z22.y == 1], 60_000)])
    rows, _ = evaluate("E3_cross_dataset", feats_all.loc[tr_df.index, conn_cols], tr_df.y.values,
                       X24, z24.y.values, None, "conn")
    results += rows

    # E4: per-tactic analysis within UWF-ZeekData24 (subject to the temporal separation caveat)
    tactics = ["Benign", "Reconnaissance", "Credential_Access", "Initial_Access", "Defense_Evasion",
               "Persistence", "Privilege_Escalation", "Exfiltration"]
    z24a = load_z24(tactics)
    z24a["y"] = (z24a.tactic != "none").astype(int)
    X24a = build_features(z24a)[conn_cols]
    tr, te = train_test_split(z24a.index, test_size=0.3, stratify=z24a.tactic, random_state=SEED)
    rows, fitted = evaluate("E4_z24_random", X24a.loc[tr], z24a.loc[tr, "y"].values,
                            X24a.loc[te], z24a.loc[te, "y"].values, None, "conn")
    results += rows
    per_tactic = []
    for mname, model in fitted.items():
        if mname == "IsolationForest":
            iso, thr = model
            pred = (-iso.score_samples(X24a.loc[te]) > thr).astype(int)
        else:
            pred = (model.predict_proba(X24a.loc[te])[:, 1] >= 0.5).astype(int)
        t = z24a.loc[te, "tactic"].values
        for tac in sorted(set(t)):
            mask = t == tac
            per_tactic.append({"model": mname, "tactic": tac, "n": int(mask.sum()),
                               "detected_frac": float(pred[mask].mean())})
    pd.DataFrame(per_tactic).to_csv(OUT / "E4_per_tactic.csv", index=False)

    res = pd.DataFrame(results)
    res.to_csv(OUT / "all_results.csv", index=False)
    with open(OUT / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "rule_thresholds": {"dports": RULE_DPORTS, "dips": RULE_DIPS,
                   "min_conn": RULE_MIN_CONN, "fail_frac": RULE_FAIL_FRAC},
                   "n_features_conn": len(conn_cols), "n_features_all": feats_all.shape[1],
                   "z22_counts": z22.tactic.value_counts().to_dict(),
                   "z24_counts": z24a.tactic.value_counts().to_dict()}, f, indent=2, default=str)
    pd.set_option("display.width", 220)
    print(res[["experiment", "features", "model", "precision", "recall", "f1", "fpr", "roc_auc"]].round(4).to_string())


if __name__ == "__main__":
    sys.exit(main())
