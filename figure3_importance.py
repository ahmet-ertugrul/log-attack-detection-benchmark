"""Figure 3: permutation importance of the 12 most influential features for the random forest
(E2, first held-out reconnaissance campaign). Input: results/perm_importance_E2_part4_RF.csv (from experiments.py).
Output: figures/figure3_importance.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif"]

NAMES = {"w_dports": "Distinct destination ports per source-minute", "w_dips": "Distinct destination addresses per source-minute",
         "w_conn": "Connections per source-minute", "w_fail_frac": "Share of failed connections per source-minute",
         "duration": "Duration", "orig_bytes": "Bytes sent by originator", "resp_bytes": "Bytes sent by responder",
         "orig_pkts": "Packets sent by originator", "resp_pkts": "Packets sent by responder",
         "orig_ip_bytes": "IP bytes sent by originator", "resp_ip_bytes": "IP bytes sent by responder",
         "port_wellknown": "Destination port < 1024", "port_registered": "Destination port 1024-49151",
         "port_dynamic": "Destination port > 49151", "proto_tcp": "Protocol: TCP", "proto_udp": "Protocol: UDP",
         "svc_none": "No service identified", "local_orig": "Originator is local", "local_resp": "Responder is local"}


def nice(f):
    if f in NAMES:
        return NAMES[f]
    if f.startswith("cs_"):
        return f"Connection state: {f[3:]}"
    if f == "h_^":
        return "History flag: direction flipped (^)"
    if f.startswith("h_"):
        return f"History flag count: {f[2:]}"
    if f.startswith("port_"):
        return f"Destination port {f[5:]}"
    if f.startswith("svc_"):
        return f"Service: {f[4:]}"
    return f


pi = pd.read_csv(ROOT / "results" / "perm_importance_E2_part4_RF.csv").head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.7, 4.0), dpi=300)
ax.barh([nice(f) for f in pi.feature], pi.importance_mean, xerr=pi.importance_std, color="#2e75b6",
        error_kw={"elinewidth": 0.6, "capsize": 2})
ax.set_xlabel("Mean decrease in F1 score when the feature is permuted", fontsize=9.5)
ax.tick_params(axis="both", labelsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linewidth=0.3, color="#cccccc")
fig.tight_layout()
fig.savefig(FIG / "figure3_importance.png", dpi=300, bbox_inches="tight", facecolor="white")
print("saved", FIG / "figure3_importance.png")
