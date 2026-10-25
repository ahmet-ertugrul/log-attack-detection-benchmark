"""Figure 1: architecture of the proposed learning-based detection module (drawn at 17 cm print width).
Output: figures/figure1_architecture.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif"]
fig, ax = plt.subplots(figsize=(6.7, 4.9), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 66)
ax.axis("off")

EDGE = "#333333"
FILL_EXIST = "#F2F2F2"   # existing components
FILL_ML = "#D6E4F0"      # proposed module
FILL_LEGAL = "#FBF3E4"   # signed archive (existing)
FS = 8.5


def box(x, y, w, h, text, fill, size=FS, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", linewidth=0.8,
                                edgecolor=EDGE, facecolor=fill))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(x1, y1, x2, y2, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, linewidth=0.8,
                                 color=EDGE, linestyle="--" if dashed else "-"))


def path_arrow(points, dashed=False):
    ls = "--" if dashed else "-"
    for (x1, y1), (x2, y2) in zip(points[:-2], points[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=EDGE, linewidth=0.8, linestyle=ls)
    arrow(*points[-2], *points[-1], dashed=dashed)


ax.text(11.5, 66, "OPNsense-based platform\n(existing log sources)", ha="center", va="top", fontsize=9, fontweight="bold")
ax.text(50, 66, "Proposed learning-based\ndetection module", ha="center", va="top", fontsize=9, fontweight="bold")
ax.text(88, 66, "Decision and response", ha="center", va="top", fontsize=9, fontweight="bold")

sources = [("Packet filter log\n(pass/block)", 47), ("Suricata EVE\n(IPS signatures)", 37),
           ("Flow records\n(NetFlow)", 27), ("nDPI application\nlabels", 17)]
for text, y in sources:
    box(2, y, 19, 7, text, FILL_EXIST)

box(2, 1.5, 19, 9, "Log signing and\ntimestamping\n(Law No. 5651)", FILL_LEGAL)
path_arrow([(1.6, 50.5), (0.2, 50.5), (0.2, 6), (1.4, 6)], dashed=True)
ax.text(11.5, 13.3, "original filter logs", fontsize=7.5, ha="center", va="center", style="italic")

box(29, 42, 18, 9, "Read-only\ncollector\n(log copies)", FILL_ML)
box(29, 27, 18, 9, "Parsing and\nnormalization", FILL_ML)
box(29, 12, 18, 10, "Feature extraction:\nper connection,\nper source-minute", FILL_ML)
box(53, 27, 18, 10, "ML scoring:\nclassifier +\nanomaly detector", FILL_ML)
box(53, 12, 18, 9, "Explanation\n(feature\nimportance)", FILL_ML)
for _, y in sources:
    arrow(21.8, y + 3.5, 28.2, 46.5)
arrow(38, 42, 38, 36.8)
arrow(38, 27, 38, 22.8)
arrow(47.8, 17, 52.2, 30)
arrow(62, 27, 62, 21.8)

box(78, 42, 20, 9, "Threshold-based\ncorrelation\n(existing)", FILL_EXIST)
box(78, 27, 20, 9, "Combined\nrisk score", FILL_ML, bold=True)
box(78, 12, 20, 10, "Automated response\n(block table, expiry,\nallowlist)", FILL_EXIST)
box(78, 1.5, 20, 7.5, "Analyst dashboard\n/ SIEM", FILL_EXIST)
path_arrow([(21.8, 39), (25, 39), (25, 57), (88, 57), (88, 51.8)])
ax.text(56, 57.8, "signature alerts", fontsize=7.5, ha="center", style="italic")
arrow(71.8, 32, 77.2, 31.5)
arrow(88, 42, 88, 36.8)
arrow(88, 27, 88, 22.8)
arrow(88, 12, 88, 9.8)
arrow(71.8, 15, 77.2, 5.5, dashed=True)

fig.savefig(OUT / "figure1_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
print("saved", OUT / "figure1_architecture.png")
