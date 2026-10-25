# Log-Based Attack Detection Benchmark

Code and results for the book chapter:

> Ertuğrul, A., & Taymaz, T. (2026). *Machine learning–based cyber attack detection through log analysis: Architecture and benchmark evaluation for an OPNsense-based firewall* [Book chapter, forthcoming]. Özgür Yayınları.

The repository reproduces every table and figure in the chapter that is based on data. It evaluates machine learning detectors and a threshold rule on public, labeled Zeek connection logs. The protocols move from the naive random split common in the literature to held-out attack campaigns and a different network.

## Main findings

- **Duplicates inflate naive results.** In the CSV release of UWF-ZeekData22, 506,662 attack rows correspond to only 4,057 distinct connections; every repeated row is an exact copy. Removing duplicates lowers the attack share from 24.8% to 0.37%. On the raw files, a random forest classifies the test set without error.
- **The evaluation protocol matters more than the model.** Holding out whole attack campaigns or a whole network changes the conclusions far more than switching between models.
- **Detecting an unseen tactic depends on the operating point.** With a discovery episode held out, the random forest flagged none of its connections at the default threshold or at a threshold calibrated for a 0.1% false-positive rate. At a threshold calibrated to target 1%, it detected 99.7% of them. All three thresholds were applied to the same model.
- **Thresholds do not transfer between networks.** A threshold calibrated on the first network for 1% produced 0.03% false positives on the second, but recall fell to 31%.

## Repository layout

| File | Purpose | Chapter output |
|---|---|---|
| `download_data.py` | Downloads the 13 CSV files (about 470 MB) into `data/` and verifies SHA-256 checksums | — |
| `experiments.py` | Protocols E0–E4, all detectors, rule baselines, permutation importance | Table 3 (counts), Table 4 (default-threshold rows), Figure 3 input |
| `supplementary_checks.py` | Exact-duplicate check, campaign times, train/test counts, E1 window overlap | Methods text |
| `calibrated_thresholds.py` | Same model at the default threshold and at thresholds calibrated for 0.1% and 1% FPR; OR combination | Table 4 (E2 and E3 rows) |
| `figure1_architecture.py` | Architecture diagram of the proposed module | Figure 1 |
| `figure2_operating_points.py` | ROC curves and operating points for E2 | Figure 2 |
| `figure3_importance.py` | Permutation importance plot | Figure 3 |
| `results/` | All result files produced by the scripts | — |
| `figures/` | Figures as used in the chapter (300 dpi) | — |

## Reproducing the results

Tested with Python 3.14 on Windows 11. The versions are pinned in `requirements.txt`.

```bash
python -m venv .venv
.venv\Scripts\activate            # on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python download_data.py
python experiments.py              # about 20 minutes on a laptop
python supplementary_checks.py
python calibrated_thresholds.py
python figure1_architecture.py
python figure2_operating_points.py
python figure3_importance.py
```

All sampling, splitting and model fitting use the random seed 42.

## Data

| Dataset | Content | License | Reference |
|---|---|---|---|
| UWF-ZeekData22 | Zeek connection logs, labeled with MITRE ATT&CK tactics (reconnaissance, discovery), January–February 2022 | CC BY 4.0 | Bagui et al. (2023), *Data*, 8(1), 18. https://doi.org/10.3390/data8010018 |
| UWF-ZeekData24 | Zeek connection logs, labeled with MITRE ATT&CK tactics, 2024 | CC BY 4.0 | Elam et al. (2025), *Data*, 10(5), 59. https://doi.org/10.3390/data10050059 |

The data are not redistributed here; `download_data.py` fetches them from https://datasets.uwf.edu/.

Two properties of the datasets shape the design:

- **UWF-ZeekData22:** each attack episode occupies a single hour without benign traffic and comes from one or two source addresses.
- **UWF-ZeekData24:** all benign records come from October and November 2024 and all attack records from February and March 2024.

For this reason, IP addresses, timestamps, `uid` and `community_id` are never model inputs. UWF-ZeekData24 is used only as an external test set (E3) and for a supplementary analysis (E4).

## Method summary

- **Deduplication.** Duplicates are removed by Zeek `uid`, keeping the first occurrence, except in E0.
- **Features.** Each connection has 72 per-connection features: log-scaled byte, packet and duration counts, connection state, Zeek history flag counts, destination port class, common ports, protocol, service and locality. There are also 4 per-source window features: connections, distinct destination ports, distinct destination addresses, and the share of failed connections in the same calendar minute.
- **Window timing.** Window features are computed on the full deduplicated traffic before sampling, grouping connections by their start minute. A deployed module could therefore score a connection only after its Zeek record is complete and its start minute has closed.
- **Protocols:**
  - E0: random split of the raw files.
  - E1: random split after deduplication; optimistic, since test attacks share windows with training records.
  - E2: leave one attack campaign out, with benign traffic split by time.
  - E3: train on UWF-ZeekData22, test on UWF-ZeekData24.
  - E4: supplementary random split within UWF-ZeekData24.
- **Calibrated thresholds** are chosen on benign traffic held out from fitting:
  - E2: the most recent 20% of benign training traffic.
  - E3: 50,000 benign UWF-ZeekData22 connections not used for training.
- **Rule baseline.** The thresholds are heuristic values chosen by the authors before the experiments. The failure condition was restricted to TCP after the first run: counting unanswered UDP as failed flagged 81% of benign connections. That first version is kept in the results as `RuleBaseline_naive`.

## License

- Code: MIT License (see `LICENSE`).
- Result files and figures: Creative Commons Attribution 4.0 International (CC BY 4.0), consistent with the book.

## Conflict of interest

The first author is a co-founder of a technology company that develops the OPNsense-based firewall platform discussed in the chapter. The evaluation in this repository uses only public data.

## Citation

Please cite the book chapter and this repository (see `CITATION.cff`).
