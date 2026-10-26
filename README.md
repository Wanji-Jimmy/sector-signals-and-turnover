# Sector Signals and Turnover

Can holding positions longer improve a short-term reversal strategy? This study compares two signals across eleven US sector ETFs, with either fresh weekly selections or a fixed rule that retains some existing holdings.

**Trading less saved fees, but did not improve the 2026 result.** Across 41 complete weeks, the retention rule cut turnover by 24-27%, while both signals had worse net P&L. All four variants lost money in the earlier 2022-2025 period.

![2026 cumulative net P&L: both buffered policies finish below their fresh-selection counterparts](results/extension_pnl.png)

Solid lines use fresh selections; dashed lines use the retention rule. The raw signal saved $842 in fees but lost $11,307 in gross P&L relative to fresh selection.

The model targets $100,000 gross exposure ($50,000 long and $50,000 short), not account capital. Net P&L deducts 5 bp per dollar traded and 1% annual short borrow. Adjusted opens are price proxies; spread, impact and borrow availability are unobserved.

The retention rule was designed after seeing 2019-2025 results. Its protocol preceded the 2026 download, but this is historical research, not a live test or evidence of a persistent edge.

**View the results:** download the repository using **Code > Download ZIP**, unzip it, and open [results/report.html](results/report.html) in a browser. GitHub displays its source. The saved figures and weekly ETF table need no market data.

**Run the tests:** from the repository root, install dependencies and run 29 synthetic checks:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

**Recalculate:** `python study/report.py --extension` requires the excluded original price snapshots. See the [data instructions](docs/STUDY.md#view-or-run-locally), [method](docs/STUDY.md#the-four-way-comparison) and [code reading order](docs/STUDY.md#reading-the-code).
