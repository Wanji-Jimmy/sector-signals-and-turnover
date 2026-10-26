# Sector Signals and Turnover

Can trading less improve a short-term reversal strategy? This Python study compares two signals across eleven US sector ETFs, then tests whether keeping positions longer saves enough in trading costs to help.

Each week, the signals rank five-day returns, either raw or adjusted for exposure to SPY estimated from 126 earlier days. Fresh selection buys the three weakest sectors and shorts the three strongest. A fixed rank buffer keeps a long at ranks 1-5 or a short at ranks 7-11, filling vacancies from the extremes. Signals use the weekly close; trading starts at the next session's adjusted-open proxy.

All four variants lost money in 2022-2025. Across 41 complete weeks in 2026, buffering cut turnover by 24-27%, but both signals had worse net P&L. The raw signal saved $842 in fees but lost $11,307 in gross P&L relative to fresh selection. Lower turnover alone did not help.

The model resets to $100,000 gross exposure ($50,000 long and $50,000 short), not account capital. Net P&L deducts 5 bp per dollar traded and 1% annual short borrow. Adjusted opens are price proxies; spread, impact and borrow availability are not observed.

The buffer was designed after seeing 2019-2025 results. Its protocol preceded the 2026 download, but this is historical research, not a live test. Neither 2026 signal's mean rank-correlation interval excludes zero.

Open [results/report.html](results/report.html) locally to view the saved figures and weekly ETF attribution. No market data is needed. [Full study and sources](docs/STUDY.md) explain the accounting and limitations.

To run the 29 synthetic tests from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The four scripts are in `study/`. Recalculation uses `python study/report.py --extension` and requires the original snapshots listed in the [data manifests](docs/STUDY.md#view-or-run-locally). Raw downloads are excluded; the loader checks their hashes. The saved results retain their original evaluation provenance. Earlier reports remain on the [archive branch](https://github.com/Wanji-Jimmy/sector-signals-and-turnover/tree/archive/before-2026-10-26-simplification).
