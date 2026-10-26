# Sector Signals and Turnover

**Can trading less improve a short-term reversal strategy?**

A Python study of eleven US sector ETFs. It compares two signals and two ways to update holdings, then traces the differences to individual ETFs and trading costs. The useful finding is a failed improvement: keeping positions longer saved fees, but also kept some losing positions.

[Full method and references](docs/STUDY.md) · [Interactive report](results/report.html) · [Extension protocol](extension_protocol.json)

## Main result

Each rule targets **$100,000 gross exposure: $50,000 long and $50,000 short**, reset weekly. This is a model position size, not invested capital. Net P&L includes 5 bp per dollar traded and 1% annual short borrow. A basis point is 0.01%.

| Net P&L | 2019–2021 | 2022–2025 | Jan–Oct 2026 |
|---|---:|---:|---:|
| Raw signal, fresh holdings | −$3,807 | −$9,125 | +$9,183 |
| Raw signal, rank buffer | +$7,583 | −$7,423 | −$1,282 |
| Market-adjusted, fresh holdings | +$4,335 | −$16,596 | +$4,977 |
| Market-adjusted, rank buffer | +$4,502 | −$12,473 | +$1,527 |
| Complete weekly intervals | 155 | 208 | 41 |

In 2026, the raw-signal buffer reduced turnover by **27.3%** and saved **$842** in trading costs. But gross P&L fell **$11,307**, leaving net P&L **$10,465 worse**. About 69% of that gross shortfall came from differences in short holdings. Reducing turnover did not automatically improve the strategy.

![2026 cumulative net P&L](results/extension_pnl.png)

## The experiment

- **Signals:** rank five-day returns, either raw or adjusted for each ETF's estimated exposure to SPY. Beta uses 126 prior daily returns ending before the signal window.
- **Fresh holdings:** buy the three weakest sectors and short the three strongest. Form signals at the weekly close and trade at the following session's adjusted-open reference.
- **Rank buffer:** retain a long while it stays in ranks 1–5 and a short while it stays in ranks 7–11. Refill vacancies and reset equal dollar weights each week. The two-rank buffer is fixed.

The report explains each week's difference through gross P&L, transaction costs and borrow costs. Its interactive table shows all eleven ETFs. A concentration check also asks how much of the result comes from a few good weeks; this is arithmetic sensitivity, not an executable strategy that can skip weeks in advance.

## View and run

Download the repository and open `results/report.html` locally. Its interactive table and figures work without market data. GitHub's file view displays HTML source rather than running the report.

Tests use synthetic data; no download or API key is needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_analysis.py test_extension.py
```

**Historical recalculation requires the original price snapshots.** Raw Yahoo responses are excluded. The [original manifest](data/manifest.json) and [2026 manifest](data/extension_2026/manifest.json) record filenames, source URLs, retrieval times and checksums. Place matching files in `data/` and `data/extension_2026/`, then run `python report.py --extension`. The loader rejects different snapshots. New downloads may contain revisions and cannot be called exact reproduction; see the [data notes](docs/STUDY.md#view-or-run-locally).

## What the results support

The buffer was designed after examining 2019–2025 results. Those years are exploratory for this extension. The extension protocol preceded the new 2026 download, but this remains historical research, not prospective live trading. No parameters were changed to improve its reported results.

All four variants lost money in 2022–2025. The 2026 results span only 41 weeks; both signals' mean rank-correlation intervals include zero. Adjusted opens are price proxies, costs are assumptions, and the model omits impact, spread, borrow availability and margin constraints. It does not establish a persistent trading edge.

The core is four Python files and two test files. The [full study](docs/STUDY.md) documents the accounting, earlier experiment, source projects and limitations.

The published acquisition metadata omits first/last raw price-row previews; source-file checksums are unchanged. The original manifest hash is recorded in the export.
