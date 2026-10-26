"""Recalculate the study from the original snapshots and write a local report."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import STRATEGIES, run_study
from data import ROOT, load_data


LABELS = {"raw_reversal": "Raw reversal", "market_adjusted_reversal": "Market-adjusted reversal"}
COLORS = {"raw_reversal": "#7c8998", "market_adjusted_reversal": "#176a75"}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":"))
                             if isinstance(v, (list, dict)) else v for k, v in row.items()})


def make_figures(summary, rows, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    period = summary["periods"]["evaluation"]
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    for name in STRATEGIES:
        selected = [r for r in rows if r["period"] == "evaluation" and r["strategy"] == name]
        dates = [datetime.fromisoformat(selected[0]["entry_date"])]
        dates += [datetime.fromisoformat(r["exit_date"]) for r in selected]
        pnl = np.r_[0, np.cumsum([r["net_pnl"] for r in selected])] / 1000
        ax.plot(dates, pnl, color=COLORS[name], linewidth=2, label=LABELS[name])
    ax.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax.set(ylabel="Cumulative net P&L (USD thousands)")
    ax.set_title("What happened after next-open entry?", loc="left", pad=36)
    ax.text(0, 1.02, "$100,000 fixed gross notional | 5 bp one-way cost | 1% annual borrow",
            transform=ax.transAxes, fontsize=9, color="#54616e")
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="y", alpha=0.15)
    fig.savefig(output / "pnl.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    annual = period["comparison"]["yearly"]
    years = sorted(annual)
    differences = [annual[y]["net_pnl_difference"] for y in years]
    axes[0].bar(years, differences, color=["#176a75" if x >= 0 else "#b45d4e" for x in differences])
    axes[0].set(title="Does adjustment help every year?", ylabel="Adjusted minus raw: net P&L (USD)")
    axes[0].axhline(0, color="#bbbbbb", linewidth=0.8)
    axes[0].text(0, -0.19, "Whole weekly intervals assigned to their exit year.", transform=axes[0].transAxes,
                 fontsize=9, color="#54616e")
    x = np.arange(2)
    for j, name in enumerate(STRATEGIES):
        selected = [r for r in rows if r["period"] == "evaluation" and r["strategy"] == name]
        means = [np.mean([r["gap_diagnostic_return"] for r in selected]) * 10000,
                 np.mean([r["gross_return"] for r in selected]) * 10000]
        axes[1].bar(x + (j - .5) * .32, means, .32, color=COLORS[name], label=LABELS[name])
    axes[1].set_xticks(x, ["Before entry\n(diagnostic only)", "After entry\n(holding interval)"])
    axes[1].set(title="Where does the return appear?", ylabel="Mean gross basket return (bp)")
    axes[1].axhline(0, color="#bbbbbb", linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].text(0, -0.19, "Unequal durations; pre-entry returns are excluded from P&L.",
                 transform=axes[1].transAxes, fontsize=9, color="#54616e")
    fig.savefig(output / "diagnostics.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    for name in STRATEGIES:
        costs = period["strategies"][name]["costs"]
        levels = sorted(float(c) for c in costs)
        vals = [costs[str(int(c))]["net_pnl"] / 1000 for c in levels]
        ax.plot(levels, vals, marker="o", color=COLORS[name], label=LABELS[name], linewidth=2)
    ax.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax.axvline(5, color="#8c969f", linestyle="--", linewidth=1)
    ax.set(title="How much trading cost can the signal withstand?",
           xlabel="Assumed one-way cost per dollar traded (bp)", ylabel="Evaluation net P&L (USD thousands)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.15)
    fig.savefig(output / "costs.png", dpi=160)
    plt.close(fig)


def write_html(summary, output):
    period = summary["periods"]["evaluation"]
    comparison = period["comparison"]
    low, high = comparison["bootstrap"]["annualized_mean_difference_ci95"]
    conclusion = ("The paired interval crosses zero: this sample does not establish a reliable advantage from market adjustment."
                  if low <= 0 <= high else "The paired interval excludes zero under the fixed block-bootstrap assumptions; the short sample and cost model still limit the conclusion.")
    table = []
    money = lambda value: f'{"−" if value < 0 else ""}${abs(value):,.0f}'
    for name in STRATEGIES:
        m = period["strategies"][name]["primary"]
        break_even = m["break_even_one_way_bps_after_borrow"]
        cost_label = f'{break_even:.2f} bp' if break_even >= 0 else 'None at ≥0 bp'
        table.append(f'<tr><td>{LABELS[name]}</td><td>{money(m["gross_pnl"])}</td>'
                     f'<td>{money(m["net_pnl"])}</td><td>{cost_label}</td>'
                     f'<td>${m["max_drawdown_dollars"]:,.0f}</td></tr>')
    earlier = summary["periods"]["development"]["strategies"]
    earlier_raw = money(earlier["raw_reversal"]["primary"]["net_pnl"])
    earlier_adjusted = money(earlier["market_adjusted_reversal"]["primary"]["net_pnl"])
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sector Reversal Research</title><style>
body{{margin:0;background:#f2f4f3;color:#19282e;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1050px;margin:auto;padding:52px 28px}}
h1{{font-size:42px;line-height:1.15;max-width:780px}}h2{{margin-top:36px;font-size:23px}}.eyebrow{{color:#176a75;font-weight:650;letter-spacing:.08em;font-size:12px}}
.note{{background:#e6efed;padding:20px 24px;border-left:4px solid #176a75}}.small{{font-size:13px;color:#53626a}}img{{width:100%;display:block;background:white;margin:22px 0}}
table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}td,th{{text-align:left;padding:13px;border-bottom:1px solid #dce2e1}}a{{color:#176a75}}
code{{background:#e6e9e8;padding:2px 5px}}@media(max-width:700px){{h1{{font-size:30px}}table{{font-size:12px}}td,th{{padding:7px}}}}
</style><main><div class="eyebrow">QUANTITATIVE RESEARCH · FIRST IMPLEMENTATION</div>
<h1>Does removing the market improve a sector reversal signal?</h1>
<p>Eleven US sector ETFs. Two fixed signals. Next-open entry. A small test of whether recent sector weakness predicts a rebound.</p>
<div class="note"><b>Evaluation: {comparison["weeks"]} weekly observations, 2022–2025.</b><br>{conclusion}<br>
Annualized mean net difference: <b>{comparison["annualized_mean_difference"]*100:+.2f} percentage points</b>; descriptive 95% interval [{low*100:+.2f}, {high*100:+.2f}].</div>
<h2>The comparison</h2><p>Both rules buy the three weakest sectors and short the three strongest. The raw rule ranks five-session returns.
The adjusted rule first removes each sector's estimated response to SPY. Betas use 126 earlier sessions ending before the signal week.
Signals are observed after the weekly close; returns start at the following session's open reference.</p>
<table><thead><tr><th>Signal</th><th>Gross P&amp;L</th><th>Net P&amp;L</th><th>Break-even cost</th><th>Weekly drawdown</th></tr></thead><tbody>{''.join(table)}</tbody></table>
<p class="small">USD 100,000 fixed gross notional; 50,000 long and 50,000 short at entry. Net includes assumed 5 bp one-way net turnover costs and 1% annual borrow.
Break-even cost is after borrow. These are historical proxy calculations, not live fills, compounded returns, or demonstrated alpha.</p>
<img src="pnl.png" alt="Cumulative net PnL for the two fixed sector signals">
<p>In the separate 2019–2021 period, raw reversal returned {earlier_raw} and market-adjusted reversal {earlier_adjusted} net.
The earlier ranking did not persist in the later evaluation. Both periods use the same pre-specified rules.</p>
<h2>Two checks behind the headline</h2><p>Annual differences reveal concentration in particular years. The pre-entry gap is shown separately: a completed closing-price signal cannot earn a return before the stated entry.</p>
<img src="diagnostics.png" alt="Year-by-year incremental PnL and returns before versus after entry">
<h2>Trading costs change the decision</h2><p>Unchanged holdings are netted when rebalancing. Entry, changed exposure and final liquidation all incur cost. Borrow is a scenario, not a historical quote.</p>
<img src="costs.png" alt="Net PnL under four fixed one-way cost assumptions">
<h2>What this can and cannot establish</h2><p>The question, universe and parameters were saved before this experiment's first run.
2019–2021 is reported separately from the 2022–2025 evaluation. The work is informed by prior literature and is a retrospective replication and extension.</p>
<p>Prices are Yahoo Finance historical snapshots fetched on 26 October 2026. Adjusted opens are total-return proxies, not executable dollar prices.
ETF holdings and index definitions change. Dollar-neutral entry weights do not guarantee beta neutrality. Borrow availability, market impact and exact dividend cash accounting are not observed.
The interval resamples four-week blocks; it cannot create more independent years of evidence.</p>
<p class="small">Source files: <a href="summary.json">results and provenance</a> · <a href="weekly.csv">weekly ledger</a> · <a href="../study/protocol.json">pre-run protocol</a> · <a href="../README.md">method and sources</a></p>
</main></html>'''
    (output / "report.html").write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", action="store_true", help="Run the frozen rank-buffer extension from both snapshots")
    if parser.parse_args().extension:
        return run_extension_report()
    dates, symbols, opens, closes, protocol, provenance = load_data()
    summary, weekly, signals = run_study(dates, symbols, opens, closes, protocol)
    summary["provenance"] = provenance
    summary["generated_utc"] = datetime.now(timezone.utc).isoformat()
    summary["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("study/analysis.py", "study/data.py", "study/report.py", "study/protocol.json")}
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(output / "weekly.csv", weekly)
    write_csv(output / "signals.csv", signals)
    make_figures(summary, weekly, output)
    write_html(summary, output)
    print(json.dumps(summary["periods"]["evaluation"]["comparison"], indent=2))



PERIOD_LABELS = {"earlier_seen": "2019–2021 · seen", "later_seen": "2022–2025 · seen",
                 "new_period_2026": "2026 · new historical period"}
SHORT_LABELS = {"raw_reversal": "Raw", "market_adjusted_reversal": "Market-adjusted"}


def extension_figures(summary, rows, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    colors = {"raw_reversal": "#245b78", "market_adjusted_reversal": "#bd7045"}
    fig, ax = plt.subplots(figsize=(11, 4.7), layout="constrained")
    for signal in STRATEGIES:
        for policy in ("fresh", "buffer"):
            selected = [r for r in rows if r["period"] == "new_period_2026"
                        and r["strategy"] == f"{signal}_{policy}"]
            dates = [datetime.fromisoformat(selected[0]["entry_date"])]
            dates += [datetime.fromisoformat(r["exit_date"]) for r in selected]
            pnl = np.r_[0, np.cumsum([r["net_pnl"] for r in selected])] / 1000
            ax.plot(dates, pnl, color=colors[signal], lw=2.1,
                    linestyle="-" if policy == "fresh" else "--",
                    label=f'{SHORT_LABELS[signal]} · {"fresh" if policy == "fresh" else "buffered"}')
    ax.axhline(0, color="#9aa5aa", lw=.8)
    ax.set(title="2026: where did the cost savings lead?", ylabel="Cumulative net P&L (USD thousands)")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=.15)
    fig.savefig(output / "extension_pnl.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.7), layout="constrained", sharey=True,
                             gridspec_kw={"width_ratios": [1.2, 1]})
    labels = []
    for index, (period_name, signal) in enumerate((p, s) for p in PERIOD_LABELS for s in STRATEGIES):
        period = summary["periods"][period_name]
        d, ic = period["comparisons"][signal]["decomposition"], period["diagnostics"][signal]["ic"]
        years = "2026" if period_name == "new_period_2026" else PERIOD_LABELS[period_name][:9]
        labels.append(f'{years} · {SHORT_LABELS[signal]}')
        for offset, field, color, label in ((-.15, "gross_pnl_difference", "#8598a6", "Gross change"),
                (.15, "trading_cost_saved", "#b89050", "Trading cost saved")):
            axes[0].barh(index + offset, d[field] / 1000, height=.26, color=color,
                         label=label if index == 0 else None)
        axes[0].scatter(d["net_pnl_difference"] / 1000, index, marker="D", color="#1d343e",
                        s=30, label="Net change" if index == 0 else None, zorder=3)
        low, high = ic["bootstrap"]["mean_ci95"]
        axes[1].plot([low, high], [index, index], color=colors[signal], lw=2)
        axes[1].scatter(ic["mean"], index, color=colors[signal], s=35, zorder=3)
    axes[0].set_yticks(range(6), labels)
    axes[0].invert_yaxis()
    axes[0].set_title("Buffer minus fresh: the cost trade-off", pad=34)
    axes[0].set(xlabel="Total P&L difference (USD thousands)")
    axes[1].set_title("The signal IC is unchanged by buffering", pad=34)
    axes[1].set(xlabel="Mean weekly Spearman IC and 95% interval")
    axes[0].legend(frameon=False, ncol=3, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.01))
    for ax in axes:
        ax.axvline(0, color="#9aa5aa", lw=.8)
        ax.grid(axis="x", alpha=.12)
        for y in (1.5, 3.5):
            ax.axhline(y, color="#d8dddf", lw=.8)
    fig.savefig(output / "extension_attribution_ic.png", dpi=170)
    plt.close(fig)


def explanation_html(explanation, assets):
    current = explanation["periods"]["new_period_2026"]
    money = lambda x: f'{"−" if x < 0 else "+"}${abs(x):,.0f}'
    pct = lambda x: "—" if x is None else f"{100*x:.1f}%"
    sides, concentration = [], []
    for signal in STRATEGIES:
        s = current["signals"][signal]
        sides.append(f'<tr><td>{SHORT_LABELS[signal]}</td><td>{money(s["long_gross_difference"])}</td>'
                     f'<td>{money(s["short_gross_difference"])}</td><td>{money(s["gross_difference"])}</td></tr>')
        for policy in ("fresh", "buffer"):
            m = current["strategies"][f"{signal}_{policy}"]
            concentration.append(f'<tr><td>{SHORT_LABELS[signal]} · {policy}</td><td>{money(m["net_pnl"])}</td>'
                f'<td>{pct(m["top3_share_of_positive_pnl"])}</td><td>{pct(m["top3_share_of_negative_pnl"])}</td>'
                f'<td>{money(m["net_without_top3_positive_weeks"])}</td></tr>')
    fields = ("symbol", "rank", "fresh_weight", "buffer_weight", "asset_return", "gross_difference",
              "trading_cost_saved", "borrow_cost_saved", "net_difference")
    examples = {}
    for row in assets:
        if row["period"] != "new_period_2026":
            continue
        week = examples.setdefault(row["signal"], {}).setdefault(row["entry_date"],
            {"exit": row["exit_date"], "signal_date": row["signal_date"], "assets": []})
        week["assets"].append([row[field] for field in fields])
    defaults = {signal: current["signals"][signal]["illustrative_week"]["entry_date"] for signal in STRATEGIES}
    payload = json.dumps({"weeks": examples, "defaults": defaults}, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return f'''<h2>Which side explains the difference?</h2>
<p>Both policies use the same prices and target exposure. Separating their long and short holdings shows where the gross difference arises.</p>
<div class="scroll"><table><thead><tr><th>2026 · buffer minus fresh</th><th>Long contribution</th><th>Short contribution</th><th>Total gross difference</th></tr></thead><tbody>{''.join(sides)}</tbody></table></div>
<p class="small">These are accounting contributions on the observed price path, not causal explanations of market moves. Trading and borrow costs are reconciled separately below; they are not arbitrarily assigned to the current holding side.</p>
<details><summary>Inspect any 2026 week · all eleven ETFs</summary>
<p><label>Signal <select id="explain-signal" aria-label="Attribution signal"><option value="raw_reversal">Raw reversal</option><option value="market_adjusted_reversal">Market-adjusted reversal</option></select></label>
<label>Holding week <select id="explain-week" aria-label="Attribution holding week"></select></label></p>
<p class="small">The default example has the largest absolute gross difference, selected after observing results. Every complete week is available. Positive weight means long; negative means short. Rank 1 is the weakest signal.</p>
<p id="explain-dates"></p><div class="scroll"><table><thead><tr><th>ETF</th><th>Rank</th><th>Fresh weight</th><th>Buffer weight</th><th>ETF return</th><th>Δ gross</th><th>Trade fee saved</th><th>Borrow saved</th><th>Δ net</th></tr></thead><tbody id="explain-body"></tbody><tfoot id="explain-total"></tfoot></table></div>
<p class="small">All differences are buffer minus fresh; cost savings are fresh cost minus buffer cost. Matching current weights can still have different fees because earlier holdings differ. Even an ETF that is now flat can carry a closing cost.</p></details>
<h2>How much rests on a few weeks?</h2>
<p>Each variant has its own largest winning and losing weeks. Three is a fixed reporting choice; it was not selected to produce a particular conclusion.</p>
<div class="scroll"><table><thead><tr><th>2026 variant</th><th>Net P&amp;L</th><th>Top 3 wins / all gains</th><th>Top 3 losses / all losses</th><th>Net minus top 3 wins</th></tr></thead><tbody>{''.join(concentration)}</tbody></table></div>
<p class="small">Concentration uses positive and negative weekly contributions separately, not total net profit. Subtracting three recorded weeks is an arithmetic sensitivity of the existing ledger, not a strategy that could identify or skip those weeks in advance. Actually skipping trades would change later positions and costs. Full statistics and dates for every period are in the <a href="explanation_summary.json">explanation summary</a>; every ETF contribution is in the <a href="asset_attribution.csv">asset ledger</a>. These diagnostics were added after all three periods had been observed.</p>
<script id="explain-data" type="application/json">{payload}</script>''' + '''
<script>
(() => {
  const data = JSON.parse(document.getElementById('explain-data').textContent);
  const signal = document.getElementById('explain-signal');
  const week = document.getElementById('explain-week');
  const dollars = new Intl.NumberFormat('en-US', {style:'currency',currency:'USD',maximumFractionDigits:2,signDisplay:'exceptZero'});
  const percent = value => (100*value).toFixed(2)+'%';
  function appendRow(target, values) {
    const tr = document.createElement('tr');
    values.forEach(value => { const td = document.createElement('td'); td.textContent = value; tr.appendChild(td); });
    target.appendChild(tr);
  }
  function render() {
    const record = data.weeks[signal.value][week.value];
    document.getElementById('explain-dates').textContent = 'Signal observed '+record.signal_date+'; holding '+week.value+' to '+record.exit+'.';
    const body = document.getElementById('explain-body'); body.replaceChildren();
    record.assets.forEach(r => appendRow(body,[r[0],r[1],percent(r[2]),percent(r[3]),percent(r[4]),...r.slice(5).map(v=>dollars.format(v))]));
    const totals = [5,6,7,8].map(index => record.assets.reduce((sum,r)=>sum+r[index],0));
    const footer = document.getElementById('explain-total'); footer.replaceChildren();
    appendRow(footer,['Total','—','—','—','—',...totals.map(v=>dollars.format(v))]);
  }
  function setWeeks() {
    week.replaceChildren();
    Object.keys(data.weeks[signal.value]).sort().forEach(entry => {
      const option = document.createElement('option'); option.value = entry;
      option.textContent = entry+' to '+data.weeks[signal.value][entry].exit; week.appendChild(option);
    });
    week.value = data.defaults[signal.value]; render();
  }
  signal.addEventListener('change',setWeeks); week.addEventListener('change',render); setWeeks();
})();
</script>
'''


def extension_html(summary, output, explanation, assets):
    money = lambda x: f'{"−" if x < 0 else ""}${abs(x):,.0f}'
    current = summary["periods"]["new_period_2026"]
    first = current["strategies"]["raw_reversal_fresh"]["primary"]
    table, comparisons, savings, deltas = [], [], [], []
    for signal in STRATEGIES:
        fresh = current["strategies"][f"{signal}_fresh"]["primary"]
        buffered = current["strategies"][f"{signal}_buffer"]["primary"]
        savings.append(100 * (1 - buffered["turnover_total"] / fresh["turnover_total"]))
        deltas.append(f'{SHORT_LABELS[signal].lower()} {money(buffered["net_pnl"] - fresh["net_pnl"])}')
        for policy in ("fresh", "buffer"):
            m = current["strategies"][f"{signal}_{policy}"]["primary"]
            reduction = "—" if policy == "fresh" else f'{savings[-1]:.1f}%'
            table.append(f'<tr><td>{SHORT_LABELS[signal]} · {"fresh" if policy == "fresh" else "buffered"}</td>'
                         f'<td>{money(m["gross_pnl"])}</td><td><b>{money(m["net_pnl"])}</b></td>'
                         f'<td>{money(m["trading_cost"])}</td><td>${m["turnover_dollars"]/1e6:.2f}m</td><td>{reduction}</td></tr>')
    for period_name, label in PERIOD_LABELS.items():
        block = summary["periods"][period_name]
        for signal in STRATEGIES:
            c = block["comparisons"][signal]
            d = c["decomposition"]
            low, high = c["bootstrap"]["mean_weekly_difference_ci95"]
            comparisons.append(f'<tr><td>{label}<br><span class="small">{SHORT_LABELS[signal]} · {c["weeks"]} weeks</span></td>'
                f'<td>{money(d["gross_pnl_difference"])}</td><td>{money(d["trading_cost_saved"])}</td>'
                f'<td><b>{money(d["net_pnl_difference"])}</b></td>'
                f'<td>{c["mean_weekly_difference"]*10000:+.1f} bp<br><span class="small">[{low*10000:+.1f}, {high*10000:+.1f}]</span></td></tr>')
    crosses = "; ".join(f'{SHORT_LABELS[s]} {current["comparisons"][s]["decomposition"]["crossover_one_way_bps"]:.1f} bp'
                        for s in STRATEGIES)
    later = summary["periods"]["later_seen"]
    later_net = "; ".join(f'{SHORT_LABELS[s].lower()} {money(later["strategies"][s+"_buffer"]["primary"]["net_pnl"])}'
                          for s in STRATEGIES)
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sector Signals and Turnover</title><style>
:root{{color-scheme:light}}body{{margin:0;background:#f4f2ed;color:#20343e;font:16px/1.65 system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:48px 28px 60px}}h1{{font-size:46px;letter-spacing:-.035em;line-height:1.12;max-width:820px;margin:14px 0 22px}}
h2{{font-size:25px;letter-spacing:-.02em;margin:36px 0 12px}}p{{max-width:1000px}}.eyebrow{{color:#936b35;font-size:12px;font-weight:700;letter-spacing:.11em}}
.note{{background:#e6ecec;border-left:4px solid #245b78;padding:19px 23px}}.small{{font-size:13px;color:#58666d}}.rule{{border-top:1px solid #cad1d1;border-bottom:1px solid #cad1d1;padding:14px 0}}
.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;background:#fff;font-size:14px;font-variant-numeric:tabular-nums}}td,th{{padding:12px 13px;text-align:right;border-bottom:1px solid #e0e5e5;white-space:nowrap}}td:first-child,th:first-child{{text-align:left}}th{{font-weight:650;background:#edf1f0}}img{{width:100%;height:auto;display:block;margin:24px 0;background:white}}a{{color:#245b78}}details{{margin-top:23px}}summary{{cursor:pointer;font-weight:650}}select{{font:inherit;padding:6px;margin:6px 12px 6px 4px;border:1px solid #bac7cd;background:white;color:inherit}}tfoot{{font-weight:650;background:#edf1f0}}
@media(max-width:700px){{main{{padding:30px 18px}}h1{{font-size:34px}}table{{font-size:12px}}td,th{{padding:9px}}}}
</style></head><body><main><div class="eyebrow">SECTOR SIGNALS &amp; TURNOVER · RESEARCH EXTENSION</div>
<h1>Does trading less preserve the signal?</h1>
<p>Eleven sector ETFs, two reversal signals, one fixed rank buffer. The question is whether fewer replacements save enough trading cost to offset any change in gross P&amp;L.</p>
<div class="note"><b>2026: lower turnover, lower net P&amp;L for both buffered policies.</b><br>
Turnover fell {min(savings):.1f}–{max(savings):.1f}%, while buffered minus fresh net P&amp;L was {"; ".join(deltas)}.
The improvement seen in 2022–2025 did not carry into this {first["weeks"]}-week historical period. No parameter was retuned after seeing it.</div>
<p class="small">2019–2021 and 2022–2025 are previously observed, exploratory periods. The 2026 snapshot was acquired after the extension specification; it is historical evidence, not a prospective live test.</p>
<h2>The rule, in one example</h2><p>Fresh selection buys the lowest three scores and shorts the highest three every week. The buffer keeps an existing long at ranks 1–5 and an existing short at ranks 7–11, then fills vacant slots from the extremes. Each side still holds three ETFs at ±1/6 weight.</p>
<p class="rule"><b>A long moves from rank 3 to rank 5:</b> keep it. At rank 6, replace it.<br><b>A short is at rank 7:</b> keep it. At rank 6, replace it. Score ties use alphabetical symbols.</p>
<p class="small">Raw scores use five-session returns; market-adjusted scores subtract the sector’s estimated beta times the SPY move. Betas use 126 earlier daily returns ending before the signal window. The buffer changes membership, not either signal. All four variants were specified in advance of acquiring this 2026 snapshot.</p>
<h2>2026 · {first["weeks"]} complete holding weeks</h2><p class="small">{first["first_entry"]} to {first["last_exit"]}. Prices end 23 October; the incomplete final holding week is excluded.</p>
<div class="scroll"><table><thead><tr><th>Variant</th><th>Gross P&amp;L</th><th>Net P&amp;L</th><th>Trading cost</th><th>Dollars traded</th><th>Turnover saved</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>
<p class="small">Turnover saved is relative to the same signal’s fresh policy. Net deducts 5 bp per dollar traded and 1% annual short borrow. USD 100,000 is fixed gross exposure ($50,000 long + $50,000 short), not invested cash or compounded wealth.</p>
<img src="extension_pnl.png" alt="Four cumulative net PnL curves in the 2026 historical period, with solid fresh and dashed buffered lines">
<h2>Was the saving worth the changed basket?</h2><p>Within each signal: <b>Δ net P&amp;L = Δ gross P&amp;L + trading cost saved + borrow cost saved.</b> Borrow savings are zero here because short notional and holding dates match. In 2022–2025 buffering improved both comparisons, but buffered net P&amp;L remained negative: {later_net}.</p>
<div class="scroll"><table><thead><tr><th>Period / signal</th><th>Δ gross</th><th>Cost saved</th><th>Δ net</th><th>Mean Δ / week<br><span class="small">95% interval</span></th></tr></thead><tbody>{''.join(comparisons)}</tbody></table></div>
<p class="small">All differences are buffered minus fresh. Intervals use paired four-week moving blocks, 5,000 draws and seed 260126. Each week is one observation; the small 2026 sample does not establish a persistent edge.</p>
<img src="extension_attribution_ic.png" alt="Gross changes, cost savings and net changes across all periods beside weekly mean information coefficients and confidence intervals">
<p><b>Buffering cannot improve the underlying IC.</b> Each weekly IC is the Spearman correlation of negative signal scores with the following open-to-open sector returns. Fresh and buffered policies share exactly the same scores and IC; sector observations within one week are not independent weeks.</p>
{explanation_html(explanation, assets)}
<details><summary>Accounting, uncertainty and source files</summary><p>Holdings start flat separately in each of the three periods. Every weekly entry rebalances the selected six names to the same target dollar exposure, including drift trades when membership is unchanged; final liquidation also incurs cost. Fixed gross exposure does not imply fixed risk or beta neutrality. Annual summaries, replacements, basket beta, historical volatility and rank persistence are in the linked files.</p>
<p>The ex-post 2026 crossover one-way costs are {crosses}. These solve the linear cost accounting identity; they are not recommended trading costs or optimized settings.</p>
<p>2019–2021 and 2022–2025 were already observed and are exploratory for this extension. 2026 prices were acquired after the extension specification, but remain historical data, not a prospective live test or globally unseen history. Earlier periods retain their original snapshots; the 2026 run uses its own internally consistent snapshot with prior history for signals. Price levels are not spliced.</p>
<p>Adjusted opens are total-return proxies rather than executable fills. Vendor revisions, changing ETF holdings, unavailable spreads and depth, assumed borrow, and simplified distribution and funding accounting remain limitations.</p>
<p><a href="extension_summary.json">Summary and provenance</a> · <a href="extension_weekly.csv">Weekly ledger</a> · <a href="extension_diagnostics.csv">Weekly signal diagnostics</a> · <a href="../study/extension_protocol.json">Extension protocol</a> · <a href="../README.md">Method and sources</a> · <a href="https://github.com/Wanji-Jimmy/sector-signals-and-turnover/blob/archive/before-2026-10-26-simplification/results/first_version/report.html">Archived first report</a></p></details>
</main></body></html>'''
    (output / "report.html").write_text(html)


def run_extension_report():
    from extension import attribute_portfolios, run_extension
    specification = json.loads((ROOT / "study" / "extension_protocol.json").read_text())
    expected = specification["original_protocol_sha256"]
    if hashlib.sha256((ROOT / "study" / "protocol.json").read_bytes()).hexdigest() != expected:
        raise ValueError("Original protocol differs from the extension's frozen reference")
    summary, weekly, diagnostics, provenance = None, [], [], {}
    explanation, assets = {"periods": {}}, []
    for snapshot, periods, directory, end in (
            ("original", specification["periods"][:2], ROOT / "data", None),
            ("extension_2026", specification["periods"][2:], ROOT / "data" / "extension_2026", "2026-10-23")):
        dates, symbols, opens, closes, protocol, source = load_data(directory, end=end)
        partial, rows, diag = run_extension(dates, symbols, opens, closes, protocol,
                                            {**specification, "periods": periods})
        source["available_signal_count"] = partial.pop("signal_count")
        provenance[snapshot] = source
        if summary is None:
            summary = partial
        else:
            summary["periods"].update(partial["periods"])
        weekly.extend(rows)
        diagnostics.extend(diag)
        explained, contributions = attribute_portfolios(dates, symbols, opens, closes, protocol, rows)
        explanation.update({k: v for k, v in explained.items() if k != "periods"})
        explanation["periods"].update(explained["periods"])
        assets.extend(contributions)
    summary.update(provenance=provenance, generated_utc=datetime.now(timezone.utc).isoformat(),
                   extension_protocol_sha256=hashlib.sha256((ROOT / "study" / "extension_protocol.json").read_bytes()).hexdigest(),
                   source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ("study/analysis.py", "study/data.py", "study/extension.py", "study/report.py", "study/protocol.json", "study/extension_protocol.json")})
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    (output / "extension_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(output / "extension_weekly.csv", weekly)
    write_csv(output / "extension_diagnostics.csv", diagnostics)
    extension_figures(summary, weekly, output)
    explanation["interpretation"] = "Post-result attribution and concentration of the unchanged ledger; not a new holdout, tuned strategy or tradable omitted-week simulation."
    (output / "explanation_summary.json").write_text(json.dumps(explanation, indent=2, allow_nan=False) + "\n")
    write_csv(output / "asset_attribution.csv", assets)
    extension_html(summary, output, explanation, assets)
    print(json.dumps({p: {s: b["comparisons"][s]["net_pnl_difference"] for s in STRATEGIES}
                      for p, b in summary["periods"].items()}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        raise SystemExit(f"Missing input: {error.filename}. See docs/STUDY.md for the required snapshots.") from error
