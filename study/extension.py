"""Frozen rank-buffer extension; reuses the original signals and PnL ledger.

No fetching or historical performance calculation happens at import time.
Membership state resets in each requested reporting period; annual views do not.
"""
import math

import numpy as np

from analysis import (NOTIONAL, STRATEGIES, _metrics, _validate, account_period,
                      build_signals, paired_comparison, target_weights)


POLICIES = ("fresh", "buffer")
VARIANTS = tuple(f"{signal}_{policy}" for signal in STRATEGIES for policy in POLICIES)


def buffered_weights(scores, symbols, previous=None, buffer=2):
    """Retain ranks <= 3+buffer / >= 9-buffer, then fill extreme vacancies."""
    fresh = target_weights(scores, symbols)
    if len(set(symbols)) != 11 or not isinstance(buffer, int) or not 0 <= buffer <= 2:
        raise ValueError("Eleven unique symbols and integer buffer zero to two required")
    if previous is None:
        return fresh
    previous = np.asarray(previous, float)
    if previous.shape != (11,) or not np.isfinite(previous).all():
        raise ValueError("Previous weights must be eleven finite values")
    if not (np.all(previous == 0) or
            (np.count_nonzero(previous > 0) == np.count_nonzero(previous < 0) == 3
             and np.all(np.isin(previous, [-1 / 6, 0, 1 / 6])))):
        raise ValueError("Previous membership must be flat or three equal weights per side")
    if buffer == 0:
        return fresh
    order = sorted(range(11), key=lambda j: (scores[j], symbols[j]))
    ranks = {j: rank for rank, j in enumerate(order, 1)}
    longs = {j for j in order if previous[j] > 0 and ranks[j] <= 3 + buffer}
    shorts = {j for j in order if previous[j] < 0 and ranks[j] >= 9 - buffer}
    for j in order:
        if len(longs) < 3 and j not in shorts:
            longs.add(j)
    for j in reversed(order):
        if len(shorts) < 3 and j not in longs:
            shorts.add(j)
    weights = np.zeros(11)
    weights[list(longs)], weights[list(shorts)] = 1 / 6, -1 / 6
    return weights


def _memberships(weights, symbols, previous):
    result = {}
    for side, sign in (("long", 1), ("short", -1)):
        names = {symbols[j] for j in range(11) if weights[j] * sign > 0}
        old = set() if previous is None else {
            symbols[j] for j in range(11) if previous[j] * sign > 0}
        result[f"{side}_names"] = sorted(names)
        result[f"{side}_removed"] = sorted(old - names)
        result[f"{side}_added"] = sorted(names - old)
        result[f"{side}_replacements"] = len(old - names)
    result["replacement_count"] = result["long_replacements"] + result["short_replacements"]
    result["initial_position_count"] = 6 if previous is None else 0
    return result


def _period_signals(signals, dates, start, end, symbols, buffer):
    """Only eligible complete intervals update policy state; append their exit marker."""
    eligible = [(a, b) for a, b in zip(signals, signals[1:])
                if start <= dates[a["entry_index"]] <= dates[b["entry_index"]] <= end]
    output, metadata, previous = [], {}, {}
    for source, _ in eligible:
        signal = dict(source, weights=dict(source["weights"]))
        for name, score_key in zip(STRATEGIES, ("raw_score", "adjusted_score")):
            for policy in POLICIES:
                variant = f"{name}_{policy}"
                weights = (np.asarray(source["weights"][name]) if policy == "fresh" else
                           buffered_weights(source[score_key], symbols, previous.get(variant), buffer))
                signal["weights"][variant] = weights.tolist()
                metadata[(source["entry_index"], variant)] = _memberships(
                    weights, symbols, previous.get(variant))
                previous[variant] = weights
        output.append(signal)
    if eligible:
        output.append(eligible[-1][1])
    return output, metadata


def _average_ranks(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind="stable")
    ranks, left = np.empty(len(values), float), 0
    while left < len(values):
        right = left + 1
        while right < len(values) and values[order[right]] == values[order[left]]:
            right += 1
        ranks[order[left:right]] = (left + right + 1) / 2
        left = right
    return ranks


def spearman(left, right):
    """Average ranks for ties; undefined constant cross-sections return None."""
    a, b = _average_ranks(left), _average_ranks(right)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else None


def _diagnostic_summary(rows, key):
    values = np.array([np.nan if r[key] is None else r[key] for r in rows])
    finite = np.isfinite(values)
    result = {"weeks": len(rows), "observations": int(finite.sum()),
              "mean": float(values[finite].mean()) if finite.any() else None}
    block, draws, seed = 4, 5000, 260126
    result["bootstrap"] = {"block_weeks": block, "draws": draws, "seed": seed}
    if len(values) < block or not finite.any():
        result["bootstrap"]["status"] = "insufficient_weeks_or_no_defined_observations"
    else:
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, len(values) - block + 1,
                              size=(draws, math.ceil(len(values) / block)))
        indices = (starts[:, :, None] + np.arange(block)).reshape(draws, -1)[:, :len(values)]
        sample = values[indices]
        count = np.isfinite(sample).sum(axis=1)
        means = np.nansum(sample, axis=1)[count > 0] / count[count > 0]
        result["bootstrap"].update(method="overlapping moving blocks, no wrap",
            mean_ci95=np.quantile(means, [.025, .975]).tolist(),
            interpretation="Conditional descriptive interval; weeks, not sectors, are observations.")
    return result


def _attribution(fresh, buffered):
    gross = sum(b["gross_pnl"] - f["gross_pnl"] for f, b in zip(fresh, buffered))
    trading = sum(f["trading_cost"] - b["trading_cost"] for f, b in zip(fresh, buffered))
    borrow = sum(f["borrow_cost"] - b["borrow_cost"] for f, b in zip(fresh, buffered))
    net = sum(b["net_pnl"] - f["net_pnl"] for f, b in zip(fresh, buffered))
    turnover_saved = sum(f["turnover"] - b["turnover"] for f, b in zip(fresh, buffered))
    crossover = (-gross - borrow) / (NOTIONAL * turnover_saved) * 10_000 if turnover_saved > 1e-12 else None
    return {"gross_pnl_difference": gross, "trading_cost_saved": trading,
            "borrow_cost_saved": borrow, "net_pnl_difference": net,
            "identity_residual": net - gross - trading - borrow,
            "turnover_saved": turnover_saved, "turnover_dollars_saved": NOTIONAL * turnover_saved,
            "crossover_one_way_bps": crossover,
            "crossover_interpretation": (
                "Buffered net exceeds fresh above this signed cost; ex-post accounting only."
                if crossover is not None else "Undefined: fresh turnover does not exceed buffered turnover.")}


def _comparison(fresh, buffered):
    result = paired_comparison(fresh, buffered, bootstrap=True)
    result["decomposition"] = _attribution(fresh, buffered)
    for year in result.get("yearly", {}):
        result["yearly"][year]["decomposition"] = _attribution(
            [r for r in fresh if r["year"] == year], [r for r in buffered if r["year"] == year])
    return result


def _variant_metrics(rows, cost):
    result = _metrics(rows, cost)
    result.update(replacement_count=sum(r["replacement_count"] for r in rows),
                  turnover_dollars=sum(r["turnover_dollars"] for r in rows))
    if rows:
        volatility = [r["ex_ante_basket_volatility"] for r in rows]
        result.update(ex_ante_basket_volatility_mean=float(np.mean(volatility)),
                      ex_ante_basket_volatility_min=min(volatility),
                      ex_ante_basket_volatility_max=max(volatility))
    return result


def run_extension(dates, symbols, adj_open, adj_close, original_protocol, extension_protocol):
    """Return JSON-serializable (summary, weekly_rows, diagnostic_rows).

    Supports any subset of frozen periods; snapshots can be run independently.
    Diagnostic rows occur once per signal/week because policy does not change IC.
    """
    p, e = original_protocol, extension_protocol
    opens, closes = np.asarray(adj_open, float), np.asarray(adj_close, float)
    dates, _ = _validate(dates, symbols, opens, closes, p)
    if e["holding_count_each_side"] != 3 or e["rank_buffer"] not in (0, 2):
        raise ValueError("Only frozen three-per-side design and zero-buffer control supported")
    signals = build_signals(dates, symbols, closes, p)
    columns = [symbols.index(s) for s in p["symbols"]]
    sector_open, sector_close = opens[:, columns], closes[:, columns]
    daily = np.diff(np.log(sector_close), axis=0)
    summary = {"notional": NOTIONAL, "primary_cost_bps": p["primary_cost_bps"],
        "rank_buffer": e["rank_buffer"], "signal_count": len(signals), "periods": {},
        "comparison_direction": "buffer minus fresh, separately within each signal",
        "replacement_definition": "Prior side-membership slots removed; initial entries and final liquidation excluded.",
        "year_attribution": "Whole holding interval exit year; annual views do not reset membership or holdings.",
        "annualization": "52 times mean weekly fixed-notional return, not CAGR; volatility uses square-root scaling.",
        "ex_ante_volatility": "sqrt(252) times sample SD of weighted sector daily log returns in the same 126-return beta window ending t-5.",
        "diagnostics": "One cross-section per week; IC and score persistence are shared by fresh and buffered policies.",
        "caveat": "2026 is a partial historical period with few weekly observations; intervals do not establish a persistent or prospective edge."}
    weekly_rows, diagnostic_rows = [], []
    for period in e["periods"]:
        name, start, end = period["name"], period["start"], period["end"]
        selected, metadata = _period_signals(signals, dates, start, end, p["symbols"], e["rank_buffer"])
        block = {**period, "strategies": {}, "comparisons": {}, "diagnostics": {}}
        by_variant = {}
        for variant in VARIANTS:
            signal_name, policy = variant.rsplit("_", 1)
            rows = account_period(dates, sector_open, sector_close, selected, start, end, variant,
                                  cost_bps=p["cost_bps_one_way"], primary_cost_bps=p["primary_cost_bps"],
                                  borrow_apr=p["short_borrow_apr"])
            for row, signal in zip(rows, selected):
                fit_end = signal["signal_index"] - p["signal_sessions"]
                portfolio_daily = daily[fit_end - p["beta_sessions"]:fit_end] @ np.array(row["weights"])
                row.update(period=name, signal=signal_name, policy=policy,
                    turnover_dollars=NOTIONAL * row["turnover"],
                    ex_ante_basket_volatility=float(np.std(portfolio_daily, ddof=1) * math.sqrt(252)),
                    **metadata[(signal["entry_index"], variant)])
            by_variant[variant] = rows
            weekly_rows.extend(rows)
            block["strategies"][variant] = {
                "primary": _variant_metrics(rows, p["primary_cost_bps"]),
                "costs": {str(c): _variant_metrics(rows, c) for c in p["cost_bps_one_way"]},
                "yearly": {y: _variant_metrics([r for r in rows if r["year"] == y], p["primary_cost_bps"])
                           for y in sorted({r["year"] for r in rows})}}
        for signal_name, score_key in zip(STRATEGIES, ("raw_score", "adjusted_score")):
            previous_score, diagnostics = None, []
            for signal, following in zip(selected, selected[1:]):
                score = np.asarray(signal[score_key])
                future_return = sector_open[following["entry_index"]] / sector_open[signal["entry_index"]] - 1
                row = {"period": name, "signal": signal_name, "signal_date": signal["signal_date"],
                       "entry_date": signal["entry_date"], "exit_date": following["entry_date"],
                       "year": following["entry_date"][:4], "ic": spearman(-score, future_return),
                       "rank_autocorrelation": None if previous_score is None else spearman(score, previous_score)}
                diagnostics.append(row)
                previous_score = score
            diagnostic_rows.extend(diagnostics)
            block["diagnostics"][signal_name] = {
                "ic": _diagnostic_summary(diagnostics, "ic"),
                "rank_autocorrelation": _diagnostic_summary(diagnostics, "rank_autocorrelation")}
            block["comparisons"][signal_name] = _comparison(
                by_variant[f"{signal_name}_fresh"], by_variant[f"{signal_name}_buffer"])
        summary["periods"][name] = block
    return summary, weekly_rows, diagnostic_rows


def _concentration(rows):
    """Subtract selected recorded weeks; never rerun or alter a trading path."""
    positive = sorted((r for r in rows if r["net_pnl"] > 0),
                      key=lambda r: (-r["net_pnl"], r["entry_date"], r["exit_date"]))
    negative = sorted((r for r in rows if r["net_pnl"] < 0),
                      key=lambda r: (r["net_pnl"], r["entry_date"], r["exit_date"]))
    net = float(sum(r["net_pnl"] for r in rows))
    negative_sum = float(sum(r["net_pnl"] for r in negative))
    positive_sum = float(sum(r["net_pnl"] for r in positive))
    pack = lambda r: {k: r[k] for k in ("entry_date", "exit_date", "net_pnl")}
    best = [pack(r) for r in positive[:3]]
    best_sum = float(sum(r["net_pnl"] for r in best))
    largest = pack(positive[0]) if positive else None
    worst = [pack(r) for r in negative[:3]]
    worst_sum = float(sum(r["net_pnl"] for r in worst))
    return {"weeks": len(rows), "net_pnl": net,
            "sum_positive_week_pnl": positive_sum,
            "sum_negative_week_pnl": negative_sum, "negative_week_count": len(negative),
            "top3_negative_weeks": worst, "top3_negative_pnl": worst_sum,
            "top3_share_of_negative_pnl": worst_sum / negative_sum if negative_sum else None,
            "net_without_top3_negative_weeks": net - worst_sum,
            "positive_week_count": len(positive), "top3_positive_weeks": best,
            "top3_positive_pnl": best_sum,
            "top3_share_of_positive_pnl": best_sum / positive_sum if positive_sum else None,
            "net_without_top3_positive_weeks": net - best_sum,
            "largest_winning_week": largest,
            "net_without_largest_winning_week": net - (largest["net_pnl"] if largest else 0)}


def attribute_portfolios(dates, symbols, adj_open, adj_close, protocol, weekly_rows):
    """Explain the existing ledger; no new weights, forecasts or strategy runs.

    All eleven ETFs remain present, including flat positions with closing costs.
    Inputs must contain complete four-variant reporting periods, not sliced weeks.
    """
    opens, closes = np.asarray(adj_open, float), np.asarray(adj_close, float)
    dates, parsed = _validate(dates, symbols, opens, closes, protocol)
    index = {d: i for i, d in enumerate(dates)}
    sectors = protocol["symbols"]
    sector_open = opens[:, [symbols.index(s) for s in sectors]]
    signals = build_signals(dates, symbols, closes, protocol)
    intervals = {(a["entry_date"], b["entry_date"]): a for a, b in zip(signals, signals[1:])}
    grouped = {}
    for row in weekly_rows:
        if row["strategy"] not in VARIANTS:
            raise ValueError("Unknown strategy in attribution input")
        signal, policy = row["strategy"].rsplit("_", 1)
        if row["signal"] != signal or row["policy"] != policy:
            raise ValueError("Inconsistent signal/policy labels")
        grouped.setdefault(row["period"], {}).setdefault(row["strategy"], []).append(row)
    summary = {"periods": {}, "notional": NOTIONAL,
        "interpretation": "Post-hoc accounting diagnosis on already-seen data, not a new strategy or holdout test.",
        "concentration_interpretation": "Top-three positive and negative weeks are selected after outcomes. Subtraction retains the original ledger and is not a tradable rerun.",
        "illustrative_week_selection": "Largest absolute gross difference within each period/signal; ties use earliest entry date. Selected after outcomes.",
        "side_attribution": "Long/short decomposition applies only to gross PnL. Fees use each policy's own trade history, including flat and unchanged positions."}
    asset_rows = []
    summed = ("fresh_gross", "buffer_gross", "fresh_trade_cost", "buffer_trade_cost",
              "fresh_borrow", "buffer_borrow", "fresh_net", "buffer_net",
              "gross_difference", "trading_cost_saved", "borrow_cost_saved", "net_difference",
              "long_gross_difference", "short_gross_difference")
    aggregate = lambda rows: {key: float(sum(r[key] for r in rows)) for key in summed}
    for period, variants in sorted(grouped.items()):
        if set(variants) != set(VARIANTS):
            raise ValueError("Every period needs both signals and both policies")
        keys = {}
        for variant, rows in variants.items():
            rows = sorted(rows, key=lambda r: (r["entry_date"], r["exit_date"]))
            variants[variant] = rows
            keys[variant] = [(r["entry_date"], r["exit_date"]) for r in rows]
            if len(set(keys[variant])) != len(rows):
                raise ValueError("Duplicate weekly observation")
            if any(a["exit_date"] != b["entry_date"] for a, b in zip(rows, rows[1:])):
                raise ValueError("Incomplete reporting period: missing holding interval")
        if any(k != keys[VARIANTS[0]] for k in keys.values()):
            raise ValueError("All four variants must pair on exactly the same dates")
        block = {"strategies": {v: _concentration(r) for v, r in variants.items()}, "signals": {}}
        for signal, score_key in zip(STRATEGIES, ("raw_score", "adjusted_score")):
            fresh, buffered = variants[f"{signal}_fresh"], variants[f"{signal}_buffer"]
            signal_assets, weekly = [], []
            for k, (f, b) in enumerate(zip(fresh, buffered)):
                pair = (f["entry_date"], f["exit_date"])
                if pair not in intervals:
                    raise ValueError("Row is not a complete observed weekly interval")
                source = intervals[pair]
                if any(r["signal_date"] != source["signal_date"] for r in (f, b)):
                    raise ValueError("Signal date does not match the price history")
                start, end = (index[d] for d in pair)
                days = (parsed[end] - parsed[start]).days
                returns = sector_open[end] / sector_open[start] - 1
                order = sorted(range(11), key=lambda j: (source[score_key][j], sectors[j]))
                ranks = {j: rank for rank, j in enumerate(order, 1)}
                values = {}
                for policy, row in (("fresh", f), ("buffer", b)):
                    w, trades = np.asarray(row["weights"], float), np.asarray(row["trade_dollars"], float)
                    if w.shape != (11,) or trades.shape != (11,) or not np.isfinite(trades).all():
                        raise ValueError("Invalid attribution vectors")
                    if not np.isin(w, [-1 / 6, 0, 1 / 6]).all() or (w > 0).sum() != 3 or (w < 0).sum() != 3:
                        raise ValueError("Expected three equal-weight positions per side")
                    if row["calendar_days"] != days:
                        raise ValueError("Borrow day count disagrees with dates")
                    final = np.abs(NOTIONAL * w * (1 + returns)) if k == len(fresh) - 1 else np.zeros(11)
                    gross = NOTIONAL * w * returns
                    cost = (np.abs(trades) + final) * protocol["primary_cost_bps"] / 10_000
                    borrow = NOTIONAL * np.maximum(-w, 0) * protocol["short_borrow_apr"] * days / 365
                    values[policy] = {"weight": w, "gross": gross, "trade_cost": cost,
                                      "borrow": borrow, "net": gross - cost - borrow}
                    checks = [(final.sum() / NOTIONAL, row["final_liquidation_turnover"])]
                    checks += [(v.sum(), row[field]) for v, field in ((gross, "gross_pnl"),
                               (cost, "trading_cost"), (borrow, "borrow_cost"), (gross - cost - borrow, "net_pnl"))]
                    if any(not np.isclose(a, expected, rtol=1e-10, atol=1e-7) for a, expected in checks):
                        raise ValueError("Per-asset accounting does not reconcile to the supplied ledger")
                assets = []
                for j, symbol in enumerate(sectors):
                    fw, bw = float(values["fresh"]["weight"][j]), float(values["buffer"]["weight"][j])
                    category = ("same_position" if fw == bw else "buffer_only" if fw == 0 else
                                "fresh_only" if bw == 0 else "opposite_side")
                    item = {"period": period, "signal": signal, "signal_date": f["signal_date"],
                            "entry_date": pair[0], "exit_date": pair[1], "symbol": symbol,
                            "rank": ranks[j], "asset_return": float(returns[j]), "category": category}
                    item.update({f"{policy}_{key}": float(vector[j])
                                 for policy, parts in values.items() for key, vector in parts.items()})
                    item.update(gross_difference=item["buffer_gross"] - item["fresh_gross"],
                        trading_cost_saved=item["fresh_trade_cost"] - item["buffer_trade_cost"],
                        borrow_cost_saved=item["fresh_borrow"] - item["buffer_borrow"],
                        net_difference=item["buffer_net"] - item["fresh_net"],
                        long_gross_difference=float(NOTIONAL * (max(bw, 0) - max(fw, 0)) * returns[j]),
                        short_gross_difference=float(NOTIONAL * (min(bw, 0) - min(fw, 0)) * returns[j]))
                    assets.append(item)
                record = {"signal_date": f["signal_date"], "entry_date": pair[0], "exit_date": pair[1],
                          **aggregate(assets), "same_position_cost_saved": float(sum(
                              a["trading_cost_saved"] for a in assets if a["category"] == "same_position"))}
                weekly.append(record)
                signal_assets.extend(assets)
            totals = aggregate(signal_assets)
            totals.update(weeks=len(weekly),
                per_etf={s: aggregate([r for r in signal_assets if r["symbol"] == s]) for s in sectors},
                weekly=weekly, same_position_cost_saved=float(sum(r["same_position_cost_saved"] for r in weekly)),
                illustrative_week=min(weekly, key=lambda r: (-abs(r["gross_difference"]), r["entry_date"])) if weekly else None)
            block["signals"][signal] = totals
            asset_rows.extend(signal_assets)
        summary["periods"][period] = block
    return summary, asset_rows
