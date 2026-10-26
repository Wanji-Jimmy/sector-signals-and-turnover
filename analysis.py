"""Frozen two-signal sector study: numpy and standard library only.

All returns/PnL use synthetic adjusted-price units and fixed gross notional.
This module has no data fetching and performs no work at import time.
"""
from datetime import date
import math

import numpy as np


STRATEGIES = ("raw_reversal", "market_adjusted_reversal")
NOTIONAL = 100_000.0


def target_weights(scores, symbols):
    """Bottom three long, top three short; ascending symbol breaks score ties."""
    scores = np.asarray(scores, dtype=float)
    if len(scores) != 11 or len(symbols) != 11 or not np.isfinite(scores).all():
        raise ValueError("Exactly eleven finite sector scores are required")
    order = sorted(range(11), key=lambda j: (scores[j], symbols[j]))
    weights = np.zeros(11)
    weights[order[:3]], weights[order[-3:]] = 1 / 6, -1 / 6
    return weights


def _validate(dates, symbols, adj_open, adj_close, protocol):
    dates = [str(d) for d in dates]
    parsed = [date.fromisoformat(d) for d in dates]
    if len(set(dates)) != len(dates) or parsed != sorted(parsed):
        raise ValueError("Dates must be unique and increasing")
    if len(set(symbols)) != len(symbols):
        raise ValueError("Duplicate symbols")
    expected = protocol["symbols"] + [protocol["market"]]
    if len(protocol["symbols"]) != 11 or set(symbols) != set(expected):
        raise ValueError("Inputs must contain the fixed eleven sectors and market")
    for prices in (adj_open, adj_close):
        if prices.shape != (len(dates), len(symbols)):
            raise ValueError("Price matrix does not match dates and symbols")
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("All aligned prices must be positive and finite")
    return dates, parsed


def build_signals(dates, symbols, adj_close, protocol):
    """Build JSON-compatible weekly signals using only closes through signal t.

    Signals retain integer row indices for account_period. Output vectors follow
    protocol['symbols'], irrespective of input matrix column order.
    """
    dates = [str(d) for d in dates]
    closes = np.asarray(adj_close, dtype=float)
    dates, parsed = _validate(dates, symbols, closes, closes, protocol)
    sector_cols = [symbols.index(s) for s in protocol["symbols"]]
    market_col = symbols.index(protocol["market"])
    daily = np.diff(np.log(closes), axis=0)
    horizon, window = protocol["signal_sessions"], protocol["beta_sessions"]
    weeks = [d.isocalendar()[:2] for d in parsed]
    signals = []
    for t in range(window + horizon, len(dates) - 1):
        if weeks[t] == weeks[t + 1]:
            continue
        end = t - horizon
        # daily[j-1] ends on price row j; slice ends at return row end.
        fit = daily[end - window:end]
        market = fit[:, market_col]
        centered = market - market.mean()
        denominator = centered @ centered
        if denominator <= 1e-18:
            raise ValueError(f"Degenerate market beta fit at {dates[t]}")
        sectors = fit[:, sector_cols]
        beta = centered @ (sectors - sectors.mean(axis=0)) / denominator
        raw = np.log(closes[t, sector_cols] / closes[end, sector_cols])
        market_move = math.log(closes[t, market_col] / closes[end, market_col])
        adjusted = raw - beta * market_move
        signals.append({
            "signal_date": dates[t], "signal_index": t,
            "entry_date": dates[t + 1], "entry_index": t + 1,
            "beta_fit_first_return_date": dates[end - window + 1],
            "beta_fit_last_return_date": dates[end],
            "raw_score": raw.tolist(), "adjusted_score": adjusted.tolist(),
            "market_signal_return": market_move, "beta": beta.tolist(),
            "weights": {name: target_weights(score, protocol["symbols"]).tolist()
                        for name, score in zip(STRATEGIES, (raw, adjusted))},
        })
    return signals


def account_period(dates, adj_open, adj_close, signals, start, end, strategy,
                   *, notional=NOTIONAL, cost_bps=(0, 2, 5, 10),
                   primary_cost_bps=5, borrow_apr=0.01):
    """Account one period, starting flat and liquidating its last complete week.

    Price matrices here contain only the eleven sectors in signal-vector order.
    Costs of the last liquidation belong to the final holding-period row.
    """
    dates = [str(d) for d in dates]
    opens, closes = np.asarray(adj_open), np.asarray(adj_close)
    intervals = [(a, b["entry_index"]) for a, b in zip(signals, signals[1:])
                 if start <= dates[a["entry_index"]] <= dates[b["entry_index"]] <= end]
    rows, previous_quantity = [], np.zeros(opens.shape[1])
    for k, (signal, exit_index) in enumerate(intervals):
        entry_index, t = signal["entry_index"], signal["signal_index"]
        if rows and rows[-1]["exit_date"] != dates[entry_index]:
            raise ValueError("Cannot net previous holdings across a gap between intervals")
        weight = np.asarray(signal["weights"][strategy])
        entry, exit_price = opens[entry_index], opens[exit_index]
        target = notional * weight
        quantity = target / entry
        trade_dollars = target - previous_quantity * entry
        entry_turnover = float(np.abs(trade_dollars).sum() / notional)
        liquidation = (float(np.abs(quantity * exit_price).sum() / notional)
                       if k == len(intervals) - 1 else 0.0)
        turnover = entry_turnover + liquidation
        gross = float(quantity @ (exit_price - entry))
        days = (date.fromisoformat(dates[exit_index])
                - date.fromisoformat(dates[entry_index])).days
        borrow = float(-target[weight < 0].sum() * borrow_apr * days / 365)
        net = {str(c): gross - borrow - notional * turnover * c / 10_000
               for c in cost_bps}
        rows.append({
            "strategy": strategy, "signal_date": dates[t],
            "entry_date": dates[entry_index], "exit_date": dates[exit_index],
            "year": dates[exit_index][:4], "calendar_days": days,
            "gross_pnl": gross, "gross_return": gross / notional,
            "borrow_cost": borrow, "entry_turnover": entry_turnover,
            "final_liquidation_turnover": liquidation, "turnover": turnover,
            "net_pnl_by_cost": net, "net_pnl": net[str(primary_cost_bps)],
            "net_return": net[str(primary_cost_bps)] / notional,
            "trading_cost": notional * turnover * primary_cost_bps / 10_000,
            "ex_ante_basket_beta": float(weight @ np.asarray(signal["beta"])),
            "gap_diagnostic_return": float(weight @ (entry / closes[t] - 1)),
            "weights": weight.tolist(), "adjusted_price_units": quantity.tolist(),
            "trade_dollars": trade_dollars.tolist(),
        })
        previous_quantity = quantity
    return rows


def _metrics(rows, cost, notional=NOTIONAL):
    """Weekly-marked account statistics; no daily drawdown claim."""
    if not rows:
        return {"weeks": 0}
    pnl = np.array([r["net_pnl_by_cost"][str(cost)] for r in rows])
    returns = pnl / notional
    equity = np.r_[notional, notional + np.cumsum(pnl)]
    peak = np.maximum.accumulate(equity)
    volatility = float(np.std(returns, ddof=1)) if len(rows) > 1 else 0.0
    turnover = sum(r["turnover"] for r in rows)
    gross = sum(r["gross_pnl"] for r in rows)
    borrow = sum(r["borrow_cost"] for r in rows)
    beta = np.array([r["ex_ante_basket_beta"] for r in rows])
    gap = np.array([r["gap_diagnostic_return"] for r in rows])
    return {
        "weeks": len(rows), "first_entry": rows[0]["entry_date"],
        "last_exit": rows[-1]["exit_date"], "one_way_cost_bps": cost,
        "gross_pnl": gross, "borrow_cost": borrow,
        "trading_cost": notional * turnover * cost / 10_000,
        "net_pnl": float(pnl.sum()), "net_pnl_per_notional": float(returns.sum()),
        "mean_weekly_return": float(returns.mean()),
        "annualized_mean_return": float(returns.mean() * 52),
        "annualized_volatility": volatility * math.sqrt(52),
        "sharpe_zero_cash_rate": (float(returns.mean() / volatility * math.sqrt(52))
                                  if volatility > 0 else None),
        "max_drawdown_weekly_account": float(np.max((peak - equity) / peak)),
        "max_drawdown_dollars": float(np.max(peak - equity)),
        "turnover_total": turnover, "turnover_mean_weekly": turnover / len(rows),
        "break_even_one_way_bps_after_borrow": ((gross - borrow) / (notional * turnover) * 10_000
                                                 if turnover > 0 else None),
        "basket_beta_mean": float(beta.mean()),
        "basket_beta_mean_absolute": float(np.abs(beta).mean()),
        "basket_beta_min": float(beta.min()), "basket_beta_max": float(beta.max()),
        "gap_diagnostic_mean_weekly_return": float(gap.mean()),
        "gap_diagnostic_sum_return": float(gap.sum()),
    }


def paired_comparison(raw_rows, adjusted_rows, *, bootstrap=False):
    """Paired weekly return differences; moving blocks do not wrap end to start."""
    keys = lambda rows: [(r["entry_date"], r["exit_date"]) for r in rows]
    if keys(raw_rows) != keys(adjusted_rows):
        raise ValueError("Strategy observations must pair exactly")
    if not raw_rows:
        return {"weeks": 0}
    delta = np.array([a["net_return"] - r["net_return"]
                      for r, a in zip(raw_rows, adjusted_rows)])
    years = np.array([r["year"] for r in raw_rows])
    annual = {y: {"weeks": int((years == y).sum()),
                  "mean_weekly_difference": float(delta[years == y].mean()),
                  "annualized_mean_difference": float(delta[years == y].mean() * 52),
                  "net_pnl_difference": float(delta[years == y].sum() * NOTIONAL)}
              for y in sorted(set(years))}
    leave_out = {y: float(delta[years != y].mean() * 52)
                 for y in sorted(set(years)) if (years != y).any()}
    result = {"weeks": len(delta), "mean_weekly_difference": float(delta.mean()),
              "annualized_mean_difference": float(delta.mean() * 52),
              "net_pnl_difference": float(delta.sum() * NOTIONAL),
              "yearly": annual, "leave_one_year_out_annualized_mean": leave_out}
    if bootstrap:
        block, draws, seed = 4, 5000, 260126
        if len(delta) < block:
            result["bootstrap"] = {"status": "insufficient_weeks", "block_weeks": block}
        else:
            rng = np.random.default_rng(seed)
            starts = rng.integers(0, len(delta) - block + 1,
                                  size=(draws, math.ceil(len(delta) / block)))
            indices = (starts[:, :, None] + np.arange(block)).reshape(draws, -1)[:, :len(delta)]
            means = delta[indices].mean(axis=1)
            interval = np.quantile(means, [0.025, 0.975])
            result["bootstrap"] = {"method": "overlapping moving blocks, no wrap",
                "interpretation": "conditional descriptive interval, not a prospective test",
                "block_weeks": block, "draws": draws, "seed": seed,
                "mean_weekly_difference_ci95": interval.tolist(),
                "annualized_mean_difference_ci95": (interval * 52).tolist()}
    return result


def run_study(dates, symbols, adj_open, adj_close, protocol):
    """Return (summary, weekly_rows, sector_signal_rows), all JSON serializable."""
    opens, closes = np.asarray(adj_open, float), np.asarray(adj_close, float)
    dates, _ = _validate(dates, symbols, opens, closes, protocol)
    signals = build_signals(dates, symbols, closes, protocol)
    columns = [symbols.index(s) for s in protocol["symbols"]]
    summary = {"notional": NOTIONAL, "primary_cost_bps": protocol["primary_cost_bps"],
               "year_attribution": "Whole holding interval exit year, not exact calendar-year marked PnL; annual views do not reset holdings.",
               "drawdown_basis": "Weekly-marked synthetic account starting at USD 100000; annual views rebase.",
               "annualization": "52 times mean weekly fixed-notional PnL return; not CAGR.",
               "signal_count": len(signals), "periods": {}}
    weekly_rows = []
    for period in ("development", "evaluation"):
        start, end = protocol[f"{period}_start"], protocol[f"{period}_end"]
        strategies, by_strategy = {}, {}
        for strategy in STRATEGIES:
            rows = account_period(dates, opens[:, columns], closes[:, columns], signals,
                                  start, end, strategy, cost_bps=protocol["cost_bps_one_way"],
                                  primary_cost_bps=protocol["primary_cost_bps"],
                                  borrow_apr=protocol["short_borrow_apr"])
            for row in rows:
                row["period"] = period
            by_strategy[strategy] = rows
            weekly_rows.extend(rows)
            strategies[strategy] = {
                "primary": _metrics(rows, protocol["primary_cost_bps"]),
                "costs": {str(c): _metrics(rows, c) for c in protocol["cost_bps_one_way"]},
                "yearly": {y: _metrics([r for r in rows if r["year"] == y], protocol["primary_cost_bps"])
                           for y in sorted({r["year"] for r in rows})},
            }
        summary["periods"][period] = {"start": start, "end": end,
            "strategies": strategies, "comparison": paired_comparison(
                by_strategy[STRATEGIES[0]], by_strategy[STRATEGIES[1]], bootstrap=period == "evaluation")}
    signal_rows = [{"signal_date": sig["signal_date"], "entry_date": sig["entry_date"],
                    "symbol": symbol, "beta": sig["beta"][j],
                    "beta_fit_first_return_date": sig["beta_fit_first_return_date"],
                    "beta_fit_last_return_date": sig["beta_fit_last_return_date"],
                    "raw_score": sig["raw_score"][j], "adjusted_score": sig["adjusted_score"][j],
                    "raw_weight": sig["weights"][STRATEGIES[0]][j],
                    "adjusted_weight": sig["weights"][STRATEGIES[1]][j]}
                   for sig in signals for j, symbol in enumerate(protocol["symbols"])]
    return summary, weekly_rows, signal_rows
