"""Synthetic correctness checks; never fetch or inspect historical performance."""
from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path
import unittest

import numpy as np

from analysis import STRATEGIES, account_period, build_signals, run_study, target_weights


PROTOCOL = json.loads((Path(__file__).parent / "protocol.json").read_text())
SYMBOLS = PROTOCOL["symbols"] + [PROTOCOL["market"]]


def weekdays(start, count):
    day, result = date.fromisoformat(start), []
    while len(result) < count:
        if day.weekday() < 5:
            result.append(day.isoformat())
        day += timedelta(days=1)
    return result


def synthetic(count=900):
    dates = weekdays("2018-06-18", count)
    rng = np.random.default_rng(417)
    daily = rng.normal(0.0001, 0.008, size=(count - 1, 12))
    close = 100 * np.exp(np.vstack([np.zeros(12), daily.cumsum(axis=0)]))
    opened = close * np.exp(rng.normal(0, 0.002, size=close.shape))
    return dates, opened, close


def account_fixture(flip=False):
    dates = ["2020-01-03", "2020-01-06", "2020-01-10", "2020-01-13",
             "2020-01-17", "2020-01-20"]
    weights = np.array([1 / 6] * 3 + [0.0] * 5 + [-1 / 6] * 3)
    signals = [{"signal_index": t, "entry_index": t + 1,
                "weights": {s: ((-weights if flip and t == 2 else weights).tolist())
                            for s in STRATEGIES}, "beta": [1.0] * 11}
               for t in (0, 2, 4)]
    return dates, np.full((6, 11), 100.0), signals, weights


class AnalysisTests(unittest.TestCase):
    def account(self, dates, opened, signals, **kwargs):
        return account_period(dates, opened, np.full_like(opened, 100.0), signals,
                              kwargs.pop("start", dates[0]), kwargs.pop("end", dates[-1]),
                              STRATEGIES[0], **kwargs)

    def test_flat_initial_and_final_cost_and_calendar_borrow(self):
        dates, opened, signals, _ = account_fixture()
        rows = self.account(dates, opened, signals)
        self.assertAlmostEqual(rows[0]["entry_turnover"], 1)
        self.assertAlmostEqual(rows[1]["entry_turnover"], 0)
        self.assertAlmostEqual(rows[1]["final_liquidation_turnover"], 1)
        for row in rows:
            self.assertEqual(row["gross_pnl"], 0)
            self.assertEqual(row["calendar_days"], 7)
            self.assertAlmostEqual(row["borrow_cost"], 50_000 * .01 * 7 / 365)
        self.assertAlmostEqual(sum(r["trading_cost"] for r in rows), 100)

    def test_reversing_basket_trades_both_sides(self):
        dates, opened, signals, _ = account_fixture(flip=True)
        rows = self.account(dates, opened, signals)
        self.assertAlmostEqual(rows[1]["entry_turnover"], 2)
        self.assertAlmostEqual(rows[1]["turnover"], 3)

    def test_uniform_drift_zero_pnl_but_nonzero_rebalance(self):
        dates, opened, signals, _ = account_fixture()
        opened[3:] = 110
        rows = self.account(dates, opened, signals)
        self.assertAlmostEqual(rows[0]["gross_pnl"], 0)
        self.assertAlmostEqual(rows[1]["entry_turnover"], .1)

    def test_hand_calculated_units_pnl_cost_and_terminal_account(self):
        dates, opened, signals, weight = account_fixture()
        opened[3, 0], opened[3, 8] = 110, 90
        opened[5, 0], opened[5, 8] = 121, 81
        rows = self.account(dates, opened, signals)
        np.testing.assert_allclose(rows[0]["adjusted_price_units"], 100_000 * weight / 100)
        np.testing.assert_allclose(rows[1]["adjusted_price_units"], 100_000 * weight / opened[3])
        for row in rows:
            self.assertAlmostEqual(row["gross_pnl"], 100_000 / 30)
        self.assertAlmostEqual(rows[1]["entry_turnover"], 1 / 30)
        self.assertAlmostEqual(rows[1]["final_liquidation_turnover"], 1)
        expected = 100_000 + 2 * 100_000 / 30 - 50_000 * .01 * 14 / 365 - 50 * (2 + 1 / 30)
        self.assertAlmostEqual(100_000 + sum(r["net_pnl"] for r in rows), expected)

    def test_cost_monotonicity_and_single_long_gain(self):
        dates, opened, signals, _ = account_fixture()
        opened[3:, 0] = 110
        rows = self.account(dates, opened, signals)
        self.assertAlmostEqual(rows[0]["gross_pnl"], 100_000 * .1 / 6)
        for row in rows:
            net = [row["net_pnl_by_cost"][str(c)] for c in (0, 2, 5, 10)]
            self.assertTrue(all(a >= b for a, b in zip(net, net[1:])))
            self.assertAlmostEqual(net[0] - net[-1], row["turnover"] * 100)

    def test_rank_invariance_common_beta_and_alphabetical_ties(self):
        scores = np.linspace(-.05, .05, 11)
        np.testing.assert_array_equal(target_weights(scores, PROTOCOL["symbols"]),
                                      target_weights(scores - 1.3 * .04, PROTOCOL["symbols"]))
        tied = target_weights(np.zeros(11), PROTOCOL["symbols"])
        ordered = sorted(PROTOCOL["symbols"])
        for j, symbol in enumerate(PROTOCOL["symbols"]):
            self.assertEqual(tied[j], 1 / 6 if symbol in ordered[:3]
                             else -1 / 6 if symbol in ordered[-3:] else 0)

    def test_ols_intercept_exact_beta_and_nonoverlapping_windows(self):
        dates = weekdays("2018-06-18", 200)
        market = .01 * np.sin(np.arange(199) * .41) + .0007
        beta, alpha = np.arange(11) / 5, np.arange(11) * .00001
        daily = np.column_stack([market[:, None] * beta + alpha, market])
        close = 100 * np.exp(np.vstack([np.zeros(12), daily.cumsum(axis=0)]))
        signals = build_signals(dates, SYMBOLS, close, PROTOCOL)
        signal = signals[0]
        t = signal["signal_index"]
        np.testing.assert_allclose(signal["beta"], beta, atol=1e-12)
        np.testing.assert_allclose(signal["adjusted_score"], alpha * 5, atol=1e-12)
        self.assertEqual(signal["beta_fit_last_return_date"], dates[t - 5])
        self.assertEqual(signal["beta_fit_first_return_date"], dates[t - 130])
        perturbed = close.copy()
        perturbed[t - 4:t + 1, :11] *= 1.25
        changed = build_signals(dates, SYMBOLS, perturbed, PROTOCOL)[0]
        np.testing.assert_array_equal(changed["beta"], signal["beta"])
        self.assertFalse(np.allclose(changed["raw_score"], signal["raw_score"]))

    def test_future_perturbations_do_not_change_earlier_signals(self):
        dates, _, close = synthetic()
        cutoff = 400
        before = build_signals(dates, SYMBOLS, close, PROTOCOL)
        altered = close.copy()
        altered[cutoff + 1:] *= np.linspace(.7, 1.4, 12)
        after = build_signals(dates, SYMBOLS, altered, PROTOCOL)
        self.assertEqual([s for s in before if s["signal_index"] <= cutoff],
                         [s for s in after if s["signal_index"] <= cutoff])

    def test_period_boundaries_start_flat_and_never_fill_signal_close(self):
        dates, opened, signals, _ = account_fixture()
        opened[0] = 1  # The signal bar's open must never enter trading PnL.
        rows = self.account(dates, opened, signals, start="2020-01-07")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["entry_date"], "2020-01-13")
        self.assertEqual(rows[0]["exit_date"], "2020-01-20")
        self.assertAlmostEqual(rows[0]["entry_turnover"], 1)
        self.assertAlmostEqual(rows[0]["turnover"], 2)
        self.assertEqual(rows[0]["gross_pnl"], 0)
        self.assertEqual(self.account(dates, opened, signals, start="2020-01-07", end="2020-01-19"), [])
        alternate_close = np.full_like(opened, 100.0)
        alternate_close[0, 0] = 50.0
        different_gap = account_period(dates, opened, alternate_close, signals, dates[0], dates[-1], STRATEGIES[0])
        self.assertEqual(different_gap[0]["gross_pnl"], 0)
        self.assertAlmostEqual(different_gap[0]["gap_diagnostic_return"], 1 / 6)

    def test_holiday_week_entry_is_next_available_session(self):
        dates, _, close = synthetic(240)
        signals = build_signals(dates, SYMBOLS, close, PROTOCOL)
        chosen = signals[4]
        removed = chosen["entry_index"]  # remove a synthetic Monday holiday
        dates.pop(removed)
        close = np.delete(close, removed, axis=0)
        signals = build_signals(dates, SYMBOLS, close, PROTOCOL)
        signal = next(s for s in signals if s["signal_date"] == chosen["signal_date"])
        self.assertEqual(date.fromisoformat(signal["entry_date"]).weekday(), 1)
        self.assertGreater(signal["entry_date"], signal["signal_date"])

    def test_price_scale_invariance_and_complete_json_study(self):
        dates, opened, close = synthetic(900)
        protocol = deepcopy(PROTOCOL)
        protocol.update(development_end="2019-12-31", evaluation_start="2020-01-01", evaluation_end="2021-11-15")
        summary, rows, signal_rows = run_study(dates, SYMBOLS, opened, close, protocol)
        scale = np.linspace(.7, 1.4, 12)
        rescaled, other_rows, _ = run_study(dates, SYMBOLS, opened * scale, close * scale, protocol)
        self.assertEqual(len(rows), len(other_rows))
        for a, b in zip(rows, other_rows):
            self.assertAlmostEqual(a["gross_pnl"], b["gross_pnl"], places=7)
            self.assertAlmostEqual(a["turnover"], b["turnover"], places=10)
        self.assertTrue(signal_rows)
        self.assertEqual(summary["periods"]["evaluation"]["comparison"]["bootstrap"]["draws"], 5000)
        for period, block in summary["periods"].items():
            for strategy in STRATEGIES:
                subset = [r for r in rows if r["period"] == period and r["strategy"] == strategy]
                self.assertTrue(all(block["start"] <= r["entry_date"] < r["exit_date"] <= block["end"] for r in subset))
                self.assertAlmostEqual(subset[0]["entry_turnover"], 1)
                self.assertGreater(subset[-1]["final_liquidation_turnover"], 0)
        json.dumps((summary, rows, signal_rows), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
