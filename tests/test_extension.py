"""Synthetic tests for buffering; importing/running this file reads no market data."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "study"))

from analysis import NOTIONAL, STRATEGIES, account_period, build_signals, target_weights
from extension import (_attribution, _concentration, _diagnostic_summary, _period_signals,
                       attribute_portfolios, buffered_weights,
                       run_extension, spearman)
from test_analysis import PROTOCOL, SYMBOLS, synthetic


EXTENSION = json.loads((ROOT / "study" / "extension_protocol.json").read_text())
SECTORS = PROTOCOL["symbols"]


def scores_for_order(order):
    values = np.empty(11)
    values[order] = np.arange(11)
    return values


def signal_fixture():
    dates = ["2020-01-03", "2020-01-06", "2020-01-10", "2020-01-13",
             "2020-01-17", "2020-01-20", "2020-01-24", "2020-01-27"]
    orders = [list(range(11)), [0, 1, 3, 4, 2, 5, 6, 7, 8, 9, 10],
              [0, 1, 3, 4, 5, 2, 6, 7, 8, 9, 10], list(range(11))]
    signals = []
    for t, order in zip((0, 2, 4, 6), orders):
        score = scores_for_order(order)
        signals.append({"signal_index": t, "entry_index": t + 1,
                        "signal_date": dates[t], "entry_date": dates[t + 1],
                        "raw_score": score.tolist(), "adjusted_score": score.tolist(),
                        "weights": {s: target_weights(score, SECTORS).tolist() for s in STRATEGIES},
                        "beta": [1.0] * 11})
    return dates, signals


class ExtensionTests(unittest.TestCase):
    def test_long_rank_five_is_kept_and_six_is_replaced(self):
        previous = target_weights(np.arange(11), SECTORS)
        rank_five = scores_for_order([0, 1, 3, 4, 2, 5, 6, 7, 8, 9, 10])
        kept = buffered_weights(rank_five, SECTORS, previous)
        np.testing.assert_array_equal(kept, previous)
        rank_six = scores_for_order([0, 1, 3, 4, 5, 2, 6, 7, 8, 9, 10])
        replaced = buffered_weights(rank_six, SECTORS, kept)
        self.assertEqual(replaced[2], 0)
        self.assertEqual(replaced[3], 1 / 6)

    def test_short_rank_seven_is_kept_and_six_is_replaced(self):
        previous = target_weights(np.arange(11), SECTORS)
        rank_seven = scores_for_order([0, 1, 2, 3, 4, 5, 8, 6, 7, 9, 10])
        kept = buffered_weights(rank_seven, SECTORS, previous)
        np.testing.assert_array_equal(kept, previous)
        rank_six = scores_for_order([0, 1, 2, 3, 4, 8, 5, 6, 7, 9, 10])
        replaced = buffered_weights(rank_six, SECTORS, kept)
        self.assertEqual(replaced[8], 0)
        self.assertEqual(replaced[7], -1 / 6)

    def test_sides_remain_disjoint_full_and_equal_weight(self):
        rng, previous = np.random.default_rng(98), None
        for _ in range(100):
            previous = buffered_weights(rng.normal(size=11), SECTORS, previous)
            self.assertEqual(np.count_nonzero(previous > 0), 3)
            self.assertEqual(np.count_nonzero(previous < 0), 3)
            self.assertAlmostEqual(abs(previous).sum(), 1)
            self.assertAlmostEqual(previous.sum(), 0)
            self.assertTrue(np.isin(previous, [-1 / 6, 0, 1 / 6]).all())

    def test_alphabetical_ties_and_zero_buffer_exact_fresh(self):
        symbols = SECTORS[::-1]
        tied = buffered_weights(np.zeros(11), symbols)
        expected = target_weights(np.zeros(11), symbols)
        np.testing.assert_array_equal(tied, expected)
        rng, previous = np.random.default_rng(9), -expected
        for _ in range(30):
            scores = rng.integers(-2, 3, size=11)
            zero = buffered_weights(scores, symbols, previous, buffer=0)
            np.testing.assert_array_equal(zero, target_weights(scores, symbols))
            previous = buffered_weights(scores, symbols, previous)

    def test_period_reset_excludes_prior_membership_and_partial_interval(self):
        dates, signals = signal_fixture()
        full, _ = _period_signals(signals, dates, dates[0], dates[-1], SECTORS, 2)
        reset, metadata = _period_signals(signals, dates, "2020-01-07", dates[-1], SECTORS, 2)
        variant = "raw_reversal_buffer"
        self.assertEqual(full[1]["weights"][variant][2], 1 / 6)
        self.assertEqual(reset[0]["entry_date"], "2020-01-13")
        self.assertEqual(reset[0]["weights"][variant][2], 0)
        self.assertEqual(metadata[(3, variant)]["initial_position_count"], 6)
        self.assertEqual(metadata[(3, variant)]["replacement_count"], 0)
        empty, _ = _period_signals(signals, dates, "2020-01-07", "2020-01-19", SECTORS, 2)
        self.assertEqual(empty, [])

    def test_unchanged_members_still_rebalance_drift_and_charge_liquidation(self):
        dates, signals = signal_fixture()
        selected, metadata = _period_signals(signals, dates, dates[0], "2020-01-20", SECTORS, 2)
        opened = np.full((len(dates), 11), 100.0)
        opened[3:] = 110.0
        rows = account_period(dates, opened, opened, selected, dates[0], "2020-01-20", "raw_reversal_buffer")
        self.assertEqual(metadata[(3, "raw_reversal_buffer")]["replacement_count"], 0)
        self.assertAlmostEqual(rows[1]["entry_turnover"], .1)
        self.assertAlmostEqual(rows[1]["final_liquidation_turnover"], 1)
        self.assertAlmostEqual(rows[1]["trading_cost"], 55)

    def test_replacement_names_count_slots_but_not_initial_positions(self):
        dates, signals = signal_fixture()
        _, metadata = _period_signals(signals, dates, dates[0], dates[-1], SECTORS, 2)
        self.assertEqual(metadata[(1, "raw_reversal_buffer")]["replacement_count"], 0)
        self.assertEqual(metadata[(3, "raw_reversal_buffer")]["replacement_count"], 0)
        final = metadata[(5, "raw_reversal_buffer")]
        self.assertEqual(final["replacement_count"], 1)
        self.assertEqual(final["long_removed"], [SECTORS[2]])
        self.assertEqual(final["long_added"], [SECTORS[3]])

    def test_attribution_identity_and_signed_cost_crossover(self):
        fresh = [{"gross_pnl": 100, "trading_cost": 20, "borrow_cost": 5,
                  "net_pnl": 75, "turnover": .4}]
        buffered = [{"gross_pnl": 90, "trading_cost": 5, "borrow_cost": 3,
                     "net_pnl": 82, "turnover": .1}]
        result = _attribution(fresh, buffered)
        self.assertEqual(result["gross_pnl_difference"], -10)
        self.assertEqual(result["trading_cost_saved"], 15)
        self.assertEqual(result["borrow_cost_saved"], 2)
        self.assertEqual(result["net_pnl_difference"], 7)
        self.assertEqual(result["identity_residual"], 0)
        cost = result["crossover_one_way_bps"]
        self.assertAlmostEqual(cost, 8 / 3)
        f = fresh[0]["gross_pnl"] - fresh[0]["borrow_cost"] - NOTIONAL * fresh[0]["turnover"] * cost / 10_000
        b = buffered[0]["gross_pnl"] - buffered[0]["borrow_cost"] - NOTIONAL * buffered[0]["turnover"] * cost / 10_000
        self.assertAlmostEqual(f, b)
        buffered[0].update(gross_pnl=110, net_pnl=102)
        self.assertLess(_attribution(fresh, buffered)["crossover_one_way_bps"], 0)
        self.assertIsNone(_attribution(fresh, fresh)["crossover_one_way_bps"])

    def test_spearman_average_ties_and_weekly_observation_bootstrap(self):
        self.assertAlmostEqual(spearman([1, 1, 2, 3], [4, 4, 2, 1]), -1)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))
        values = [{"ic": x} for x in [.2, .1, -.3, .4, -.2, .1, .6, 0]]
        result = _diagnostic_summary(values, "ic")
        self.assertEqual(result["observations"], 8)
        self.assertAlmostEqual(result["mean"], .1125)
        self.assertEqual(result["bootstrap"]["block_weeks"], 4)
        self.assertEqual(result["bootstrap"]["draws"], 5000)
        self.assertEqual(result, _diagnostic_summary(values, "ic"))

    def test_api_fresh_matches_original_exactly_and_zero_buffer_matches_fresh(self):
        dates, opened, close = synthetic(600)
        extension = deepcopy(EXTENSION)
        extension["periods"] = [{"name": "synthetic", "start": "2019-01-01", "end": dates[-1]}]
        summary, rows, diagnostic = run_extension(dates, SYMBOLS, opened, close, PROTOCOL, extension)
        signals = build_signals(dates, SYMBOLS, close, PROTOCOL)
        for signal in STRATEGIES:
            original = account_period(dates, opened[:, :11], close[:, :11], signals,
                                      "2019-01-01", dates[-1], signal)
            fresh = [r for r in rows if r["strategy"] == f"{signal}_fresh"]
            self.assertEqual(len(original), len(fresh))
            for old, new in zip(original, fresh):
                self.assertEqual({k: v for k, v in old.items() if k != "strategy"},
                                 {k: new[k] for k in old if k != "strategy"})
            diag = [r for r in diagnostic if r["signal"] == signal]
            self.assertEqual(len(diag), len(fresh))
            self.assertIsNone(diag[0]["rank_autocorrelation"])
            source = next(s for s in signals if s["entry_date"] == diag[0]["entry_date"])
            following = signals[signals.index(source) + 1]
            score = source["raw_score" if signal == STRATEGIES[0] else "adjusted_score"]
            expected_ic = spearman(-np.array(score), opened[following["entry_index"], :11] / opened[source["entry_index"], :11] - 1)
            self.assertEqual(diag[0]["ic"], expected_ic)
        extension["rank_buffer"] = 0
        zero_summary, zero_rows, zero_diag = run_extension(dates, SYMBOLS, opened, close, PROTOCOL, extension)
        self.assertEqual(diagnostic, zero_diag)
        for signal in STRATEGIES:
            compared = zero_summary["periods"]["synthetic"]["comparisons"][signal]
            self.assertEqual(compared["net_pnl_difference"], 0)
            self.assertEqual(compared["decomposition"]["identity_residual"], 0)
            self.assertEqual(compared["bootstrap"]["mean_weekly_difference_ci95"], [0, 0])
            fresh = [r for r in zero_rows if r["strategy"] == f"{signal}_fresh"]
            buffer = [r for r in zero_rows if r["strategy"] == f"{signal}_buffer"]
            for f, b in zip(fresh, buffer):
                self.assertEqual({k: v for k, v in f.items() if k not in ("strategy", "policy")},
                                 {k: v for k, v in b.items() if k not in ("strategy", "policy")})
        json.dumps((summary, rows, diagnostic, zero_summary), allow_nan=False)

    def test_future_mutation_cannot_change_earlier_buffer_decisions_or_pnl(self):
        dates, opened, close = synthetic(600)
        extension = deepcopy(EXTENSION)
        extension["periods"] = [{"name": "synthetic", "start": "2019-01-01", "end": dates[-1]}]
        _, before, _ = run_extension(dates, SYMBOLS, opened, close, PROTOCOL, extension)
        changed_open, changed_close = opened.copy(), close.copy()
        cutoff = 400
        changed_open[cutoff + 1:] *= np.linspace(.6, 1.6, 12)
        changed_close[cutoff + 1:] *= np.linspace(.6, 1.6, 12)
        _, after, _ = run_extension(dates, SYMBOLS, changed_open, changed_close, PROTOCOL, extension)
        self.assertEqual([r for r in before if r["exit_date"] <= dates[cutoff]],
                         [r for r in after if r["exit_date"] <= dates[cutoff]])
        self.assertEqual([(r["strategy"], r["weights"]) for r in before if r["signal_date"] <= dates[cutoff]],
                         [(r["strategy"], r["weights"]) for r in after if r["signal_date"] <= dates[cutoff]])

    def test_empty_and_separate_periods_are_json_safe_and_reset(self):
        dates, opened, close = synthetic(600)
        extension = deepcopy(EXTENSION)
        extension["periods"] = [{"name": "a", "start": "2019-01-01", "end": "2019-12-31"},
                                {"name": "b", "start": "2020-01-01", "end": dates[-1]},
                                {"name": "empty", "start": "2026-01-01", "end": "2026-10-23"}]
        summary, rows, diagnostics = run_extension(dates, SYMBOLS, opened, close, PROTOCOL, extension)
        for period in ("a", "b"):
            for signal in STRATEGIES:
                fresh = [r for r in rows if r["period"] == period and r["strategy"] == f"{signal}_fresh"]
                buffered = [r for r in rows if r["period"] == period and r["strategy"] == f"{signal}_buffer"]
                self.assertEqual(fresh[0]["weights"], buffered[0]["weights"])
                self.assertAlmostEqual(buffered[0]["entry_turnover"], 1)
                self.assertEqual(buffered[0]["replacement_count"], 0)
                self.assertGreater(buffered[-1]["final_liquidation_turnover"], 0)
                self.assertAlmostEqual(summary["periods"][period]["comparisons"][signal]["decomposition"]["identity_residual"], 0)
        self.assertEqual(summary["periods"]["empty"]["strategies"]["raw_reversal_fresh"]["primary"]["weeks"], 0)
        json.dumps((summary, rows, diagnostics), allow_nan=False)


def attribution_fixture():
    dates, opened, close = synthetic(170)
    selected = deepcopy(build_signals(dates, SYMBOLS, close, PROTOCOL)[:3])
    opened[:] = 100.0
    opened[selected[1]["entry_index"]:, 0] = 110.0
    opened[selected[2]["entry_index"]:, 8] = 120.0
    f, b, common = np.zeros(11), np.zeros(11), np.zeros(11)
    f[[0, 1, 2]], f[[8, 9, 10]] = 1 / 6, -1 / 6
    b[[0, 1, 3]], b[[7, 9, 10]] = 1 / 6, -1 / 6
    common[[0, 1, 4]], common[[8, 9, 10]] = 1 / 6, -1 / 6
    rows = []
    for signal in STRATEGIES:
        for policy, first in (("fresh", f), ("buffer", b)):
            variant = f"{signal}_{policy}"
            for source, weights in zip(selected, (first, common)):
                source["weights"][variant] = weights.tolist()
            ledger = account_period(dates, opened[:, :11], close[:, :11], selected,
                                    selected[0]["entry_date"], selected[2]["entry_date"], variant)
            for row in ledger:
                row.update(period="hand", signal=signal, policy=policy)
            rows.extend(ledger)
    return dates, opened, close, rows


class AttributionTests(unittest.TestCase):
    def test_hand_short_borrow_fees_and_final_liquidation(self):
        dates, opened, close, rows = attribution_fixture()
        summary, assets = attribute_portfolios(dates, SYMBOLS, opened, close, PROTOCOL, rows)
        entry = rows[1]["entry_date"]
        short = next(a for a in assets if a["signal"] == STRATEGIES[0]
                     and a["entry_date"] == entry and a["symbol"] == SECTORS[8])
        self.assertAlmostEqual(short["fresh_gross"], -100_000 / 6 * .2)
        self.assertAlmostEqual(short["buffer_gross"], short["fresh_gross"])
        self.assertAlmostEqual(short["fresh_borrow"], 100_000 / 6 * .01 * 7 / 365)
        self.assertAlmostEqual(short["fresh_trade_cost"], 20_000 * .0005)
        self.assertAlmostEqual(short["buffer_trade_cost"], (100_000 / 6 + 20_000) * .0005)
        self.assertAlmostEqual(short["net_difference"], -100_000 / 6 * .0005)
        first_long = next(a for a in assets if a["signal"] == STRATEGIES[0]
                         and a["entry_date"] == rows[0]["entry_date"] and a["symbol"] == SECTORS[0])
        self.assertAlmostEqual(first_long["fresh_gross"], 100_000 / 6 * .1)
        self.assertAlmostEqual(first_long["fresh_trade_cost"], 100_000 / 6 * .0005)
        self.assertEqual(first_long["fresh_borrow"], 0)
        self.assertEqual(len(assets), 2 * 2 * 11)
        json.dumps((summary, assets), allow_nan=False)

    def test_both_flat_and_same_weights_keep_history_dependent_fees(self):
        dates, opened, close, rows = attribution_fixture()
        summary, assets = attribute_portfolios(dates, SYMBOLS, opened, close, PROTOCOL, rows)
        last = [a for a in assets if a["signal"] == STRATEGIES[0] and a["entry_date"] == rows[1]["entry_date"]]
        closed_fresh = next(a for a in last if a["symbol"] == SECTORS[2])
        self.assertEqual((closed_fresh["fresh_weight"], closed_fresh["buffer_weight"]), (0, 0))
        self.assertEqual(closed_fresh["category"], "same_position")
        self.assertAlmostEqual(closed_fresh["fresh_trade_cost"], 100_000 / 6 * .0005)
        self.assertEqual(closed_fresh["buffer_trade_cost"], 0)
        self.assertAlmostEqual(closed_fresh["trading_cost_saved"], 100_000 / 6 * .0005)
        self.assertTrue(any(a["gross_difference"] == 0 and a["trading_cost_saved"] != 0 for a in last))
        week = summary["periods"]["hand"]["signals"][STRATEGIES[0]]["weekly"][1]
        self.assertAlmostEqual(week["same_position_cost_saved"], sum(a["trading_cost_saved"] for a in last))

    def test_existing_results_reconcile_and_are_not_changed(self):
        dates, opened, close = synthetic(600)
        spec = deepcopy(EXTENSION)
        spec["periods"] = [{"name": "synthetic", "start": "2019-01-01", "end": dates[-1]}]
        original, rows, _ = run_extension(dates, SYMBOLS, opened, close, PROTOCOL, spec)
        saved = deepcopy(rows)
        summary, assets = attribute_portfolios(dates, SYMBOLS, opened, close, PROTOCOL, rows)
        self.assertEqual(rows, saved)
        for signal in STRATEGIES:
            result = summary["periods"]["synthetic"]["signals"][signal]
            old = original["periods"]["synthetic"]["comparisons"][signal]["decomposition"]
            self.assertAlmostEqual(result["gross_difference"], old["gross_pnl_difference"])
            self.assertAlmostEqual(result["net_difference"], old["net_pnl_difference"])
            self.assertAlmostEqual(result["long_gross_difference"] + result["short_gross_difference"], result["gross_difference"])
            for field in ("net_difference", "fresh_net", "buffer_net", "trading_cost_saved"):
                self.assertAlmostEqual(sum(a[field] for a in result["per_etf"].values()), result[field])
            chosen = result["illustrative_week"]
            self.assertEqual(abs(chosen["gross_difference"]), max(abs(w["gross_difference"]) for w in result["weekly"]))
        self.assertEqual(len(assets), len(rows) // 2 * 11)

    def test_column_permutation_and_empty_input(self):
        dates, opened, close, rows = attribution_fixture()
        expected = attribute_portfolios(dates, SYMBOLS, opened, close, PROTOCOL, rows)
        order = list(reversed(range(12)))
        permuted = attribute_portfolios(dates, [SYMBOLS[j] for j in order],
                                       opened[:, order], close[:, order], PROTOCOL, list(reversed(rows)))
        self.assertEqual(expected, permuted)
        summary, assets = attribute_portfolios(dates, SYMBOLS, opened, close, PROTOCOL, [])
        self.assertEqual(summary["periods"], {})
        self.assertEqual(assets, [])

    def test_missing_duplicates_misalignment_and_tampered_ledger_rejected(self):
        dates, opened, close, rows = attribution_fixture()
        cases = [rows[:-2], rows + [rows[0]]]
        wrong = deepcopy(rows)
        wrong[0]["net_pnl"] += 1
        cases.append(wrong)
        wrong = deepcopy(rows)
        wrong[0]["exit_date"] = rows[0]["entry_date"]
        cases.append(wrong)
        for bad in cases:
            with self.assertRaises(ValueError):
                attribute_portfolios(dates, SYMBOLS, opened, close, PROTOCOL, bad)

    def test_concentration_handles_signs_small_samples_and_ties(self):
        def sample(values):
            return [{"entry_date": f"2020-01-{i + 1:02}", "exit_date": f"2020-01-{i + 2:02}", "net_pnl": v}
                    for i, v in enumerate(values)]
        mixed = _concentration(sample([10, -4, 5, 10, -2, 1, -1, -3, 0]))
        self.assertEqual(mixed["top3_positive_pnl"], 25)
        self.assertAlmostEqual(mixed["top3_share_of_positive_pnl"], 25 / 26)
        self.assertEqual(mixed["top3_negative_pnl"], -9)
        self.assertAlmostEqual(mixed["top3_share_of_negative_pnl"], .9)
        self.assertEqual(mixed["net_without_top3_positive_weeks"], -9)
        self.assertEqual(mixed["net_without_top3_negative_weeks"], 25)
        self.assertEqual(mixed["largest_winning_week"]["entry_date"], "2020-01-01")
        negative = _concentration(sample([-3, 0, -2]))
        self.assertIsNone(negative["top3_share_of_positive_pnl"])
        self.assertIsNone(negative["largest_winning_week"])
        self.assertEqual(negative["net_without_top3_positive_weeks"], -5)
        self.assertEqual(negative["top3_negative_pnl"], -5)
        small = _concentration(sample([2, 1]))
        self.assertEqual(len(small["top3_positive_weeks"]), 2)
        self.assertEqual(small["top3_share_of_positive_pnl"], 1)
        self.assertIsNone(small["top3_share_of_negative_pnl"])
        self.assertEqual(small["net_without_top3_positive_weeks"], 0)
        self.assertEqual(_concentration([])["net_pnl"], 0)


if __name__ == "__main__":
    unittest.main()
