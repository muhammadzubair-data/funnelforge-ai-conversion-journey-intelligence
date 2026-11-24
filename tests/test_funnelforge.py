"""Tests for the v2 core: preparation edge cases, funnel maths, SQL vs pandas, daily attribution, scenarios."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from funnelforge.core import (STAGES, daily, daily_full, funnel, funnel_detail, generate, kpis, prepare, scenario,
                              scenario_multi, segment_compare, segment_sql, segments, sensitivity, stage_reach_by_segment,
                              true_conversion, validate, wilson)
from scripts.make_messy_sample import build as messy


@pytest.fixture(scope="module")
def d():
    return validate(generate())


def ev(rows):
    return pd.DataFrame(rows, columns=["session_id", "timestamp", "channel", "device", "stage", "revenue"])


def session(sid, n, start="2026-01-01 10:00", ch="Email", dev="Mobile", rev=50.0):
    t = pd.Timestamp(start)
    return [(sid, t + pd.Timedelta(minutes=i), ch, dev, STAGES[i], rev if STAGES[i] == "purchase" else 0) for i in range(n)]


# ---------------------------------------------------------------- data contract
def test_bundled_csv_is_generator_output():
    pd.testing.assert_frame_equal(pd.read_csv("data/sample.csv", parse_dates=["timestamp"]), generate())


def test_vectorised_validate_matches_original_rules(d):
    assert d.session_id.nunique() == 4000 and len(d) == 9095


@pytest.mark.parametrize("mutate, msg", [
    (lambda x: x.drop(columns="stage"), "Required columns"),
    (lambda x: x.assign(stage=x.stage.where(x.index != 0, "unknown")), "Invalid"),
    (lambda x: x.assign(revenue=np.where(x.index == 0, 5.0, x.revenue)), "only on purchase"),
    (lambda x: x[~((x.session_id == x.loc[x.stage == "purchase", "session_id"].iloc[0]) & (x.stage == "visit"))],
     "contiguous"),
])
def test_strict_validate_rejects(mutate, msg):
    with pytest.raises(ValueError, match=msg):
        validate(mutate(generate(50)))


def test_strict_validate_rejects_backwards_time():
    rows = session("A", 3)
    rows[2] = (*rows[2][:1], pd.Timestamp("2026-01-01 09:00"), *rows[2][2:])
    with pytest.raises(ValueError, match="chronological"):
        validate(ev(rows))


def test_strict_validate_rejects_changing_device():
    rows = session("A", 3)
    rows[1] = ("A", rows[1][1], "Email", "Desktop", rows[1][4], 0)
    with pytest.raises(ValueError, match="consistent"):
        validate(ev(rows))


def test_prepare_round_trip_on_clean_data(d):
    c, rep = prepare(generate())
    assert rep.rows_out == 9095 and rep.sessions_out == 4000 and rep.invalid_sequence_sessions == 0
    pd.testing.assert_frame_equal(funnel(c), funnel(d))


def test_prepare_messy_export_counts_every_problem():
    c, rep = prepare(messy())
    assert (rep.dropped_session_id, rep.dropped_timestamp, rep.dropped_stage, rep.dropped_revenue) == (3, 3, 6, 2)
    assert (rep.revenue_zeroed, rep.exact_duplicates, rep.repeated_stage_rows) == (4, 8, 5)
    assert rep.inconsistent_dimension_sessions == 4 and rep.invalid_sequence_sessions == 7
    assert rep.unknown_stage_values == ["scroll"] and rep.stages_mapped["view_item"] == "product_view"
    assert rep.renamed["utm_source"] == "channel"
    validate(c)                                     # cleaned output satisfies the original strict contract


def test_prepare_optional_columns_and_aliases():
    raw = pd.DataFrame({"Session": ["a", "a", "b"], "Event Time": ["2026-01-01 10:00", "2026-01-01 10:05", "2026-01-02"],
                        "Event": ["Session Start", "Order", "visit"]})
    c, rep = prepare(raw)                            # 'Event Time' → timestamp, 'Event' → stage, 'Order' → purchase
    assert rep.renamed == {"Session": "session_id", "Event Time": "timestamp", "Event": "stage"}
    assert rep.filled_channel and rep.filled_device and rep.filled_revenue
    assert rep.invalid_sequence_sessions == 1 and c.session_id.unique().tolist() == ["b"]


def test_prepare_converts_time_zones_to_utc():
    raw = pd.DataFrame({"session_id": ["a", "a"], "timestamp": ["2026-01-01T23:30:00-05:00", "2026-01-02T04:40:00Z"],
                        "stage": ["visit", "product_view"]})
    c, _ = prepare(raw)
    assert str(c.timestamp.iloc[0]) == "2026-01-02 04:30:00+00:00"
    assert daily(c).date.tolist() == [pd.Timestamp("2026-01-02").date()]


def test_prepare_same_timestamp_uses_stage_order():
    raw = ev(session("a", 3)).assign(timestamp="2026-01-01 10:00")
    c, rep = prepare(raw)
    validate(c)
    assert c.stage.tolist() == STAGES[:3] and rep.notes


def test_prepare_repeated_stage_keeps_first():
    rows = session("a", 3) + [("a", pd.Timestamp("2026-01-01 10:01:30"), "Email", "Mobile", "product_view", 0)]
    c, rep = prepare(ev(rows))
    assert rep.repeated_stage_rows == 1 and len(c) == 3


@pytest.mark.parametrize("raw, msg", [(pd.DataFrame(), "no rows"),
                                      (pd.DataFrame({"session_id": ["a"], "timestamp": ["x"], "stage": ["visit"]}), "No valid")])
def test_prepare_rejects_unusable(raw, msg):
    with pytest.raises(ValueError, match=msg):
        prepare(raw)


# ---------------------------------------------------------------- funnel and SQL
def test_funnel_known_counts():
    rows = session("a", 5) + session("b", 3) + session("c", 1) + session("d", 4)
    f = funnel(validate(ev(rows)))
    assert f.sessions.tolist() == [4, 3, 3, 2, 1]
    assert f.step_rate.iloc[1:].tolist() == pytest.approx([.75, 1, 2 / 3, .5])
    assert f.overall_rate.iloc[-1] == .25


def test_zero_denominator_is_undefined_not_zero():
    f = funnel(validate(ev(session("a", 1))))
    assert f.sessions.tolist() == [1, 0, 0, 0, 0] and np.isnan(f.step_rate.iloc[2])


def test_sql_matches_pandas(d):
    s = segment_sql(d, "device").set_index("device")
    for dev, g in d.groupby("device"):
        assert s.loc[dev, "sessions"] == g.session_id.nunique()
        assert s.loc[dev, "purchases"] == (g.stage == "purchase").sum()
        assert s.loc[dev, "add_to_cart"] == (g.stage == "add_to_cart").sum()
        assert s.loc[dev, "revenue"] == pytest.approx(g.revenue.sum())


def test_sql_rejects_unknown_dimension(d):
    with pytest.raises(ValueError):
        segment_sql(d, "channel; DROP TABLE events")
    with pytest.raises(ValueError, match="Missing required"):
        prepare(pd.DataFrame({"session_id": ["a"], "stage": ["visit"]}))


def test_segment_values_with_quotes_are_safe():
    rows = session("a", 5, ch="O'Brien \"; DROP TABLE events; --") + session("b", 2)
    s = segments(validate(ev(rows)), "channel")
    assert s.sessions.sum() == 2


def test_segment_test_matches_scipy_and_truth(d):
    ch = segment_compare(d, "channel")
    assert ch["test"]["p_value"] > .05                 # generator: channels identical
    dev = segment_compare(d, "device")
    t = dev["table"].set_index("device")
    assert dev["test"]["p_value"] < 1e-6
    tab = np.c_[t.purchases, t.sessions - t.purchases]
    assert dev["test"]["p_value"] == pytest.approx(stats.chi2_contingency(tab, correction=False)[1])
    for k in ("Mobile", "Desktop"):
        assert t.loc[k, "conv_low"] <= true_conversion(k) <= t.loc[k, "conv_high"]


def test_wilson_reference():
    assert wilson(81, 263) == pytest.approx((0.2553, 0.3662), abs=5e-5)


def test_funnel_detail_intervals_contain_rates(d):
    f = funnel_detail(d).iloc[1:]
    assert ((f.step_low <= f.step_rate) & (f.step_rate <= f.step_high)).all()


def test_reach_by_segment_starts_at_one(d):
    r = stage_reach_by_segment(d, "channel")
    assert (r[r.stage == "Visit"].reach == 1).all()


def test_kpis(d):
    k = kpis(d)
    assert k["sessions"] == 4000 and k["purchases"] == 444 and k["aov"] == pytest.approx(d.revenue.sum() / 444)
    assert k["worst_stage"] == "add_to_cart"


# ---------------------------------------------------------------- daily
def test_daily_attributes_to_session_start():
    rows = session("a", 5, start="2026-01-01 23:58")         # purchase happens on 2 Jan
    t = daily(validate(ev(rows)))
    assert len(t) == 1 and t.purchases.iloc[0] == 1 and str(t.date.iloc[0]) == "2026-01-01"


def test_daily_full_includes_empty_days_and_pooled_rolling():
    rows = session("a", 5, start="2026-01-01") + session("b", 1, start="2026-01-04")
    t = daily_full(validate(ev(rows)), window=7)
    assert len(t) == 4 and t.sessions.tolist() == [1, 0, 0, 1]
    assert np.isnan(t.conversion.iloc[1]) and t.conversion_7d.iloc[-1] == pytest.approx(.5)


def test_daily_totals_reconcile(d):
    t = daily_full(d)
    assert t.sessions.sum() == 4000 and t.revenue.sum() == pytest.approx(d.revenue.sum())


# ---------------------------------------------------------------- scenarios
def test_original_scenario_works_on_pandas_3(d):
    assert scenario(d, "checkout", 0)["scenario_purchases"] == pytest.approx(444)


def test_scenario_multi_matches_original_for_one_step(d):
    a = scenario(d, "checkout", .05)
    b = scenario_multi(d, {"checkout": .05})
    assert b["scenario_purchases"] == pytest.approx(a["scenario_purchases"])
    assert b["incremental_revenue"] == pytest.approx(a["incremental_revenue"])
    lo, hi = b["incremental_purchases_80"]
    assert lo < b["incremental_purchases"] < hi


def test_scenario_caps_at_100(d):
    r = scenario_multi(d, {"purchase": .9})
    assert r["capped"] and r["rates"].scenario_rate.max() == 1.0


def test_scenario_combined_is_multiplicative(d):
    f = funnel(d)
    r = scenario_multi(d, {"add_to_cart": .05, "checkout": .05})
    rates = f.step_rate.iloc[1:].to_numpy().copy()
    rates[1] += .05
    rates[2] += .05
    assert r["scenario_purchases"] == pytest.approx(4000 * np.prod(rates))


def test_scenario_invalid(d):
    with pytest.raises(ValueError):
        scenario_multi(d, {"visit": .1})
    with pytest.raises(ValueError):
        scenario_multi(d, {"checkout": 1.5})


def test_sensitivity_ranks_weakest_step_first(d):
    s = sensitivity(d)
    assert s.stage.iloc[0] == "add_to_cart" and (np.diff(s.incremental_purchases) <= 1e-9).all()
