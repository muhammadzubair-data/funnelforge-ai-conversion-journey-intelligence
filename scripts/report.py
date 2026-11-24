"""Print the funnel, segment comparison and a sample scenario from the command line.

    python -m scripts.report                    # synthetic demo
    python -m scripts.report --csv export.csv   # any event export in the upload format
"""
from __future__ import annotations

import argparse

import pandas as pd

from funnelforge.core import funnel_detail, generate, prepare, scenario_multi, segment_compare, sensitivity, validate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    a = ap.parse_args()
    if a.csv:
        d, rep = prepare(pd.read_csv(a.csv))
        print(rep.as_frame().to_string(index=False), "\n")
    else:
        d = validate(generate())
    pd.set_option("display.width", 200)
    print(funnel_detail(d).round(4).to_string(index=False))
    for dim in ("channel", "device"):
        s = segment_compare(d, dim)
        print(f"\nBy {dim} (chi-square p = {s['test']['p_value']:.3g})")
        print(s["table"][[dim, "sessions", "purchases", "conversion", "conv_low", "conv_high", "revenue"]].round(4).to_string(index=False))
    r = scenario_multi(d, {"checkout": .05})
    print(f"\nScenario +5 pts into checkout: {r['baseline_purchases']} -> {r['scenario_purchases']:.1f} purchases "
          f"(80% band {r['incremental_purchases_80'][0]:+.1f} to {r['incremental_purchases_80'][1]:+.1f}), "
          f"illustrative extra revenue ${r['incremental_revenue']:,.0f}")
    print("\n+1 point sensitivity\n" + sensitivity(d).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
