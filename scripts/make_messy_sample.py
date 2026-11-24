"""Create sample_data/messy_upload_example.csv: an analytics-style export with real-world problems."""
from pathlib import Path

import numpy as np
import pandas as pd

from funnelforge.core import generate

OUT = Path(__file__).resolve().parents[1] / "sample_data" / "messy_upload_example.csv"
GA = {"visit": "session_start", "product_view": "view_item", "add_to_cart": "Add to Cart",
      "checkout": "begin_checkout", "purchase": "purchase"}


def build(seed: int = 9) -> pd.DataFrame:
    d = generate(1500, seed).astype(object)
    d["stage"] = d["stage"].map(GA)
    d["timestamp"] = pd.to_datetime(d["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    d = d.rename(columns={"session_id": "Session", "timestamp": "event_time", "channel": "utm_source",
                          "device": "device_type", "stage": "event_name", "revenue": "order_value"})
    d["page"] = "/p"                                                            # extra column, ignored
    extra = []
    s0 = d[d.event_name == "session_start"].sample(6, random_state=1).copy()
    s0["event_name"] = "scroll"                                                  # unknown stage
    extra.append(s0)
    rp = d[d.event_name == "view_item"].sample(5, random_state=2).copy()      # repeated stage, later time
    rp["event_time"] = (pd.to_datetime(rp["event_time"]) + pd.Timedelta(seconds=30)).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    extra.append(rp)
    extra.append(d.sample(8, random_state=3))                                    # exact duplicates
    d = pd.concat([d] + extra, ignore_index=True)
    idx = d.index.to_numpy()
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, 40, replace=False)
    d.loc[pick[:3], "Session"] = ""                                             # blank session IDs
    d.loc[pick[3:6], "event_time"] = "unknown"                                  # bad timestamps
    nonp = d[(d.event_name == "view_item") & ~d.index.isin(pick)].index[:4]
    d.loc[nonp, "order_value"] = 9.99                                           # revenue on a non-purchase event
    pur = d[(d.event_name == "purchase") & ~d.index.isin(pick)].index[:2]
    d.loc[pur, "order_value"] = -25.0                                           # refund rows
    flip = d[(d.event_name == "begin_checkout") & ~d.index.isin(pick)].index[:4]
    d.loc[flip, "device_type"] = "Desktop/App"                                  # device changes mid-session
    skip_ids = d[(d.event_name == "Add to Cart") & ~d.index.isin(pick)].Session.unique()[:5]
    d = d[~((d.Session.isin(skip_ids)) & (d.event_name == "view_item"))]        # skipped stage
    return d.sample(frac=1, random_state=4).reset_index(drop=True)


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    build().to_csv(OUT, index=False)
    print(OUT)
