"""FunnelForge AI — session-level e-commerce conversion intelligence (Streamlit).

Synthetic demonstration data by default. Run: streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
import io
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from funnelforge.core import (MIN_SEGMENT_SESSIONS, SEGMENT_SQL, STAGE_LABELS, STAGES, TRUE_STEP, daily_full,
                              funnel_detail, generate, kpis, prepare, scenario_multi, segment_compare, sensitivity,
                              stage_reach_by_segment, true_conversion, validate)

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
WEBSITE = "https://zubairlabs.com"
REPO = "https://github.com/muhammadzubair-data/funnelforge-ai-conversion-journey-intelligence"
NAVY, AQUA, TEAL, ORANGE, VIOLET, ROSE, AMBER, SLATE = ("#101B32", "#54E5D0", "#14B8A6", "#F97316", "#8B5CF6",
                                                        "#F43F5E", "#F59E0B", "#94A3B8")
SEQ = [TEAL, ORANGE, VIOLET, "#3B82F6", ROSE, AMBER, "#84CC16", "#EC4899", "#64748B", "#A855F7"]
STAGE_COLORS = ["#115E59", "#0F766E", "#0E7490", "#C2410C", "#EA580C"]   # dark enough for white text
MAX_UPLOAD_MB = 50

st.set_page_config(page_title="FunnelForge AI | ZubairLabs", page_icon=str(ASSETS / "icon.svg"), layout="wide",
                   menu_items={"About": "FunnelForge AI by ZubairLabs — descriptive funnel analytics on synthetic data. "
                                        "Scenarios are hypothetical, not causal forecasts.",
                               "Report a bug": REPO + "/issues"})
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), link=WEBSITE)

st.markdown(f"""<style>
.block-container {{max-width: 1400px; padding-top: 3.2rem; padding-bottom: 3rem;}}
.ff-hero {{position: relative; overflow: hidden; padding: 24px 28px 26px 28px; border-radius: 20px; margin-bottom: 16px; color: #F8FAFC;
  background: radial-gradient(680px 260px at 0% 0%, rgba(84,229,208,.28), transparent 60%),
              radial-gradient(600px 260px at 100% 0%, rgba(249,115,22,.26), transparent 62%),
              linear-gradient(120deg, #0A1224 0%, {NAVY} 55%, #18284A 100%);
  border: 1px solid rgba(148,163,184,.22); box-shadow: 0 12px 32px rgba(2,8,23,.22);}}
.ff-hero:after {{content: ""; position: absolute; right: 36px; top: 20px; width: 130px; height: 110px; opacity: .28;
  background: linear-gradient(180deg, {AQUA}, {ORANGE});
  clip-path: polygon(0 0, 100% 0, 88% 24%, 12% 24%, 0 0, 14% 30%, 86% 30%, 76% 52%, 24% 52%, 14% 30%, 26% 58%, 74% 58%, 66% 78%, 34% 78%, 26% 58%, 36% 84%, 64% 84%, 58% 100%, 42% 100%, 36% 84%);}}
.ff-hero .eyebrow {{letter-spacing: .2em; font-size: 11px; font-weight: 700; color: {AQUA}; text-transform: uppercase;}}
.ff-hero h1 {{font-size: 31px; margin: 6px 0 4px 0; padding: 0; color: #F8FAFC; font-weight: 800;}}
.ff-hero p {{margin: 0; color: #CBD5E1; max-width: min(980px, calc(100% - 180px));}}
.pill {{display: inline-block; margin: 10px 6px 0 0; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.2); color: #E2E8F0;}}
.pill.warn {{background: rgba(245,158,11,.2); border-color: rgba(245,158,11,.55); color: #FDE68A;}}
.pill.ok {{background: rgba(20,184,166,.22); border-color: rgba(84,229,208,.6); color: #CCFBF1;}}
.kpi-grid {{display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 6px 0 14px 0;}}
.kpi {{border-radius: 16px; padding: 14px 16px; border: 1px solid rgba(148,163,184,.3); position: relative; overflow: hidden;
  background: linear-gradient(160deg, rgba(20,184,166,.10), rgba(249,115,22,.06));}}
.kpi:before {{content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: linear-gradient(180deg, {AQUA}, {ORANGE});}}
.kpi .label {{font-size: 11.5px; font-weight: 700; opacity: .75; text-transform: uppercase; letter-spacing: .06em;}}
.kpi .value {{font-size: 25px; font-weight: 800; margin-top: 4px; line-height: 1.1;}}
.kpi .sub {{font-size: 12.5px; opacity: .75; margin-top: 4px;}}
.note {{border-left: 4px solid {TEAL}; padding: 10px 14px; border-radius: 10px; background: rgba(20,184,166,.09); margin: 6px 0 12px 0;}}
.note.warn {{border-left-color: {AMBER}; background: rgba(245,158,11,.11);}}
.note.ok {{border-left-color: #22C55E; background: rgba(34,197,94,.09);}}
.note.violet {{border-left-color: {VIOLET}; background: rgba(139,92,246,.10);}}
.side-head {{font-weight: 800; font-size: 13px; letter-spacing: .08em; text-transform: uppercase; opacity: .78; margin: 2px 0 6px 0;}}
.footer {{opacity: .65; font-size: 12px; text-align: center; margin-top: 28px;}}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def esc(s) -> str:
    return html.escape(str(s))


def kpi_row(items):
    st.markdown("<div class='kpi-grid'>" + "".join(
        f"<div class='kpi' role='group' aria-label='{esc(a)}'><div class='label'>{esc(a)}</div>"
        f"<div class='value'>{esc(b)}</div><div class='sub'>{esc(c)}</div></div>" for a, b, c in items) + "</div>",
        unsafe_allow_html=True)


def note(text, kind=""):
    st.markdown(f"<div class='note {kind}' role='note'>{text}</div>", unsafe_allow_html=True)


def chart(fig, height=380, caption: str | None = None):
    top = 50
    if fig.layout.legend.orientation == "h":
        fig.update_layout(legend=dict(y=1.0, yanchor="bottom", x=0, xanchor="left"))
        top = 82
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10), legend_title_text="",
                      title_font=dict(size=15), title=dict(y=.98, yanchor="top"), font=dict(size=13))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    if caption:
        st.caption(caption)


def csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.DatetimeTZDtype):
            out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        elif out[c].dtype == object or str(out[c].dtype).startswith(("string", "str", "category")):
            out[c] = out[c].astype(object).map(
                lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v)
    return out.to_csv(index=False).encode()


def pct(v, dp=1) -> str:
    return f"{v:.{dp}%}" if v is not None and np.isfinite(v) else "—"


def money(v, dp=0) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    s = "-" if v < 0 else ""
    v = abs(v)
    return f"{s}${v / 1e6:,.2f}M" if v >= 1e6 else f"{s}${v / 1e3:,.1f}k" if v >= 1e4 else f"{s}${v:,.{dp}f}"


# ------------------------------------------------------------------ data
ss = st.session_state


@st.cache_data(show_spinner=False, max_entries=2)
def load_demo() -> pd.DataFrame:
    return validate(generate())


with st.sidebar:
    st.markdown("<div class='side-head'>Data source</div>", unsafe_allow_html=True)
    has_upload = "upload_df" in ss
    opts = ["Synthetic demo (4,000 sessions)"] + ([f"Uploaded: {ss['upload_name']}"] if has_upload else [])
    if ss.get("source_choice") not in opts:          # a new upload clears the choice, so it is selected
        ss["source_choice"] = opts[-1]
    src = st.radio("Data source", opts, key="source_choice", label_visibility="collapsed")
    is_demo = not src.startswith("Uploaded")
    base = load_demo() if is_demo else ss["upload_df"]
    source_label = "Synthetic demo · 4,000 sessions · seed 42" if is_demo else ss["upload_name"]

    st.markdown("<div class='side-head'>Filters</div>", unsafe_allow_html=True)
    filt = base
    for c in ("channel", "device"):
        options = sorted(base[c].unique())
        sel = st.multiselect(c.title(), options, default=options, key=f"f_{c}_{source_label}")
        filt = filt[filt[c].isin(sel)]
    starts = base.groupby("session_id").timestamp.min()
    d0, d1 = starts.min().date(), starts.max().date()
    rng = st.date_input("Session start (UTC)", (d0, d1), min_value=d0, max_value=d1, key=f"f_date_{source_label}")
    if isinstance(rng, tuple) and len(rng) == 2:
        keep = starts[(starts.dt.date >= rng[0]) & (starts.dt.date <= rng[1])].index
        filt = filt[filt.session_id.isin(keep)]
    with st.expander("About these results", icon="🧭"):
        st.markdown("- Data is **synthetic** unless you upload your own.\n"
                    "- Each session is attributed to its own channel and device and to the day it **started**.\n"
                    "- Segment differences are **descriptive**: segments are not randomised.\n"
                    "- Scenarios are **hypothetical** what-ifs, not forecasts.")
    st.caption("Theme: ⋮ menu (top right) → Light, Dark or System")
    st.caption(f"[ZubairLabs]({WEBSITE}) · [Source code]({REPO})")

data = filt
filtered = len(data) != len(base)
fkey = (source_label, len(data), int(data.session_id.nunique()), float(data.revenue.sum()))


def cached(name: str, fn):
    """Results live in this browser session only (bounded), so one visitor's upload never reaches another's."""
    store = ss.setdefault("results", OrderedDict())
    key = (fkey, ss.get("upload_key") if not is_demo else "demo", name)
    if key not in store:
        store[key] = fn()
        while len(store) > 32:
            store.popitem(last=False)
    store.move_to_end(key)
    return store[key]


def hero(eyebrow, title, text):
    pills = ("<span class='pill warn'>Synthetic demonstration data</span>" if is_demo
             else "<span class='pill ok'>Your uploaded file · this session only</span>")
    if filtered:
        pills += f"<span class='pill'>Filtered: {data.session_id.nunique():,} sessions</span>"
    pills += "<span class='pill'>Descriptive, not causal</span>"
    st.markdown(f"<header class='ff-hero'><div class='eyebrow'>{esc(eyebrow)}</div><h1>{esc(title)}</h1>"
                f"<p>{esc(text)}</p>{pills}</header>", unsafe_allow_html=True)


def footer():
    st.markdown(f"<div class='footer'>FunnelForge AI · built by <a href='{WEBSITE}' target='_blank'>ZubairLabs</a> · "
                f"{esc(source_label)}</div>", unsafe_allow_html=True)


def need_data() -> bool:
    if data.empty:
        st.warning("No sessions match the filters. Widen the channel, device or date selection.", icon="🔎")
        footer()
        return False
    return True


# ------------------------------------------------------------------ page 1: funnel
def page_funnel():
    hero("Conversion funnel", "Funnel Overview",
         "How many sessions reach each stage, where they leave, and how sure we can be about each rate.")
    if not need_data():
        return
    k = cached("kpis", lambda: kpis(data))
    f = cached("funnel", lambda: funnel_detail(data))
    kpi_row([("Sessions", f"{k['sessions']:,}", f"{k['days']} days of session starts"),
             ("Purchases", f"{k['purchases']:,}", f"{pct(k['conversion'], 2)} of sessions"),
             ("Revenue", money(k["revenue"]), f"{money(k['rps'], 2)} per session"),
             ("Average order", money(k["aov"], 2), "revenue ÷ purchases"),
             ("Weakest step", STAGE_LABELS.get(k["worst_stage"], "—"), f"{pct(k['worst_rate'])} continue")])
    c1, c2 = st.columns([1.15, 1])
    with c1:
        fig = go.Figure(go.Funnel(y=f.label, x=f.sessions, textinfo="value+percent initial+percent previous",
                                  marker=dict(color=STAGE_COLORS), connector=dict(line=dict(color=SLATE, width=1)),
                                  insidetextfont=dict(color="white"),
                                  hovertemplate="%{y}: %{x:,} sessions<extra></extra>"))
        fig.update_layout(title="Sessions reaching each stage")
        chart(fig, 420, "Labels: sessions, % of visits, % of the previous stage.")
    with c2:
        s = f.iloc[1:]
        fig = go.Figure(go.Bar(x=s.step_rate, y=[f"→ {x}" for x in s.label], orientation="h", marker_color=TEAL,
                               text=[pct(v) for v in s.step_rate], textposition="inside", insidetextanchor="start",
                               textfont=dict(color="white"),
                               error_x=dict(type="data", symmetric=False, array=s.step_high - s.step_rate,
                                            arrayminus=s.step_rate - s.step_low, color=SLATE)))
        fig.update_layout(title="Step conversion with 95% intervals", xaxis_tickformat=".0%", xaxis_range=[0, 1],
                          yaxis=dict(autorange="reversed"))
        chart(fig, 420, "Share of sessions at the previous stage that continue. Whiskers: Wilson 95% interval.")
    s = f.iloc[1:]
    worst = s.sort_values("dropoff").iloc[-1]
    note(f"The largest loss in sessions is between <b>{esc(f.label.iloc[s.dropoff.idxmax() - 1])}</b> and "
         f"<b>{esc(worst.label)}</b>: {int(worst.dropoff):,} sessions ({pct(worst.dropoff_rate)} of those who got that far). "
         "That is where a change would touch the most visitors — whether it can be improved is a separate question.")
    if is_demo and not filtered:
        rows = [f"{STAGE_LABELS[st_]}: true {pct(p)} (Desktop/Tablet), {pct(p * .85)} (Mobile)" for st_, p in TRUE_STEP.items()]
        note("<b>Synthetic check.</b> The generator's step probabilities are " + "; ".join(rows) +
             ". Observed step rates mix the device shares, so they sit between the two. Real data has no answer key.", "ok")
    st.markdown("##### Funnel table")
    show = f[["label", "sessions", "step_rate", "step_low", "step_high", "overall_rate", "dropoff", "dropoff_rate"]]
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"label": "stage", **{c: st.column_config.NumberColumn(c, format="percent") for c in
                                                     ["step_rate", "step_low", "step_high", "overall_rate", "dropoff_rate"]}})
    st.download_button("Download funnel (CSV)", csv_bytes(show), "funnelforge_funnel.csv", "text/csv", icon="⬇️")
    footer()


# ------------------------------------------------------------------ page 2: segments
def page_segments():
    hero("Segment comparison", "Channel & Device Comparison",
         "Compare conversion, revenue per session and every funnel step across acquisition channels and devices, "
         "aggregated in SQLite, with intervals and a test of whether segments differ at all.")
    if not need_data():
        return
    dim = st.segmented_control("Compare by", ["channel", "device"], default="channel", format_func=str.title) or "channel"
    sc = cached(f"seg_{dim}", lambda: segment_compare(data, dim))
    t, test = sc["table"], sc["test"]
    best = t.sort_values("conversion").iloc[-1]
    kpi_row([("Segments", f"{len(t)}", f"by {dim}"), ("Highest conversion", str(best[dim]), pct(best.conversion, 2)),
             ("Conversion range", f"{pct(t.conversion.min(), 1)} – {pct(t.conversion.max(), 1)}", "across segments"),
             ("Do segments differ?", f"p = {test['p_value']:.3g}" if np.isfinite(test["p_value"]) else "—",
              f"χ² = {test['chi2']:.1f}, df {test['dof']}" if np.isfinite(test["chi2"]) else "needs 2+ segments")])
    if np.isfinite(test["p_value"]):
        if test["p_value"] >= .05:
            note(f"Conversion does not differ detectably between {dim} segments (p = {test['p_value']:.2f}). Ranking them "
                 "would mostly be ranking noise.", "ok")
        else:
            note(f"Conversion differs between {dim} segments (p = {test['p_value']:.2g}). This is an association: "
                 f"segments differ in many ways besides {dim}, so it does not show that the {dim} itself causes the gap.", "violet")
    if t.small.any():
        note(f"⚠️ Segments with fewer than {MIN_SEGMENT_SESSIONS} sessions are flagged; their rates are unreliable.", "warn")
    if is_demo and not filtered:
        if dim == "channel":
            note("<b>Synthetic check.</b> The generator gives every channel the same conversion, so any channel gap here is "
                 "noise.", "ok")
        else:
            note(f"<b>Synthetic check.</b> True conversion is {pct(true_conversion('Desktop'))} on Desktop and Tablet and "
                 f"{pct(true_conversion('Mobile'))} on Mobile. Tablet's observed rate rests on few sessions, so it can "
                 "drift well away from its true value.", "ok")
    c1, c2 = st.columns(2)
    with c1:
        o = t.sort_values("conversion")
        fig = go.Figure(go.Bar(x=o.conversion, y=o[dim].astype(str), orientation="h",
                               marker_color=[AMBER if s else TEAL for s in o.small],
                               text=[pct(v, 1) for v in o.conversion], textposition="inside", insidetextanchor="start",
                               textfont=dict(color="white"),
                               error_x=dict(type="data", symmetric=False, array=o.conv_high - o.conversion,
                                            arrayminus=o.conversion - o.conv_low, color=SLATE)))
        fig.update_layout(title=f"Conversion by {dim} with 95% intervals", xaxis_tickformat=".0%")
        chart(fig, 360, "Amber bars mark small segments.")
    with c2:
        o = t.sort_values("revenue_per_session")
        fig = px.bar(o, x="revenue_per_session", y=o[dim].astype(str), orientation="h", color_discrete_sequence=[ORANGE],
                     title=f"Revenue per session by {dim}", labels={"revenue_per_session": "Revenue per session ($)", "y": ""},
                     text=[money(v, 2) for v in o.revenue_per_session])
        fig.update_layout(yaxis_title=None)
        chart(fig, 360)
    reach = stage_reach_by_segment(data, dim)
    fig = px.line(reach, x="stage", y="reach", color=dim, markers=True, color_discrete_sequence=SEQ,
                  title=f"Share of sessions reaching each stage, by {dim}", labels={"reach": "Share of sessions", "stage": ""})
    fig.update_layout(yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.1, x=0))
    c3, c4 = st.columns([1.2, 1])
    with c3:
        chart(fig, 380)
    with c4:
        sr = sc["step_rates"].set_index(dim)
        hm = go.Figure(go.Heatmap(z=sr.values, x=[STAGE_LABELS[s] for s in sr.columns], y=sr.index.astype(str),
                                  colorscale=[[0, "#0F172A"], [.5, TEAL], [1, AQUA]], zmin=0, zmax=1,
                                  text=[[pct(v, 0) for v in row] for row in sr.values], texttemplate="%{text}",
                                  hovertemplate="%{y} → %{x}: %{z:.1%}<extra></extra>", colorbar=dict(tickformat=".0%")))
        hm.update_layout(title="Step conversion by segment")
        chart(hm, 380, "Each cell: share of sessions at the previous stage that continue.")
    st.dataframe(t, hide_index=True, width="stretch",
                 column_config={"revenue": st.column_config.NumberColumn("revenue", format="$%.2f"),
                                "revenue_per_session": st.column_config.NumberColumn("revenue / session", format="$%.2f"),
                                "aov": st.column_config.NumberColumn("avg order", format="$%.2f"),
                                **{c: st.column_config.NumberColumn(c, format="percent") for c in
                                   ["conversion", "conv_low", "conv_high", "share"]}})
    st.download_button("Download segment table (CSV)", csv_bytes(t), f"funnelforge_segments_{dim}.csv", "text/csv", icon="⬇️")
    with st.expander("The SQL behind this table", icon="🗄️"):
        st.code(SEGMENT_SQL.format(dim=dim).strip(), language="sql")
        st.caption("Runs on an in-memory SQLite copy of the filtered events. Only the whitelisted column name is "
                   "inserted into the query text; data values never are.")
    footer()


# ------------------------------------------------------------------ page 3: daily
def page_daily():
    hero("Daily trends", "Daily Trends",
         "Sessions, purchases, revenue and conversion by the day each session started, with a 7-day rolling view "
         "and weekday pattern.")
    if not need_data():
        return
    t = cached("daily", lambda: daily_full(data))
    kpi_row([("Days", f"{len(t):,}", f"{int((t.sessions == 0).sum())} with no sessions"),
             ("Sessions per day", f"{t.sessions.mean():,.1f}", f"max {t.sessions.max():,}"),
             ("Revenue per day", money(t.revenue.mean(), 0), f"total {money(t.revenue.sum())}"),
             ("Latest 7-day conversion", pct(t.conversion_7d.iloc[-1], 2), f"to {t.date.iloc[-1]:%d %b %Y}")])
    fig = go.Figure()
    fig.add_bar(x=t.date, y=t.sessions, name="Sessions", marker_color=TEAL, opacity=.75)
    fig.add_bar(x=t.date, y=t.purchases, name="Purchases", marker_color=ORANGE)
    fig.add_scatter(x=t.date, y=t.conversion_7d, name="7-day conversion", yaxis="y2", line=dict(color=VIOLET, width=3))
    fig.update_layout(title="Daily sessions and purchases with 7-day rolling conversion", barmode="overlay",
                      yaxis_title="Sessions / purchases",
                      yaxis2=dict(title="Conversion", overlaying="y", side="right", tickformat=".0%", showgrid=False,
                                  rangemode="tozero"), legend=dict(orientation="h", y=1.1, x=0))
    chart(fig, 420, "Rolling conversion = purchases in the last 7 days ÷ sessions in the last 7 days.")
    c1, c2 = st.columns(2)
    with c1:
        f2 = go.Figure()
        f2.add_bar(x=t.date, y=t.revenue, name="Daily revenue", marker_color=ORANGE, opacity=.55)
        f2.add_scatter(x=t.date, y=t.revenue_7d, name="7-day average", line=dict(color=VIOLET, width=3))
        f2.update_layout(title="Daily revenue", yaxis_tickprefix="$", legend=dict(orientation="h", y=1.1, x=0))
        chart(f2, 340)
    with c2:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        wd = t.groupby("weekday").agg(sessions=("sessions", "sum"), purchases=("purchases", "sum"), days=("date", "size"))
        wd = wd.reindex([w for w in order if w in wd.index])
        wd["conversion"] = wd.purchases / wd.sessions.where(wd.sessions > 0)
        f3 = px.bar(wd.reset_index(), x="weekday", y="conversion", color_discrete_sequence=[TEAL],
                    title="Conversion by weekday of session start", labels={"weekday": "", "conversion": "Conversion"},
                    text=[pct(v, 1) for v in wd.conversion])
        f3.update_layout(yaxis_tickformat=".0%")
        chart(f3, 340, "Pooled over all days in range. Small weekday gaps are usually noise.")
    st.dataframe(t, hide_index=True, width="stretch", height=280,
                 column_config={"date": st.column_config.DateColumn("date", format="YYYY-MM-DD"),
                                "revenue": st.column_config.NumberColumn("revenue", format="$%.2f"),
                                "revenue_7d": st.column_config.NumberColumn("revenue 7d avg", format="$%.2f"),
                                "conversion": st.column_config.NumberColumn("conversion", format="percent"),
                                "conversion_7d": st.column_config.NumberColumn("conversion 7d", format="percent")})
    st.download_button("Download daily series (CSV)", csv_bytes(t.assign(date=t.date.dt.date)), "funnelforge_daily.csv",
                       "text/csv", icon="⬇️")
    footer()


# ------------------------------------------------------------------ page 4: scenarios
def page_scenarios():
    hero("Scenario planner", "Scenario Planner",
         "What if one or more funnel steps converted better? Hypothetical arithmetic on the observed funnel — useful "
         "for prioritising tests, not a forecast.")
    if not need_data():
        return
    f = cached("funnel", lambda: funnel_detail(data))
    st.markdown("##### Improvements (absolute percentage points)")
    cols = st.columns(4)
    imp = {}
    for c, s in zip(cols, STAGES[1:]):
        base_rate = f.set_index("stage").step_rate[s]
        v = c.slider(f"→ {STAGE_LABELS[s]}", 0, 30, 5 if s == "checkout" else 0, key=f"imp_{s}",
                     help=f"Observed: {pct(base_rate)}. Capped at 100%.")
        if v:
            imp[s] = v / 100
    r = scenario_multi(data, imp)
    lo, hi = r["incremental_purchases_80"]
    kpi_row([("Baseline purchases", f"{r['baseline_purchases']:,}", f"from {r['visits']:,} sessions"),
             ("Scenario purchases", f"{r['scenario_purchases']:,.1f}", f"{r['incremental_purchases']:+,.1f}"),
             ("Illustrative extra revenue", money(r["incremental_revenue"]), f"at {money(r['aov'], 2)} average order"),
             ("80% sampling band", f"{lo:+,.1f} to {hi:+,.1f}", "extra purchases")])
    if not imp:
        note("Move a slider to try a scenario.")
    if r["capped"]:
        note("At least one improvement would push a step above 100%, so it was capped.", "warn")
    rt = r["rates"]
    fig = go.Figure()
    fig.add_bar(x=rt.transition, y=rt.baseline_rate, name="Observed", marker_color=SLATE)
    fig.add_bar(x=rt.transition, y=rt.scenario_rate, name="Scenario", marker_color=ORANGE)
    fig.update_layout(title="Step conversion: observed vs scenario", barmode="group", yaxis_tickformat=".0%",
                      legend=dict(orientation="h", y=1.1, x=0))
    c1, c2 = st.columns([1.2, 1])
    with c1:
        chart(fig, 360)
    with c2:
        sens = cached("sens", lambda: sensitivity(data, .01))
        f2 = px.bar(sens.sort_values("incremental_purchases"), x="incremental_purchases", y="transition", orientation="h",
                    color_discrete_sequence=[TEAL], title="Extra purchases from +1 point at each step",
                    labels={"incremental_purchases": "Extra purchases", "transition": ""},
                    text=[f"{v:,.1f}" for v in sens.sort_values("incremental_purchases").incremental_purchases])
        chart(f2, 360, "The same 1-point gain is worth most where the other steps multiply it most.")
    note("<b>Assumptions.</b> Traffic, every other step rate and the average order value stay as observed; improvements "
         "are absolute points, capped at 100%. The 80% band reflects only sampling noise in the observed rates, not "
         "whether the improvement is achievable. Validate any change with a controlled experiment.", "violet")
    out = rt.assign(improvement=[imp.get(s, 0.0) for s in rt.stage])
    summary = pd.DataFrame([("baseline_purchases", r["baseline_purchases"]), ("scenario_purchases", r["scenario_purchases"]),
                            ("incremental_purchases", r["incremental_purchases"]), ("incremental_purchases_p10", lo),
                            ("incremental_purchases_p90", hi), ("average_order_value", r["aov"]),
                            ("incremental_revenue", r["incremental_revenue"])], columns=["metric", "value"])
    d1, d2 = st.columns(2)
    d1.download_button("Download scenario (CSV)", csv_bytes(pd.concat([out, summary], axis=0, ignore_index=True)),
                       "funnelforge_scenario.csv", "text/csv", icon="⬇️", width="stretch")
    d2.download_button("Download sensitivity table (CSV)", csv_bytes(sens), "funnelforge_sensitivity.csv", "text/csv",
                       icon="⬇️", width="stretch")
    footer()


# ------------------------------------------------------------------ page 5: data
def page_data():
    hero("Data & exports", "Data Upload & Exports",
         "Upload a session event export, see exactly how it was cleaned and download the results. Uploads stay in "
         "this browser session and are never stored.")
    up = st.file_uploader("Session event CSV", type=["csv"], max_upload_size=MAX_UPLOAD_MB,
                          key=f"uploader_{ss.get('uploader_gen', 0)}",
                          help="One row per event: session_id, timestamp, stage; optional channel, device and revenue.")
    if up is not None:
        raw_bytes = up.getvalue()
        key = hashlib.sha256(raw_bytes).hexdigest()[:16]
        if ss.get("upload_key") != key:
            try:
                if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
                    raise ValueError(f"The file is larger than {MAX_UPLOAD_MB} MB.")
                raw = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
                clean, rep = prepare(raw)
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
                st.error(f"This file could not be used: {exc}", icon="⚠️")
            else:
                ss.update(upload_df=clean, upload_report=rep, upload_name=up.name, upload_key=key)
                ss.pop("results", None)
                ss.pop("source_choice", None)
                st.rerun()
    if "upload_df" in ss:
        c1, c2 = st.columns([3, 1])
        c1.success(f"**{esc(ss['upload_name'])}** is loaded for this session ({ss['upload_df'].session_id.nunique():,} "
                   "sessions). Switch between it and the demo under Data source.", icon="✅")
        if c2.button("Remove upload", icon="🗑️", width="stretch"):
            for k_ in ("upload_df", "upload_report", "upload_name", "upload_key", "results", "source_choice"):
                ss.pop(k_, None)
            ss["uploader_gen"] = ss.get("uploader_gen", 0) + 1
            st.rerun()
        with st.expander("Validation report", expanded=not is_demo, icon="🧾"):
            st.table(ss["upload_report"].as_frame().set_index("check"))
            for n_ in ss["upload_report"].notes:
                st.caption("• " + n_)
    with st.expander("Upload instructions, data contract and samples", icon="📄"):
        st.markdown(f"""
One row per **event**. Sessions move through `visit → product_view → add_to_cart → checkout → purchase`.

| column | meaning | accepted |
|:--|:--|:--|
| `session_id` | session identifier | also `session`, `visit_id`, `ga_session_id` |
| `timestamp` | event time | any common format; time zones converted to UTC; also `event_time`, `ts`, `date` |
| `stage` | funnel step | the five names above, or common analytics names (`session_start`, `view_item`, `Add to Cart`, `begin_checkout`, `order`) |
| `channel` *(optional)* | acquisition channel | also `utm_source`, `source`; missing → "(unknown)" |
| `device` *(optional)* | device type | also `device_type`, `platform`; missing → "(unknown)" |
| `revenue` *(optional)* | order value | on purchase events only; also `order_value`, `amount`; missing → 0 |

- Rows with a blank session, unparseable time, unknown stage or negative revenue are dropped and counted.
- Revenue on non-purchase events is set to 0. Exact duplicates are removed; a repeated stage keeps its first event.
- If channel or device changes inside a session, the first value is used.
- Sessions that skip a stage, lack a visit, or go backwards in time are excluded (counted), because their funnel is ambiguous.
- Up to 2,000,000 rows and {MAX_UPLOAD_MB} MB. Do not upload names, emails or other personal data.
""")
        s1, s2 = st.columns(2)
        sample = ROOT / "data" / "sample.csv"
        if sample.exists():
            s1.download_button("Clean sample (original, 4,000 sessions)", sample.read_bytes(), "funnelforge_sample.csv",
                               "text/csv", icon="⬇️", width="stretch")
        messy = ROOT / "sample_data" / "messy_upload_example.csv"
        if messy.exists():
            s2.download_button("Messy sample (tests the cleaning)", messy.read_bytes(), "funnelforge_messy_sample.csv",
                               "text/csv", icon="⬇️", width="stretch")
    st.markdown(f"#### Explore · {esc(source_label)}")
    if not need_data():
        return
    stg = st.multiselect("Stages", STAGES, placeholder="All stages", format_func=STAGE_LABELS.get)
    d = data[data.stage.isin(stg)] if stg else data
    kpi_row([("Events", f"{len(d):,}", "after sidebar filters"), ("Sessions", f"{d.session_id.nunique():,}", ""),
             ("Revenue", money(d.revenue.sum(), 2), "purchase events")])
    st.dataframe(d.head(1000), hide_index=True, width="stretch", height=300)
    st.caption(f"Showing the first {min(1000, len(d)):,} of {len(d):,} events. Downloads include all of them.")
    st.markdown("#### Exports")
    e = st.columns(4)
    e[0].download_button("Filtered events", csv_bytes(d), "funnelforge_events.csv", "text/csv", icon="⬇️", width="stretch")
    e[1].download_button("Funnel", csv_bytes(funnel_detail(data)), "funnelforge_funnel.csv", "text/csv", icon="⬇️",
                         width="stretch")
    e[2].download_button("Daily series", csv_bytes(daily_full(data).assign(date=lambda t: t.date.dt.date)),
                         "funnelforge_daily.csv", "text/csv", icon="⬇️", width="stretch")
    if "upload_report" in ss and not is_demo:
        e[3].download_button("Validation report", csv_bytes(ss["upload_report"].as_frame()), "funnelforge_validation.csv",
                             "text/csv", icon="⬇️", width="stretch")
    else:
        e[3].button("Validation report", disabled=True, width="stretch", help="Available when an uploaded file is selected")
    footer()


# ------------------------------------------------------------------ page 6: methodology
def page_method():
    hero("Methodology & limitations", "Methodology & Limitations",
         "How each number is computed, what the synthetic data contains, and what session-level funnel analytics can "
         "and cannot tell you.")
    st.image(str(ASSETS / "architecture.svg"), width="stretch",
             caption="Pipeline: events → preparation and validation → funnel, SQLite segments, daily trends and scenarios → exports")
    st.markdown(f"""
#### 1 · Data
- **Synthetic generator (unchanged from v1):** 4,000 sessions over 90 days from 1 Jan 2026, 5 channels, 3 devices.
  Step probabilities: product view {TRUE_STEP['product_view']:.0%}, add to cart {TRUE_STEP['add_to_cart']:.0%},
  checkout {TRUE_STEP['checkout']:.0%}, purchase {TRUE_STEP['purchase']:.0%}, each multiplied by 0.85 on Mobile.
  **Channel has no effect.** Order values are uniform between $15 and $200. It reproduces `data/sample.csv` exactly.
- Timestamps without a time zone are read as UTC.

#### 2 · Funnel
- Stage counts are distinct sessions reaching the stage. Step conversion = stage ÷ previous stage; overall = stage ÷ visits.
- 95% Wilson intervals for every rate. A zero denominator gives an undefined rate, never 0%.

#### 3 · Segments (SQLite)
- An in-memory SQLite table holds the filtered events; one `GROUP BY` query counts sessions per stage, purchases and
  revenue per channel or device (the query is shown on the page).
- Chi-square test of independence between segment and purchase; Wilson intervals; segments under
  {MIN_SEGMENT_SESSIONS} sessions flagged.

#### 4 · Daily trends
- Each session, its purchase and its revenue belong to the UTC date the session **started**. Every calendar day in range
  is shown, including days with no sessions. Rolling conversion = 7-day purchases ÷ 7-day sessions.

#### 5 · Scenarios
- Scenario purchases = sessions × product of step rates, after adding absolute points to chosen steps (capped at 100%).
  Extra revenue = extra purchases × observed average order value.
- The 80% band comes from 2,000 binomial re-draws of the observed step rates.
- With no change, the scenario reproduces the observed purchases exactly (tested).
""")
    st.markdown("""
#### Limitations
- **Synthetic data only** for every bundled result.
- **Descriptive, not causal.** Channel and device gaps are associations; visitors differ in many ways.
- **Session-level, single-touch attribution:** each session is credited to its own channel. No cross-session user
  identity, no multi-touch attribution, no repeated-stage journeys (a repeat keeps its first event).
- **Scenarios are arithmetic,** not forecasts: real changes interact with traffic, mix and order value.
- **Strict funnel order:** sessions that skip a stage are excluded rather than guessed at.
- Not validated for production decisions.
""")
    footer()


PAGES = [st.Page(page_funnel, title="Funnel Overview", icon="🔻", url_path="funnel", default=True),
         st.Page(page_segments, title="Channel & Device Comparison", icon="📊", url_path="segments"),
         st.Page(page_daily, title="Daily Trends", icon="📅", url_path="daily"),
         st.Page(page_scenarios, title="Scenario Planner", icon="🎛️", url_path="scenarios"),
         st.Page(page_data, title="Data Upload & Exports", icon="🗂️", url_path="data"),
         st.Page(page_method, title="Methodology & Limitations", icon="📘", url_path="method")]
st.navigation(PAGES).run()
