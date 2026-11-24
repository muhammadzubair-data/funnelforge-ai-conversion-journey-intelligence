"""FunnelForge AI analytics core.

Session-level e-commerce funnel analytics: forgiving event preparation, funnel and step conversion, SQLite segment
aggregation with uncertainty, daily trends attributed to the session start date, and hypothetical transition
scenarios. Everything is descriptive; nothing here estimates a causal effect.
"""
from __future__ import annotations

import re
import sqlite3
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

STAGES = ['visit', 'product_view', 'add_to_cart', 'checkout', 'purchase']
STAGE_LABELS = {'visit': 'Visit', 'product_view': 'Product view', 'add_to_cart': 'Add to cart', 'checkout': 'Checkout',
                'purchase': 'Purchase'}
CHANNELS = ['Organic', 'Paid Search', 'Social', 'Email', 'Direct']
COLUMNS = ['session_id', 'timestamp', 'channel', 'device', 'stage', 'revenue']
MAX_ROWS = 2_000_000
MIN_SEGMENT_SESSIONS = 100
SEED = 42

# Synthetic ground truth of the original generator: step probabilities, multiplied by 0.85 on Mobile.
TRUE_STEP = {'product_view': .78, 'add_to_cart': .45, 'checkout': .65, 'purchase': .72}
TRUE_MOBILE_FACTOR = .85


def true_conversion(device: str) -> float:
    f = TRUE_MOBILE_FACTOR if device == 'Mobile' else 1.0
    return float(np.prod([p * f for p in TRUE_STEP.values()]))


# ------------------------------------------------------------------------------------------------ original API
def generate(n=4000, seed=42):
    """Original synthetic generator (unchanged): channel has no effect; Mobile multiplies each step by 0.85."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        channel = rng.choice(CHANNELS)
        device = rng.choice(['Mobile', 'Desktop', 'Tablet'], p=[.6, .35, .05])
        start = pd.Timestamp('2026-01-01') + pd.Timedelta(minutes=int(rng.integers(0, 90 * 1440)))
        for j, stage in enumerate(STAGES):
            if j and rng.random() > [1, .78, .45, .65, .72][j] * (.85 if device == 'Mobile' else 1):
                break
            rows.append((f'S{i:05}', start + pd.Timedelta(minutes=j * 3), channel, device, stage,
                         round(float(rng.uniform(15, 200)), 2) if stage == 'purchase' else 0))
    return pd.DataFrame(rows, columns=COLUMNS)


def validate(df):
    """Original strict validation (same rules; vectorised, and fast on large files). Raises on the first problem."""
    if not set(COLUMNS) <= set(df.columns) or df.empty:
        raise ValueError('Required columns: ' + ', '.join(COLUMNS))
    d = df[COLUMNS].copy()
    d['timestamp'] = pd.to_datetime(d.timestamp, errors='coerce', utc=True)
    d['revenue'] = pd.to_numeric(d.revenue, errors='coerce')
    if d.isna().any().any() or not d.stage.isin(STAGES).all() or not np.isfinite(d.revenue).all() or (d.revenue < 0).any():
        raise ValueError('Invalid timestamps, stages, missing values or revenue')
    for c in ['session_id', 'channel', 'device']:
        d[c] = d[c].astype(str).str.strip()
        if d[c].eq('').any():
            raise ValueError('Blank identifiers or dimensions')
    if (d.loc[d.stage != 'purchase', 'revenue'] != 0).any():
        raise ValueError('Revenue must occur only on purchase events')
    d = d.drop_duplicates().sort_values('timestamp', kind='stable')
    g = d.groupby('session_id')
    if (g[['channel', 'device']].nunique() > 1).any().any():
        raise ValueError('Channel and device must be consistent within a session')
    if d.duplicated(['session_id', 'stage']).any():
        raise ValueError('Repeated stages are unsupported; supply one event per stage per session')
    bad = _bad_sequences(d, strict_time=True)
    if len(bad):
        raise ValueError('Sessions must contain a chronological, contiguous funnel starting with visit')
    return d


def _bad_sequences(d: pd.DataFrame, strict_time: bool) -> pd.Index:
    """Sessions whose stages are not visit, product_view, ... in order, or whose timestamps go backwards
    (or, with ``strict_time``, repeat)."""
    x = d.assign(_i=d.stage.map({s: i for i, s in enumerate(STAGES)})).sort_values(['session_id', '_i'], kind='stable')
    pos = x.groupby('session_id').cumcount()
    dt = x.groupby('session_id').timestamp.diff()
    bad_rows = (x._i != pos) | ((dt <= pd.Timedelta(0)) if strict_time else (dt < pd.Timedelta(0)))
    return pd.Index(x.loc[bad_rows, 'session_id'].unique())


def funnel(d):
    """Distinct sessions reaching each stage, step conversion (÷ previous stage), overall conversion (÷ visits)."""
    counts = d.groupby('stage').session_id.nunique().reindex(STAGES, fill_value=0)
    prev = counts.shift(1)
    return pd.DataFrame({'stage': STAGES, 'sessions': counts.values,
                         'step_rate': (counts / prev.replace(0, np.nan)).values,
                         'overall_rate': (counts / max(int(counts.iloc[0]), 1)).values,
                         'dropoff': (prev - counts).values})


def segments(d, dimension='channel'):
    """Original SQLite aggregation by channel or device (whitelisted column names only)."""
    if dimension not in ['channel', 'device']:
        raise ValueError('Unsupported dimension')
    return segment_sql(d, dimension)


def daily(d):
    """Daily sessions, purchases, revenue and conversion attributed to each session's start date (UTC)."""
    x = d.copy()
    x['date'] = x.timestamp.dt.date
    x['date'] = x.groupby('session_id')['date'].transform('min')
    return (x.groupby('date').agg(sessions=('session_id', 'nunique'), purchases=('stage', lambda s: (s == 'purchase').sum()),
                                  revenue=('revenue', 'sum')).reset_index()
            .assign(conversion=lambda t: t.purchases / t.sessions))


def scenario(d, stage, improvement):
    """Original single-transition scenario: absolute improvement (0–1) to the transition into ``stage``."""
    if stage not in STAGES[1:] or not 0 <= improvement <= 1:
        raise ValueError('Invalid scenario')
    f = funnel(d)
    rates = np.array(f.step_rate.iloc[1:].fillna(0).to_numpy(), dtype=float, copy=True)   # writable copy (pandas 3)
    idx = STAGES.index(stage) - 1
    rates[idx] = min(1, rates[idx] + improvement)
    expected = float(f.sessions.iloc[0] * np.prod(rates))
    actual = int(f.sessions.iloc[-1])
    aov = float(d.loc[d.stage == 'purchase', 'revenue'].mean()) if actual else 0
    return {'baseline_purchases': actual, 'scenario_purchases': expected, 'incremental_revenue': (expected - actual) * aov}


# ------------------------------------------------------------------------------------------------ preparation
ALIASES = {
    'session': 'session_id', 'sessionid': 'session_id', 'visit_id': 'session_id', 'ga_session_id': 'session_id',
    'session_key': 'session_id',
    'ts': 'timestamp', 'time': 'timestamp', 'datetime': 'timestamp', 'event_time': 'timestamp', 'event_timestamp': 'timestamp',
    'date': 'timestamp', 'created_at': 'timestamp',
    'source': 'channel', 'utm_source': 'channel', 'traffic_source': 'channel', 'medium': 'channel', 'acquisition_channel': 'channel',
    'device_type': 'device', 'device_category': 'device', 'platform': 'device',
    'event': 'stage', 'event_name': 'stage', 'step': 'stage', 'funnel_stage': 'stage', 'action': 'stage',
    'value': 'revenue', 'amount': 'revenue', 'order_value': 'revenue', 'purchase_value': 'revenue', 'revenue_usd': 'revenue',
}
STAGE_ALIASES = {
    'visit': 'visit', 'session_start': 'visit', 'page_view': 'visit', 'landing': 'visit', 'pageview': 'visit',
    'product_view': 'product_view', 'view_item': 'product_view', 'product': 'product_view', 'pdp_view': 'product_view',
    'view_product': 'product_view', 'product_page': 'product_view',
    'add_to_cart': 'add_to_cart', 'cart': 'add_to_cart', 'addtocart': 'add_to_cart', 'add_cart': 'add_to_cart',
    'checkout': 'checkout', 'begin_checkout': 'checkout', 'start_checkout': 'checkout', 'initiate_checkout': 'checkout',
    'purchase': 'purchase', 'order': 'purchase', 'transaction': 'purchase', 'order_complete': 'purchase', 'purchased': 'purchase',
}


@dataclass
class DataReport:
    rows_in: int = 0
    rows_out: int = 0
    sessions_out: int = 0
    renamed: dict = field(default_factory=dict)
    stages_mapped: dict = field(default_factory=dict)
    dropped_session_id: int = 0
    dropped_timestamp: int = 0
    dropped_stage: int = 0
    unknown_stage_values: list = field(default_factory=list)
    dropped_revenue: int = 0
    revenue_zeroed: int = 0
    exact_duplicates: int = 0
    repeated_stage_rows: int = 0
    inconsistent_dimension_sessions: int = 0
    invalid_sequence_sessions: int = 0
    invalid_sequence_rows: int = 0
    filled_channel: bool = False
    filled_device: bool = False
    filled_revenue: bool = False
    first: str = ''
    last: str = ''
    notes: list = field(default_factory=list)

    def as_frame(self) -> pd.DataFrame:
        rows = [('Rows in file', self.rows_in), ('Rows kept', self.rows_out), ('Sessions kept', self.sessions_out),
                ('Period (session start, UTC)', f'{self.first} → {self.last}' if self.first else '—'),
                ('Dropped: blank session ID', self.dropped_session_id),
                ('Dropped: unparseable timestamp', self.dropped_timestamp),
                ('Dropped: unknown stage', self.dropped_stage),
                ('Unknown stage values', ', '.join(self.unknown_stage_values) or '—'),
                ('Dropped: negative or non-numeric revenue', self.dropped_revenue),
                ('Revenue set to 0 on non-purchase events', self.revenue_zeroed),
                ('Removed: exact duplicate rows', self.exact_duplicates),
                ('Removed: repeated stage in a session (first kept)', self.repeated_stage_rows),
                ('Sessions with changing channel/device (first value used)', self.inconsistent_dimension_sessions),
                ('Sessions excluded: skipped a stage, no visit, or time went backwards', self.invalid_sequence_sessions),
                ('Rows removed with those sessions', self.invalid_sequence_rows),
                ('Stage names mapped', ', '.join(f'{a} → {b}' for a, b in self.stages_mapped.items()) or '—'),
                ('Channel column', 'filled with "(unknown)"' if self.filled_channel else 'provided'),
                ('Device column', 'filled with "(unknown)"' if self.filled_device else 'provided'),
                ('Revenue column', 'filled with 0' if self.filled_revenue else 'provided'),
                ('Columns renamed', ', '.join(f'{a} → {b}' for a, b in self.renamed.items()) or '—')]
        return pd.DataFrame(rows, columns=['check', 'result']).astype({'result': str})


def _norm(c) -> str:
    return re.sub(r'[^0-9a-z]+', '_', str(c).strip().lower()).strip('_')


def prepare(raw: pd.DataFrame, max_rows: int = MAX_ROWS) -> tuple[pd.DataFrame, DataReport]:
    """Forgiving preparation for uploads: maps header and stage names, drops or repairs bad rows and sessions (and
    counts them), and returns data that passes the original strict ``validate``."""
    rep = DataReport(rows_in=len(raw))
    if raw.empty:
        raise ValueError('The file has no rows.')
    if len(raw) > max_rows:
        raise ValueError(f'The file has {len(raw):,} rows; the limit for this demo is {max_rows:,}.')
    cols, used = {}, set()
    for c in raw.columns:
        n = _norm(c)
        t = n if n in COLUMNS else ALIASES.get(n)
        if t and t not in used:
            cols[c] = t
            used.add(t)
            if str(c) != t:
                rep.renamed[str(c)] = t
    missing = [c for c in ('session_id', 'timestamp', 'stage') if c not in used]
    if missing:
        raise ValueError('Missing required column(s): ' + ', '.join(missing) + '. Expected session_id, timestamp and '
                         'stage (channel, device and revenue are optional).')
    d = raw[list(cols)].rename(columns=cols).copy()
    for c, flag in (('channel', 'filled_channel'), ('device', 'filled_device')):
        if c not in d:
            d[c] = '(unknown)'
            setattr(rep, flag, True)
    if 'revenue' not in d:
        d['revenue'] = 0.0
        rep.filled_revenue = True

    sid = d.session_id.astype('string').str.strip()
    bad = sid.isna() | sid.isin(['', 'nan', 'None'])
    rep.dropped_session_id = int(bad.sum())
    d = d.assign(session_id=sid)[~bad]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ts = pd.to_datetime(d.timestamp, errors='coerce', utc=True, format='mixed')
    rep.dropped_timestamp = int(ts.isna().sum())
    d = d.assign(timestamp=ts)[ts.notna()]

    raw_stage = d.stage.astype('string').str.strip()
    mapped = raw_stage.map(lambda v: STAGE_ALIASES.get(_norm(v)) if isinstance(v, str) else None)
    for v in raw_stage[mapped.notna()].unique():
        m = STAGE_ALIASES[_norm(v)]
        if v != m and len(rep.stages_mapped) < 12:
            rep.stages_mapped[str(v)] = m
    bad = mapped.isna()
    rep.dropped_stage = int(bad.sum())
    rep.unknown_stage_values = [str(v) for v in raw_stage[bad].dropna().unique()[:8]]
    d = d.assign(stage=mapped)[~bad]

    rv = pd.to_numeric(d.revenue, errors='coerce')
    rv = rv.where(d.revenue.notna(), 0.0)
    bad = rv.isna() | ~np.isfinite(rv.fillna(0)) | (rv < 0)
    rep.dropped_revenue = int(bad.sum())
    d = d.assign(revenue=rv.astype(float))[~bad]
    nz = (d.stage != 'purchase') & (d.revenue != 0)
    rep.revenue_zeroed = int(nz.sum())
    d.loc[nz, 'revenue'] = 0.0

    for c in ('channel', 'device'):
        v = d[c].astype('string').str.strip()
        d[c] = v.where(v.notna() & (v != ''), '(unknown)').astype(str)

    before = len(d)
    d = d.drop_duplicates(subset=COLUMNS)
    rep.exact_duplicates = before - len(d)
    d = d.sort_values(['session_id', 'timestamp'], kind='stable')
    dup = d.duplicated(['session_id', 'stage'], keep='first')
    rep.repeated_stage_rows = int(dup.sum())
    d = d[~dup]

    first = d.groupby('session_id')[['channel', 'device']].transform('first')
    changed = (first != d[['channel', 'device']]).any(axis=1)
    rep.inconsistent_dimension_sessions = int(d.loc[changed, 'session_id'].nunique())
    d[['channel', 'device']] = first

    bad_s = _bad_sequences(d, strict_time=False)
    rep.invalid_sequence_sessions = int(len(bad_s))
    drop = d.session_id.isin(bad_s)
    rep.invalid_sequence_rows = int(drop.sum())
    d = d[~drop]
    # identical timestamps inside a session are nudged by one microsecond per stage so ``validate`` ordering holds
    i = d.stage.map({s: k for k, s in enumerate(STAGES)})
    d = d.assign(_i=i).sort_values(['session_id', '_i'], kind='stable')
    same = d.groupby('session_id').timestamp.diff().eq(pd.Timedelta(0))
    if same.any():
        d['timestamp'] = d.timestamp + pd.to_timedelta(np.where(same, d._i, 0), unit='us')
        rep.notes.append(f'{int(same.sum())} events shared a timestamp with the previous stage; stage order was used.')
    d = d.drop(columns='_i')[COLUMNS].sort_values('timestamp', kind='stable').reset_index(drop=True)
    if d.empty:
        raise ValueError('No valid sessions remain after cleaning. Each session needs a visit event and stages in order '
                         '(visit → product_view → add_to_cart → checkout → purchase).')
    rep.rows_out, rep.sessions_out = len(d), int(d.session_id.nunique())
    starts = d.groupby('session_id').timestamp.min()
    rep.first, rep.last = str(starts.min().date()), str(starts.max().date())
    if rep.sessions_out < 100:
        rep.notes.append('Fewer than 100 sessions: rates will be very uncertain.')
    return d, rep


# ------------------------------------------------------------------------------------------------ analytics
def wilson(k, n, alpha=.05):
    if n <= 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def funnel_detail(d: pd.DataFrame) -> pd.DataFrame:
    """Funnel with Wilson intervals for step and overall rates."""
    f = funnel(d)
    prev = f.sessions.shift(1)
    lo, hi, olo, ohi = [], [], [], []
    for k, p in zip(f.sessions, prev):
        a, b = wilson(k, p) if pd.notna(p) and p > 0 else (np.nan, np.nan)
        lo.append(a)
        hi.append(b)
        c, e = wilson(k, f.sessions.iloc[0])
        olo.append(c)
        ohi.append(e)
    return f.assign(label=f.stage.map(STAGE_LABELS), step_low=lo, step_high=hi, overall_low=olo, overall_high=ohi,
                    dropoff_rate=1 - f.step_rate)


def kpis(d: pd.DataFrame) -> dict:
    f = funnel(d)
    visits, purchases = int(f.sessions.iloc[0]), int(f.sessions.iloc[-1])
    rev = float(d.revenue.sum())
    worst = f.iloc[1:].sort_values('step_rate').iloc[0] if visits else None
    starts = d.groupby('session_id').timestamp.min()
    return {'sessions': visits, 'purchases': purchases, 'conversion': purchases / visits if visits else np.nan,
            'revenue': rev, 'aov': rev / purchases if purchases else np.nan, 'rps': rev / visits if visits else np.nan,
            'worst_stage': None if worst is None else worst.stage, 'worst_rate': None if worst is None else float(worst.step_rate),
            'first': starts.min(), 'last': starts.max(), 'days': int(starts.dt.date.nunique())}


SEGMENT_SQL = """
SELECT {dim} AS segment,
       COUNT(DISTINCT session_id) AS sessions,
       COUNT(DISTINCT CASE WHEN stage = 'product_view' THEN session_id END) AS product_view,
       COUNT(DISTINCT CASE WHEN stage = 'add_to_cart'  THEN session_id END) AS add_to_cart,
       COUNT(DISTINCT CASE WHEN stage = 'checkout'     THEN session_id END) AS checkout,
       COUNT(DISTINCT CASE WHEN stage = 'purchase'     THEN session_id END) AS purchases,
       SUM(revenue) AS revenue
FROM events
GROUP BY {dim}
ORDER BY sessions DESC
"""


def segment_sql(d: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """SQLite aggregation per segment. Column names are whitelisted; values never reach the SQL text."""
    if dimension not in ('channel', 'device'):
        raise ValueError('Unsupported dimension')
    with sqlite3.connect(':memory:') as conn:
        d[['session_id', 'channel', 'device', 'stage', 'revenue']].to_sql('events', conn, index=False)
        out = pd.read_sql_query(SEGMENT_SQL.format(dim=dimension), conn)
    out = out.rename(columns={'segment': dimension})
    out['revenue'] = out.revenue.fillna(0.0)
    out['conversion'] = out.purchases / out.sessions
    return out[[dimension, 'sessions', 'purchases', 'revenue', 'conversion', 'product_view', 'add_to_cart', 'checkout']]


def segment_compare(d: pd.DataFrame, dimension: str, alpha: float = .05) -> dict:
    """Segment conversion with Wilson intervals, step rates per segment, and a chi-square test of whether conversion
    differs between segments at all (descriptive: segments are not randomised)."""
    s = segment_sql(d, dimension)
    ci = [wilson(k, n, alpha) for k, n in zip(s.purchases, s.sessions)]
    s['conv_low'], s['conv_high'] = [c[0] for c in ci], [c[1] for c in ci]
    s['revenue_per_session'] = s.revenue / s.sessions
    s['aov'] = np.where(s.purchases > 0, s.revenue / s.purchases.where(s.purchases > 0, 1), np.nan)
    s['share'] = s.sessions / s.sessions.sum()
    s['small'] = s.sessions < MIN_SEGMENT_SESSIONS
    steps = s[[dimension, 'sessions', 'product_view', 'add_to_cart', 'checkout', 'purchases']].copy()
    rates = pd.DataFrame({dimension: s[dimension]})
    chain = ['sessions', 'product_view', 'add_to_cart', 'checkout', 'purchases']
    for a, b, name in zip(chain[:-1], chain[1:], STAGES[1:]):
        rates[name] = np.where(steps[a] > 0, steps[b] / steps[a].where(steps[a] > 0, 1), np.nan)
    tab = np.c_[s.purchases, s.sessions - s.purchases]
    tab = tab[tab.sum(1) > 0]
    if len(tab) >= 2 and (tab.sum(0) > 0).all():
        chi, p, dof, exp = stats.chi2_contingency(tab, correction=False)
        test = {'chi2': float(chi), 'p_value': float(p), 'dof': int(dof), 'min_expected': float(exp.min())}
    else:
        test = {'chi2': np.nan, 'p_value': np.nan, 'dof': 0, 'min_expected': np.nan}
    return {'table': s, 'step_rates': rates, 'test': test}


def daily_full(d: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Daily series with every calendar day present (zero-session days included) and a rolling conversion that
    divides rolling purchases by rolling sessions (not an average of daily rates)."""
    t = daily(d)
    if t.empty:
        return t
    t['date'] = pd.to_datetime(t.date)
    full = pd.date_range(t.date.min(), t.date.max(), freq='D')
    t = t.set_index('date').reindex(full).fillna({'sessions': 0, 'purchases': 0, 'revenue': 0.0}).rename_axis('date').reset_index()
    t['conversion'] = np.where(t.sessions > 0, t.purchases / t.sessions.where(t.sessions > 0, 1), np.nan)
    rs, rp = t.sessions.rolling(window, min_periods=1).sum(), t.purchases.rolling(window, min_periods=1).sum()
    t[f'conversion_{window}d'] = np.where(rs > 0, rp / rs.where(rs > 0, 1), np.nan)
    t[f'revenue_{window}d'] = t.revenue.rolling(window, min_periods=1).mean()
    t['weekday'] = t.date.dt.day_name()
    for c in ('sessions', 'purchases'):
        t[c] = t[c].astype(int)
    return t


def scenario_multi(d: pd.DataFrame, improvements: dict, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Hypothetical funnel with absolute improvements (0–1) to one or more transitions, each capped at 100%.
    Assumes unchanged traffic, other transition rates and average order value. The 80% band reflects sampling
    noise in the observed rates only (binomial bootstrap), not uncertainty about whether the improvement is
    achievable."""
    for s, v in improvements.items():
        if s not in STAGES[1:] or not 0 <= v <= 1:
            raise ValueError('Invalid scenario')
    f = funnel(d)
    visits = int(f.sessions.iloc[0])
    counts = f.sessions.to_numpy()
    base = np.array(f.step_rate.iloc[1:].fillna(0).to_numpy(), dtype=float, copy=True)
    new = base.copy()
    for s, v in improvements.items():
        k = STAGES.index(s) - 1
        new[k] = min(1.0, new[k] + v)
    actual = int(counts[-1])
    purchase_rev = d.loc[d.stage == 'purchase', 'revenue']
    aov = float(purchase_rev.mean()) if actual else 0.0
    expected = visits * float(np.prod(new))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        r = np.array([rng.binomial(int(n), p) / n if n > 0 else 0.0 for n, p in zip(counts[:-1], base)])
        rr = r.copy()
        for s, v in improvements.items():
            k = STAGES.index(s) - 1
            rr[k] = min(1.0, rr[k] + v)
        draws[b] = visits * (np.prod(rr) - np.prod(r))
    stage_rows = pd.DataFrame({'transition': [f'{STAGE_LABELS[a]} → {STAGE_LABELS[b]}' for a, b in zip(STAGES[:-1], STAGES[1:])],
                               'stage': STAGES[1:], 'baseline_rate': base, 'scenario_rate': new})
    return {'visits': visits, 'baseline_purchases': actual, 'scenario_purchases': expected,
            'incremental_purchases': expected - actual, 'aov': aov,
            'incremental_revenue': (expected - actual) * aov,
            'incremental_purchases_80': (float(np.percentile(draws, 10)), float(np.percentile(draws, 90))),
            'rates': stage_rows, 'capped': bool((base + np.array([improvements.get(s, 0) for s in STAGES[1:]]) > 1).any())}


def sensitivity(d: pd.DataFrame, points: float = .01) -> pd.DataFrame:
    """Incremental purchases and revenue from the same absolute improvement applied to each transition in turn.
    A +1-point gain is worth most where the other transitions multiply it the most."""
    rows = []
    for s in STAGES[1:]:
        r = scenario_multi(d, {s: points}, n_boot=200)
        rows.append({'stage': s, 'transition': r['rates'].set_index('stage').loc[s, 'transition'],
                     'baseline_rate': float(r['rates'].set_index('stage').loc[s, 'baseline_rate']),
                     'incremental_purchases': r['incremental_purchases'], 'incremental_revenue': r['incremental_revenue']})
    return pd.DataFrame(rows).sort_values('incremental_purchases', ascending=False).reset_index(drop=True)


def stage_reach_by_segment(d: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Long table: share of each segment's sessions reaching each stage (for a segmented funnel chart)."""
    s = segment_sql(d, dimension)
    out = []
    for _, r in s.iterrows():
        for stage, col in zip(STAGES, ['sessions', 'product_view', 'add_to_cart', 'checkout', 'purchases']):
            out.append({dimension: r[dimension], 'stage': STAGE_LABELS[stage], 'sessions': int(r[col]),
                        'reach': r[col] / r.sessions if r.sessions else np.nan})
    return pd.DataFrame(out)
