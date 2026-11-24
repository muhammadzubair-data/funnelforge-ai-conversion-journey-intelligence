import numpy as np
import pandas as pd
import sqlite3
STAGES=['visit','product_view','add_to_cart','checkout','purchase']
CHANNELS=['Organic','Paid Search','Social','Email','Direct']

def generate(n=4000,seed=42):
    rng=np.random.default_rng(seed); rows=[]
    for i in range(n):
        channel=rng.choice(CHANNELS); device=rng.choice(['Mobile','Desktop','Tablet'],p=[.6,.35,.05]); start=pd.Timestamp('2026-01-01')+pd.Timedelta(minutes=int(rng.integers(0,90*1440)))
        for j,stage in enumerate(STAGES):
            if j and rng.random()>[1,.78,.45,.65,.72][j]*(.85 if device=='Mobile' else 1): break
            rows.append((f'S{i:05}',start+pd.Timedelta(minutes=j*3),channel,device,stage,round(float(rng.uniform(15,200)),2) if stage=='purchase' else 0))
    return pd.DataFrame(rows,columns=['session_id','timestamp','channel','device','stage','revenue'])

def validate(df):
    cols=['session_id','timestamp','channel','device','stage','revenue']
    if not set(cols)<=set(df.columns) or df.empty: raise ValueError('Required columns: '+', '.join(cols))
    d=df[cols].copy(); d['timestamp']=pd.to_datetime(d.timestamp,errors='coerce',utc=True); d['revenue']=pd.to_numeric(d.revenue,errors='coerce')
    if d.isna().any().any() or not d.stage.isin(STAGES).all() or not np.isfinite(d.revenue).all() or (d.revenue<0).any(): raise ValueError('Invalid timestamps, stages, missing values or revenue')
    for c in ['session_id','channel','device']:
        d[c]=d[c].astype(str).str.strip()
        if d[c].eq('').any(): raise ValueError('Blank identifiers or dimensions')
    if (d.loc[d.stage!='purchase','revenue']!=0).any(): raise ValueError('Revenue must occur only on purchase events')
    d=d.drop_duplicates().sort_values('timestamp'); g=d.groupby('session_id')
    if (g[['channel','device']].nunique()>1).any().any(): raise ValueError('Channel and device must be consistent within a session')
    if d.duplicated(['session_id','stage']).any(): raise ValueError('Repeated stages are unsupported; supply one event per stage per session')
    for _,s in g:
        if s.stage.tolist()!=STAGES[:len(s)] or s.timestamp.duplicated().any(): raise ValueError('Sessions must contain a chronological, contiguous funnel starting with visit')
    return d

def funnel(d):
    counts=d.groupby('stage').session_id.nunique().reindex(STAGES,fill_value=0); prev=counts.shift(1); return pd.DataFrame({'stage':STAGES,'sessions':counts.values,'step_rate':(counts/prev.replace(0,np.nan)).values,'overall_rate':(counts/max(int(counts.iloc[0]),1)).values,'dropoff':(prev-counts).values})

def segments(d,dimension='channel'):
    if dimension not in ['channel','device']: raise ValueError('Unsupported dimension')
    with sqlite3.connect(':memory:') as conn:
        d.to_sql('events',conn,index=False)
        return pd.read_sql_query(f'''SELECT {dimension}, COUNT(DISTINCT session_id) sessions, COUNT(DISTINCT CASE WHEN stage='purchase' THEN session_id END) purchases, SUM(revenue) revenue, 1.0*COUNT(DISTINCT CASE WHEN stage='purchase' THEN session_id END)/COUNT(DISTINCT session_id) conversion FROM events GROUP BY {dimension} ORDER BY sessions DESC''',conn)

def daily(d):
    x=d.copy(); x['date']=x.timestamp.dt.date
    # Attribute conversions and revenue to the session start date.
    x['date']=x.groupby('session_id')['date'].transform('min')
    return x.groupby('date').agg(sessions=('session_id','nunique'),purchases=('stage',lambda s:(s=='purchase').sum()),revenue=('revenue','sum')).reset_index().assign(conversion=lambda t:t.purchases/t.sessions)

def scenario(d,stage,improvement):
    if stage not in STAGES[1:] or not 0<=improvement<=1: raise ValueError('Invalid scenario')
    f=funnel(d); rates=f.step_rate.iloc[1:].fillna(0).to_numpy(); idx=STAGES.index(stage)-1; rates[idx]=min(1,rates[idx]+improvement); expected=float(f.sessions.iloc[0]*np.prod(rates)); actual=int(f.sessions.iloc[-1]); aov=float(d.loc[d.stage=='purchase','revenue'].mean()) if actual else 0
    return {'baseline_purchases':actual,'scenario_purchases':expected,'incremental_revenue':(expected-actual)*aov}
