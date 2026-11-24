import streamlit as st
import pandas as pd
import plotly.express as px
from core import generate,validate,funnel,segments,daily,scenario,STAGES
st.set_page_config(page_title='FunnelForge AI | ZubairLabs',layout='wide')
st.title('FunnelForge AI'); st.caption('ZubairLabs • E-commerce conversion intelligence')
@st.cache_data
def sample(): return validate(generate())
u=st.sidebar.file_uploader('Session event CSV',type='csv')
try: d=validate(pd.read_csv(u)) if u else sample()
except ValueError as e: st.error(str(e)); st.stop()
st.sidebar.download_button('Sample CSV',sample().to_csv(index=False),'sample.csv')
for c in ['channel','device']:
    options=sorted(d[c].unique()); selected=st.sidebar.multiselect(c.title(),options,default=options); d=d[d[c].isin(selected)]
if d.empty: st.warning('No sessions match filters.'); st.stop()
f=funnel(d); n=int(f.sessions.iloc[0]); p=int(f.sessions.iloc[-1]); rev=d.revenue.sum()
for col,label,value in zip(st.columns(4),['Sessions','Purchases','Conversion','Revenue (USD)'],[f'{n:,}',f'{p:,}',f'{p/n:.1%}',f'${rev:,.2f}']): col.metric(label,value)
a,b,c,e=st.tabs(['Conversion funnel','Channel & device','Daily trends','Scenario planner'])
with a:
    st.plotly_chart(px.funnel(f,x='sessions',y='stage'),use_container_width=True); st.dataframe(f); st.download_button('Export funnel',f.to_csv(index=False),'funnel.csv')
with b:
    dim=st.radio('Group by',['channel','device']); s=segments(d,dim); st.plotly_chart(px.bar(s,x=dim,y='conversion',color=dim),use_container_width=True); st.dataframe(s); st.download_button('Export segments',s.to_csv(index=False),'segments.csv')
with c:
    t=daily(d); st.plotly_chart(px.line(t,x='date',y=['sessions','purchases']),use_container_width=True); st.dataframe(t)
with e:
    stage=st.selectbox('Improve transition into',STAGES[1:]); delta=st.slider('Absolute improvement (percentage points)',0,30,5)/100; r=scenario(d,stage,delta); st.metric('Scenario purchases',f"{r['scenario_purchases']:.1f}"); st.metric('Illustrative incremental revenue',f"${r['incremental_revenue']:,.2f}"); st.info('Assumes unchanged traffic, other transition rates and average order value. Not a causal forecast.')
st.caption('Synthetic demonstration unless a CSV is uploaded. Session-level attribution is descriptive, not causal.')
