import pytest
from core import *
@pytest.fixture
def d(): return validate(generate(200))
def test_reproducible(): assert generate(10).equals(generate(10))
def test_funnel(d): assert funnel(d).sessions.is_monotonic_decreasing
def test_sessions(d): assert funnel(d).sessions.iloc[0]==200
def test_segments(d): assert segments(d).sessions.sum()==200
def test_revenue(d): assert segments(d).revenue.sum()==pytest.approx(d.revenue.sum())
def test_daily(d): assert daily(d).sessions.sum()==200
def test_zero(d): assert scenario(d,'checkout',0)['scenario_purchases']==pytest.approx(funnel(d).sessions.iloc[-1])
def test_improve(d): assert scenario(d,'checkout',.1)['incremental_revenue']>=-1e-8
def test_invalid(d):
    with pytest.raises(ValueError): validate(d.drop(columns='stage'))
def test_negative(d):
    d.loc[d.index[0],'revenue']=-1
    with pytest.raises(ValueError): validate(d)
def test_bad_stage(d):
    d.loc[d.index[0],'stage']='unknown'
    with pytest.raises(ValueError): validate(d)
