import json
from pathlib import Path
import numpy as np
import pandas as pd
from strict_platform_scan import base

ASOF="2026-11-27"
def first_approach(gap, close_gap, start, near, rebound):
    # Intraday extreme is authoritative; a previous touch or distinct reversal invalidates the setup.
    g=np.asarray(gap[start:]); c=np.asarray(close_gap[start:])
    if len(g)==0 or np.min(g)<=0 or not 0<g[-1]<=near or not 0<c[-1]<=near:return False
    idx=np.flatnonzero(g<=near)
    if not len(idx):return False
    f=int(idx[0])
    if f<len(g)-3:return False
    if np.any(g[f:]>near):return False
    if g[-1]>np.min(g[f:])+rebound or c[-1]>np.min(c[f:])+rebound:return False
    return True

def candle(a,dates,ticker):
    o,h,l,c,v=a.T;n=len(a);hits=[]
    for i in range(n-10,n-1):
        vr=v[i]/max(1,np.mean(v[i-20:i]))
        br=abs(c[i]-o[i])/max(1e-8,np.mean(abs(c[i-20:i]-o[i-20:i])))
        if max(vr,br)<2:continue
        u,d=h[i],l[i];w=u-d
        if w<=0 or max(h[i+1:])>u or min(l[i+1:])<d:continue
        for side,gap,cgap in [("upper",(u-h)/w,(u-c)/w),("lower",(l-d)/w,(c-d)/w)]:
            if not first_approach(gap,cgap,i+1,.25,.05):continue
            hits.append(dict(ticker=ticker,side=side,event=dates[i],volume_ratio=round(vr,3),body_ratio=round(br,3),upper=round(u,4),lower=round(d,4),close=round(c[-1],4),edge_gap=round(gap[-1]*100,2),close_gap=round(cgap[-1]*100,2),following_ohlc=a[i+1:,:4].round(4).tolist(),event_ohlc=a[i,:4].round(4).tolist()))
    return sorted(hits,key=lambda x:x["close_gap"])[:1]

def platform(a,dates,ticker):
    n=len(a);o,h,l,c,v=a.T;hits=[]
    for days in [60,75,90,105,120,150,180,210,240]:
        for end in range(max(days,n-85),n-5,4):
            p=base(a[end-days:end,:4])
            if not p:continue
            u=p["upper"]
            if not 0<=c[-1]/u-1<=.03:continue
            b=next((j for j in range(end,min(end+15,n-1)) if c[j]>u*1.015 and c[j+1]>u*1.015),None)
            if b is None:continue
            depart=next((j for j in range(b,n) if l[j]>u*1.025 and h[j]>=u*1.04),None)
            if depart is None or depart>=n-1:continue
            if np.min(l[depart+1:])<u*.99 or np.min(c[depart+1:])<u:continue
            gap=(l-u)/u;cgap=(c-u)/u
            # First return after a confirmed departure; do not reset at a later peak.
            near=np.flatnonzero(gap[depart+1:]<=.025)
            if not len(near):continue
            f=depart+1+int(near[0])
            if f<n-3 or np.any(gap[f:]>.025):continue
            if gap[-1]>np.min(gap[f:])+.005 or cgap[-1]>np.min(cgap[f:])+.005:continue
            hits.append(dict(ticker=ticker,upper=round(u,4),lower=round(p["lower"],4),base_start=dates[end-days],base_end=dates[end-1],base_days=days,high_tests=p["highs"],low_tests=p["lows"],breakout=dates[b],departure=dates[depart],first_touch=dates[f],close=round(c[-1],4),distance_pct=round(cgap[-1]*100,2),score=p["score"],recent_ohlc=a[-10:,:4].round(4).tolist()))
    return sorted(hits,key=lambda x:-x["score"])[:1]

def tests():
    assert first_approach([.8,.5,.3,.2,.1],[.8,.6,.4,.23,.2],0,.25,.05)
    assert not first_approach([.8,.1,.5,.2],[.9,.2,.6,.24],0,.25,.05)
    assert not first_approach([.8,0,.1],[.9,.1,.2],0,.25,.05)
    assert not first_approach([.8,.05,.2],[.9,.1,.24],0,.25,.05)
    assert not first_approach([.8,.3,.2],[.9,.4,.5],0,.25,.05)
    print("QA_TESTS_PASS 5",flush=True)

def main():
    tests()
    hits={"platform":[],"candle":[]}; errors=[];valid=0
    for f in sorted(Path("ohlcv_cache").glob("*.parquet")):
        try:
            df=pd.read_parquet(f).sort_index()
            df.columns=[str(x).lower() for x in df.columns]
            df=df[["open","high","low","close","volume"]].dropna()
            dates=df.index.strftime("%Y-%m-%d").tolist()
            if len(df)<245 or dates[-1]!=ASOF:raise ValueError("stale or short history")
            a=df.to_numpy(float)
            if not np.isfinite(a).all() or np.any(a[:,1]<np.maximum(a[:,0],a[:,3])) or np.any(a[:,2]>np.minimum(a[:,0],a[:,3])):raise ValueError("invalid OHLC")
            valid+=1
            hits["candle"].extend(candle(a,dates,f.stem))
            hits["platform"].extend(platform(a,dates,f.stem))
        except Exception as e:errors.append([f.stem,str(e)])
    result=dict(asof=ASOF,valid=valid,errors=errors,**hits)
    Path("qa_results.json").write_text(json.dumps(result,indent=2))
    print("QA_RESULT",json.dumps(result),flush=True)
    assert valid>0
if __name__=="__main__":main()
