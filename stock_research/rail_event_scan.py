import json, collections
import numpy as np
import pandas as pd
import yfinance as yf
from strict_platform_scan import BASE
from scan_midcap import universe, screen as platform_screen
from pathlib import Path
ASOF="2026-11-27"

def screen(t,df):
    df=df.copy()
    df.columns=[str(c).lower() for c in df.columns]
    df=df[["open","high","low","close","volume"]].dropna().sort_index()
    df=df[~df.index.duplicated()].loc[:ASOF]
    if len(df)<245 or str(df.index[-1])[:10]!=ASOF:return None,"data"
    a=df.to_numpy(float);n=len(a);o,h,l,c,v=a.T
    body=abs(c-o)
    events=[]
    for i in range(n-10,n):
        vr=v[i]/max(1,np.mean(v[i-20:i]))
        br=body[i]/max(.0001,np.mean(body[i-20:i]))
        rr=(h[i]-l[i])/max(.0001,np.mean(h[i-20:i]-l[i-20:i]))
        if max(vr,br)>=2.0:events.append((i,vr,br,rr))
    if not events:return None,"no_event"
    hits=[];intact=False
    # The large candle itself defines the short-term price interval.
    # No historic platform, breakout or pivot test is required.
    for i,vr,br,rr in events:
        if i==n-1:continue
        u,d=h[i],l[i];width=u-d
        if width<=0:continue
        if max(h[i+1:])>u or min(l[i+1:])<d:continue
        intact=True
        for side,rail in [("upper",u),("lower",d)]:
            distance=(u-c)/width if side=="upper" else (c-d)/width
            edge=(u-h)/width if side=="upper" else (l-d)/width
            if not 0<distance[-1]<=.25:continue
            if min(edge[i+1:])<=0:continue
            near=distance[i+1:]<=.25
            first=i+1+int(np.flatnonzero(near)[0])
            if first<n-3:continue
            if not np.all(distance[first:]<=.25):continue
            if first==i+1:
                if distance[i]<=.25:continue
            elif distance[first-1]<=.25:continue
            if distance[-1]-min(distance[first:])>.08:continue
            hits.append(dict(ticker=t,side=side,rail=round(rail,2),close=round(c[-1],2),distance_pct=round(abs(c[-1]/rail-1)*100,2),range_distance_pct=round(distance[-1]*100,2),event=str(df.index[i])[:10],volume_ratio=round(vr,2),body_ratio=round(br,2),range_ratio=round(rr,2),first_approach=str(df.index[first])[:10],upper=round(u,2),lower=round(d,2),days_after_event=n-1-i,score=max(vr,br)))
    return (min(hits,key=lambda x:(x["range_distance_pct"],-x["score"])),"match") if hits else (None,"not_first_approach" if intact else "event_range_broken")

def main():
    names=sorted({str(t).replace(".","-") for t in set(pd.read_csv(BASE+"/manifest.csv")["ticker"].astype(str))|set(universe())})
    hits=[];failed=[];counts=collections.Counter();audit={}
    platforms=[];platform_errors=[];platform_scanned=0
    cache=Path("ohlcv_cache");cache.mkdir(exist_ok=True)
    for k in range(0,len(names),30):
        batch=names[k:k+30]
        try:
            data=yf.download(batch,start="2025-09-01",end="2026-11-28",auto_adjust=True,group_by="ticker",threads=True,progress=False,timeout=25)
            for t in batch:
                try:
                    raw=data[t].dropna(subset=["Open","High","Low","Close","Volume"]).sort_index()
                    if len(raw)>=245 and str(raw.index[-1])[:10]==ASOF:
                        raw.to_parquet(cache/(t+".parquet"))
                        try:
                            ph=platform_screen(t,raw)
                            platform_scanned+=1
                            platforms.extend(ph)
                        except Exception as pe:platform_errors.append({"ticker":t,"error":str(pe)})
                    hit,status=screen(t,raw)
                    counts[status]+=1;audit[t]=status
                    if hit:hits.append(hit)
                    if status=="data":failed.append(t)
                except Exception as e:failed.append(t);counts["error"]+=1;audit[t]=str(e)
        except Exception:failed.extend(batch);counts["error"]+=len(batch)
        print("PROGRESS",k+len(batch),len(names),dict(counts),flush=True)
    out=dict(asof=ASOF,universe=len(names),scanned=len(names)-len(failed),failed=failed,counts=dict(counts),matches=hits,regression={t:audit.get(t) for t in ["CLF","REXR","APH","CI","ELV"]})
    open("rail_event_results.json","w").write(json.dumps(dict(out,audit=audit),indent=2))
    print("RAIL_RESULT",json.dumps(out),flush=True)
    pout=dict(asof=ASOF,universe=len(names),scanned=platform_scanned,matches=platforms,errors=platform_errors)
    open("combined_platform_results.json","w").write(json.dumps(pout,indent=2))
    print("PLATFORM_RESULT",json.dumps(pout),flush=True)
if __name__=="__main__":main()
