import json, collections
import numpy as np
import pandas as pd
import yfinance as yf
from strict_platform_scan import BASE
from scan_midcap import universe
ASOF="2026-11-25"

def rails(w):
    # Independent horizontal rails, requiring repeated separated tests on both sides.
    h,l,c=w[:,1],w[:,2],w[:,3]
    n=len(w)
    tops=[i for i in range(3,n-3) if h[i]>=max(h[i-3:i+4])]
    bots=[i for i in range(3,n-3) if l[i]<=min(l[i-3:i+4])]
    def levels(p,x):
        for i in p:
            ids=[j for j in p if abs(x[j]/x[i]-1)<=.025]
            separated=[]
            for j in ids:
                if not separated or j-separated[-1]>=8:separated.append(j)
            if len(separated)>=2 and separated[-1]-separated[0]>=n*.4:
                yield float(np.median(x[separated])),separated
    for u,hi in levels(tops,h):
        for d,lo in levels(bots,l):
            width=u-d
            if d<=0 or not .04<=width/d<=.40:continue
            if np.mean((c>=d*.99)&(c<=u*1.01))<.95:continue
            if max(h)>u*1.04 or min(l)<d*.96:continue
            ev=sorted([(i,1) for i in hi]+[(i,0) for i in lo])
            alt=sum(x[1]!=y[1] for x,y in zip(ev,ev[1:]))
            if alt<3:continue
            if abs(np.median(c[:15])-np.median(c[-15:]))>.65*width:continue
            yield u,d,len(hi)+len(lo)+alt

def screen(t,df):
    df=df.copy()
    df.columns=[str(c).lower() for c in df.columns]
    df=df[["open","high","low","close","volume"]].dropna().sort_index()
    df=df[~df.index.duplicated()].loc[:ASOF]
    if len(df)<245 or str(df.index[-1])[:10]!=ASOF:return None,"data"
    a=df.to_numpy(float);n=len(a);o,h,l,c,v=a.T
    body=abs(c-o);tr=np.maximum(h-l,np.maximum(abs(h-np.roll(c,1)),abs(l-np.roll(c,1))))
    events={}
    for i in range(n-10,n):
        ratios=[v[i]/max(1,np.mean(v[i-20:i])),body[i]/max(.0001,np.mean(body[i-20:i])),tr[i]/max(.0001,np.mean(tr[i-20:i]))]
        if max(ratios)>=1.5:events[i]=ratios
    if not events:return None,"no_event"
    hits=[];found=False;valid=False;nearby=False
    for end in [n-10]:
        for days in [60,75,90,105,120,150,180,210,240]:
            if end<days:continue
            for u,d,score in rails(a[end-days:end]):
                found=True
                # Validate the entire continuation, not only the event day.
                if max(h[n-10:])>u or min(l[n-10:])<d:continue
                if max(c[n-10:])>u or min(c[n-10:])<d:continue
                valid=True
                for side,rail in [("upper",u),("lower",d)]:
                    signed=(rail-c)/rail if side=="upper" else (c-rail)/rail
                    band=min(.04,(u-d)/rail*.25)
                    if not 0<=signed[-1]<=band:continue
                    nearby=True
                    # Entire last ten sessions inside the channel; no actual rail touch.
                    edge=(rail-h)/rail if side=="upper" else (l-rail)/rail
                    if np.min(edge[-10:])<=0:continue
                    if signed[-1]<=0:continue
                    # First approach to proximity band, not a completed touch or rebound.
                    near=signed[-10:]<=band
                    first=n-10+int(np.flatnonzero(near)[0])
                    if first<n-3 or first==n-10:continue
                    if not np.all(near[first-(n-10):]):continue
                    if signed[-1]>signed[first-1]:continue
                    if signed[-1]-np.min(signed[first:])>min(.005,band*.25):continue
                    # Event is independent of approach direction and timing.
                    i,ratios=max(events.items(),key=lambda item:max(item[1]))
                    hits.append(dict(ticker=t,side=side,rail=round(rail,2),close=round(c[-1],2),distance_pct=round(signed[-1]*100,2),event=str(df.index[i])[:10],volume_ratio=round(ratios[0],2),body_ratio=round(ratios[1],2),range_ratio=round(ratios[2],2),first_approach=str(df.index[first])[:10],base_days=days,base_start=str(df.index[end-days])[:10],base_end=str(df.index[end-1])[:10],upper=round(u,2),lower=round(d,2),score=score))
    return (max(hits,key=lambda x:x["score"]),"match") if hits else (None,"first_approach_rejected" if nearby else "not_near" if valid else "invalid_continuation" if found else "no_platform")

def main():
    names=sorted({str(t).replace(".","-") for t in set(pd.read_csv(BASE+"/manifest.csv")["ticker"].astype(str))|set(universe())})
    hits=[];failed=[];counts=collections.Counter();audit={}
    for k in range(0,len(names),30):
        batch=names[k:k+30]
        try:
            data=yf.download(batch,start="2025-09-01",end="2026-11-26",auto_adjust=True,group_by="ticker",threads=True,progress=False,timeout=25)
            for t in batch:
                try:
                    hit,status=screen(t,data[t])
                    counts[status]+=1;audit[t]=status
                    if hit:hits.append(hit)
                    if status=="data":failed.append(t)
                except Exception as e:failed.append(t);counts["error"]+=1;audit[t]=str(e)
        except Exception:failed.extend(batch);counts["error"]+=len(batch)
        print("PROGRESS",k+len(batch),len(names),dict(counts),flush=True)
    out=dict(asof=ASOF,universe=len(names),scanned=len(names)-len(failed),failed=failed,counts=dict(counts),matches=hits,regression={t:audit.get(t) for t in ["CLF","REXR","APH","CI","ELV"]})
    open("rail_event_results.json","w").write(json.dumps(dict(out,audit=audit),indent=2))
    print("RAIL_RESULT",json.dumps(out),flush=True)
if __name__=="__main__":main()
