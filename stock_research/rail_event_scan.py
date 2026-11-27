import concurrent.futures,json,time
import numpy as np
import pandas as pd
import yfinance as yf
from strict_platform_scan import base,BASE
from scan_midcap import universe
ASOF="2026-11-25"
def screen(t,df):
    df=df.copy()
    df.columns=[str(c).lower() for c in df.columns]
    cols=["open","high","low","close","volume"]
    df=df[cols].dropna().sort_index()
    df=df[~df.index.duplicated()]
    df=df.loc[:ASOF]
    if len(df)<245 or str(df.index[-1])[:10]!=ASOF:return None,False
    a=df.to_numpy(float);n=len(a);c=a[:,3];body=abs(c-a[:,0]);v=a[:,4]
    events=[]
    for i in range(n-10,n-1):
        vr=v[i]/max(1,np.mean(v[i-20:i]))
        br=body[i]/max(.0001,np.mean(body[i-20:i]))
        if vr>=1.8 or br>=2:events.append((i,vr,br))
    if not events:return None,True
    hits=[]
    for days in [60,75,90,105,120,150,180,210,240]:
        for end in range(max(days,n-30),n-9,3):
            p=base(a[end-days:end,:4])
            if not p:continue
            for side,rail in [("upper",p["upper"]),("lower",p["lower"])]:
                dist=abs(c[-1]/rail-1)
                if dist>.03:continue
                near=(abs(c/rail-1)<=.03)|((a[:,2]<=rail*1.01)&(a[:,1]>=rail*.99))
                for i,vr,br in events:
                    if i<end or near[i]:continue
                    touch=np.flatnonzero(near[i+1:])
                    if not len(touch):continue
                    first=i+1+int(touch[0])
                    if not near[-1] or not np.all(near[first:]):continue
                    if first<n-4:continue
                    if side=="upper" and c[i]>rail*1.03:continue
                    if side=="lower" and c[i]<rail*.97:continue
                    hits.append(dict(ticker=t,side=side,rail=round(rail,2),close=round(c[-1],2),distance_pct=round(dist*100,2),event=str(df.index[i])[:10],volume_ratio=round(vr,2),body_ratio=round(br,2),first_touch=str(df.index[first])[:10],base_days=days,base_start=str(df.index[end-days])[:10],base_end=str(df.index[end-1])[:10],upper=round(p["upper"],2),lower=round(p["lower"],2),score=p["score"]))
    return max(hits,key=lambda h:h["score"]) if hits else None,True
def main():
    names=sorted(set(pd.read_csv(BASE+"/manifest.csv")["ticker"].astype(str))|set(universe()))
    hits=[];failed=[];scanned=0
    for k in range(0,len(names),30):
        batch=names[k:k+30]
        try:
            data=yf.download(batch,start="2025-09-01",end="2026-11-26",auto_adjust=True,group_by="ticker",threads=True,progress=False,timeout=25)
            for t in batch:
                try:
                    h,ok=screen(t,data[t])
                    if ok:scanned+=1
                    else:failed.append(t)
                    if h:hits.append(h)
                except Exception:failed.append(t)
        except Exception:failed.extend(batch)
        print("PROGRESS",k+len(batch),len(names),flush=True)
    out=dict(asof=ASOF,universe=len(names),scanned=scanned,failed=failed,matches=hits)
    open("rail_event_results.json","w").write(json.dumps(out,indent=2))
    print("RAIL_RESULT",json.dumps(out),flush=True)
if __name__=="__main__":main()
