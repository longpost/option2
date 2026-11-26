#!/usr/bin/env python3
"""Public S&P500/Nasdaq100 daily OHLC scan, strict FIRST valid upper-rail retest."""
import concurrent.futures, io, json, time, traceback
import numpy as np
import pandas as pd
import requests

BASE="https://raw.githubusercontent.com/floyds1995/Auto-Index-Constituents-Tracker/main/us_data/raw_ohlcv"
ASOF=pd.Timestamp("2026-11-25")
session=requests.Session()
session.headers.update({"User-Agent":"research-platform-screen/1.0"})

def median(x): return float(np.median(x))
def pivots(a,col,high):
    out=[]
    for i in range(4,len(a)-4):
        y=a[i,col]
        if (y>=np.max(a[i-4:i+5,col]) if high else y<=np.min(a[i-4:i+5,col])):
            out.append((i,float(y)))
    return out
def separate(points):
    out=[]
    for x in points:
        if not out or x[0]-out[-1][0]>=10: out.append(x)
    return out
def slope(p):
    x=np.array([v[0] for v in p],float)
    y=np.array([v[1] for v in p],float)
    return float(np.polyfit(x,y,1)[0])
def base(w):
    n=len(w); tops=pivots(w,1,True); bottoms=pivots(w,2,False)
    c=w[:,3]; best=None
    if len(tops)<3 or len(bottoms)<3:return None
    for _,level in tops:
        H=separate([p for p in tops if abs(p[1]/level-1)<=.018])
        if len(H)<3 or H[0][0]>.40*n or H[-1][0]<.66*n:continue
        U=median([p[1] for p in H])
        if abs(slope(H)*n/U)>.04:continue
        # Do not include an earlier breakout inside the putative platform.
        if (c>U*1.025).mean()>.03 or np.max(w[:,1])>U*1.055:continue
        L=separate([p for p in bottoms if p[1]<U*.96])
        if len(L)<3 or L[0][0]>.42*n or L[-1][0]<.65*n:continue
        D=median([p[1] for p in L]);width=U-D
        if D<=0 or not .055<=width/D<=.23:continue
        if abs(median(c[:20])-median(c[-20:]))>.25*width:continue
        if (c<D-.20*width).mean()>.06:continue
        roll=[median(c[i:i+20]) for i in range(0,n-19,5)]
        if (max(roll)-min(roll))/width>.65:continue
        drift=slope(L)*n/U
        loc=np.array([x for x,_ in L]);lev=np.array([x for _,x in L]);mid=median(loc)
        first=median(lev[loc<=mid]);last=median(lev[loc>=mid])
        rectangle=(abs(drift)<=.05 and abs((last-first)/U)<=.05)
        triangle=(.015<=drift<=.15 and last>first and last<U*.98)
        if not (rectangle or triangle):continue
        events=sorted([(i,1,v) for i,v in H]+[(i,0,v) for i,v in L])
        alternate=0;large=0
        for x,y in zip(events,events[1:]):
            if x[1]!=y[1]:
                alternate+=1
                if abs(x[2]-y[2])>=.5*width:large+=1
        if alternate<4 or large<4:continue
        score=3*len(H)+2*len(L)+alternate
        if best is None or score>best["score"]:
            best=dict(upper=U,lower=D,highs=len(H),lows=len(L),
                      kind="rectangle" if rectangle else "flat_top_triangle",score=score)
    return best
def analyze(symbol):
    url=f"{BASE}/parquet/{symbol}.parquet"
    data=None
    for attempt in range(3):
        try:
            r=session.get(url,timeout=28);r.raise_for_status()
            if r.content[:4]!=b"PAR1": raise ValueError("Invalid Parquet")
            data=pd.read_parquet(io.BytesIO(r.content));break
        except Exception:
            if attempt==2:raise
            time.sleep(1.2*(attempt+1))
    df=data.copy()
    df.columns=[str(x).lower() for x in df.columns]
    required=["date","open","high","low","close"]
    if any(x not in df for x in required):raise ValueError("Missing OHLC")
    df["date"]=pd.to_datetime(df["date"]).dt.tz_localize(None)
    df=df[(df["date"]<=ASOF)&(df["date"]>=ASOF-pd.Timedelta(days=405))]
    df=df.sort_values("date").drop_duplicates("date").dropna(subset=required[1:])
    if len(df)<245 or df["date"].iloc[-1]<ASOF-pd.Timedelta(days=4):return {"symbol":symbol,"skipped":"missing recent history"}
    if "split_ratio" in df and (df["split_ratio"].fillna(0)!=0).any():
        return {"symbol":symbol,"skipped":"recent split; requires normalization"}
    a=df[["open","high","low","close"]].to_numpy(float)
    dates=df["date"].dt.strftime("%Y-%m-%d").tolist()
    n=len(a);matches=[]
    for days in [60,75,90,105,120,150,180,210,240]:
        for end in range(max(days,n-85),n-5,4):
            b=base(a[end-days:end])
            if b is None:continue
            U=b["upper"];breakout=-1
            for j in range(end,min(end+15,n-1)):
                if a[j,3]>U*1.025 and a[j+1,3]>U*1.025:
                    breakout=j;break
            if breakout<0:continue
            peak_i=breakout+int(np.argmax(a[breakout:,1]));peak=a[peak_i,1]
            if peak/U<1.07 or peak_i>=n-2 or n-peak_i>60:continue
            close=a[-1,3];relative=close/U-1
            if not -.005<=relative<=.03:continue
            post=a[peak_i+1:]
            # No significant failed test followed by a reclaim.
            if np.min(post[:,2])<U*.99 or np.min(post[:,3])<U*.995:continue
            touches=np.where((post[:,2]<=U*1.025)&(post[:,1]>=U*.99))[0]
            if len(touches)==0:continue
            first_touch=peak_i+1+int(touches[0])
            if first_touch<n-5:continue
            matches.append({
                "ticker":symbol,"asof":dates[-1],"base_start":dates[end-days],
                "base_end":dates[end-1],"base_days":days,
                "upper":round(U,2),"lower":round(b["lower"],2),
                "high_tests":b["highs"],"low_tests":b["lows"],
                "kind":b["kind"],"breakout":dates[breakout],
                "peak":round(float(peak),2),"first_touch":dates[first_touch],
                "close":round(float(close),2),"distance_pct":round(relative*100,2),
                "min_postpeak_low":round(float(np.min(post[:,2])),2),
                "score":b["score"]
            })
    if matches: return {"symbol":symbol,"match":max(matches,key=lambda x:x["score"])}
    return {"symbol":symbol}
def main():
    manifest=pd.read_csv(f"{BASE}/manifest.csv")
    names=manifest["ticker"].dropna().astype(str).tolist()
    results=[];failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
        jobs={pool.submit(analyze,t):t for t in names}
        for f in concurrent.futures.as_completed(jobs):
            try:results.append(f.result())
            except Exception as exc:failures.append({"ticker":jobs[f],"error":str(exc)})
    hits=[x["match"] for x in results if "match" in x]
    skip=[x for x in results if "skipped" in x]
    hits.sort(key=lambda x:-x["score"])
    output={"asof":str(ASOF.date()),"universe":len(names),
       "scanned":len(results)-len(skip),"skipped":skip,"failed":failures,
       "matches":hits,"warning":"Raw technical candidates; options listings and chart review NOT verified"}
    with open("strict_platform_results.json","w") as f:json.dump(output,f,indent=2)
    print("SCAN_RESULT_START")
    print(json.dumps({"universe":len(names),"scanned":len(results)-len(skip),
      "failed":len(failures),"skipped":len(skip),"matches":hits},ensure_ascii=False))
    print("SCAN_RESULT_END")
if __name__=="__main__":main()
