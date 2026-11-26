#!/usr/bin/env python3
"""S&P 400 midcap options-stock prefilter, free Wikipedia universe and Yahoo EOD history."""
import io,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from strict_platform_scan import base

ASOF="2026-11-25"
def universe():
    url="https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    page=requests.get(url,timeout=25,headers={"User-Agent":"Mozilla/5.0 (market research)"})
    page.raise_for_status()
    tables=pd.read_html(io.StringIO(page.text))
    for tab in tables:
        for col in ["Symbol","Ticker symbol","Ticker","Ticker Symbol"]:
            if col in tab.columns and len(tab)>=300:
                return sorted({str(v).replace(".","-").strip() for v in tab[col].dropna()})
    raise RuntimeError("No S&P400 constituents table in Wikipedia")
def screen(symbol, raw):
    if raw is None or raw.empty:return []
    df=raw.copy()
    if isinstance(df.columns,pd.MultiIndex):
        return []
    df.columns=[str(x).lower() for x in df.columns]
    need=["open","high","low","close"]
    if any(x not in df.columns for x in need):return []
    df=df.dropna(subset=need).sort_index()
    dates=pd.to_datetime(df.index).strftime("%Y-%m-%d").tolist()
    if len(df)<100 or dates[-1]<"2026-11-20":return []
    a=df[need].to_numpy(float);N=len(a)
    hits=[]
    for days in [60,75,90,105,120,150,180,210,240]:
        for end in range(max(days,N-85),N-5,4):
            p=base(a[end-days:end])
            if p is None:continue
            U=p["upper"]
            bk=-1
            for j in range(end,min(end+15,N-1)):
                if a[j,3]>U*1.015 and a[j+1,3]>U*1.015:
                    bk=j;break
            if bk<0:continue
            peak_i=bk+int(np.argmax(a[bk:,1]))
            if a[peak_i,1]/U<1.04 or peak_i>=N-2 or N-peak_i>60:continue
            distance=a[-1,3]/U-1
            if not -.01<=distance<=.05:continue
            post=a[peak_i+1:];touch=np.where((post[:,2]<=U*1.025)&(post[:,1]>=U*.99))[0]
            if len(touch)==0 or peak_i+1+int(touch[0])<N-10:continue
            if np.min(post[:,2])<U*.99 or np.min(post[:,3])<U*.99:continue
            hits.append(dict(ticker=symbol,base_start=dates[end-days],base_end=dates[end-1],
                upper=round(U,2),lower=round(p["lower"],2),base_days=days,
                high_tests=p["highs"],low_tests=p["lows"],breakout=dates[bk],
                peak=round(float(a[peak_i,1]),2),first_touch=dates[peak_i+1+int(touch[0])],
                close=round(float(a[-1,3]),2),distance_pct=round(distance*100,2),
                min_postpeak_low=round(float(np.min(post[:,2])),2),score=p["score"]))
    return sorted(hits,key=lambda x:-x["score"])[:1]
def main():
    names=universe()
    exclusions=set(pd.read_csv(
       "https://raw.githubusercontent.com/floyds1995/Auto-Index-Constituents-Tracker/main/us_data/raw_ohlcv/manifest.csv"
    )["ticker"].astype(str))
    names=[n for n in names if n not in exclusions]
    hits=[];failures=[];scanned=0;skipped=0
    for idx in range(0,len(names),12):
        batch=names[idx:idx+12]
        try:
            dat=yf.download(batch,start="2025-09-01",end="2026-11-26",
                       auto_adjust=False,group_by="ticker",threads=False,
                       progress=False,timeout=22)
            if dat is None or dat.empty:raise ValueError("empty batch")
            for symbol in batch:
                try:
                    raw=dat[symbol] if isinstance(dat.columns,pd.MultiIndex) else dat
                    if raw is None or raw.empty:skipped+=1;continue
                    r=screen(symbol,raw)
                    scanned+=1
                    hits.extend(r)
                except Exception as e:failures.append({"ticker":symbol,"error":str(e)})
        except Exception as e:
            failures.extend({"ticker":n,"error":str(e)} for n in batch)
        print(f"Midcap progress {min(idx+12,len(names))}/{len(names)}",flush=True)
        time.sleep(.45)
    out=dict(asof=ASOF,universe=len(names),scanned=scanned,skipped=skipped,
             failures=failures,matches=hits,
             note="Technical candidates require chart review and listing options verification.")
    Path("midcap_strict_results.json").write_text(json.dumps(out,indent=2))
    print("MIDCAP_SCAN_RESULT_START")
    print(json.dumps({k:v for k,v in out.items() if k!="failures"},ensure_ascii=False))
    print("MIDCAP_SCAN_RESULT_END")
if __name__=="__main__":main()
