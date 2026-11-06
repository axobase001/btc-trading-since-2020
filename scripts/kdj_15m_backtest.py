#!/usr/bin/env python3
"""Reproducible Binance USDT-M perpetual 15m KDJ K/D crossover backtest.
Run: python kdj_15m_backtest.py
Data: data.binance.vision (public Binance official OHLCV archives).
UTC evaluation: 2025-11-06 00:00 through 2026-11-06 00:00 exclusive.
"""
from __future__ import annotations
import csv, io, json, math, os, statistics, time, urllib.request, zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC=timezone.utc
START=datetime(2025,11,6,tzinfo=UTC)
END=datetime(2026,11,6,tzinfo=UTC)
INTERVAL=timedelta(minutes=15)
OUT=Path('kdj_15m_results')
SYMBOLS=('BTCUSDT','ETHUSDT')
BASE_URL='https://data.binance.vision/data/futures/um'

@dataclass(frozen=True)
class Bar:
    t: datetime
    o: float
    h: float
    l: float
    c: float

def fetch(url: str) -> bytes:
    last=None
    for attempt in range(4):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 KDJ-Historical-Backtest/1.0'})
            with urllib.request.urlopen(req,timeout=45) as f:
                return f.read()
        except Exception as e:
            last=e
            print('DOWNLOAD_RETRY',attempt+1,url,str(e)[:100],flush=True)
            time.sleep(2**attempt)
    raise RuntimeError(f'Cannot download actual historical data: {url}: {last}')

def archive_bars(url:str)->list[Bar]:
    data=fetch(url)
    rows=[]
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        csvs=[x for x in z.namelist() if x.lower().endswith('.csv')]
        if len(csvs)!=1:
            raise ValueError(f'Expected one CSV in {url}, found {csvs}')
        with z.open(csvs[0]) as handle:
            text=io.TextIOWrapper(handle, encoding='utf-8-sig',newline='')
            for r in csv.reader(text):
                if not r or not r[0].strip().lstrip('-').isdigit():
                    continue
                t=int(r[0]); t=t//1000 if t>10**14 else t
                stamp=datetime.fromtimestamp(t/1000,tz=UTC)
                o,h,l,c=(float(v) for v in r[1:5])
                if not (l>0 and l<=min(o,c) and max(o,c)<=h and all(math.isfinite(x) for x in (o,h,l,c))):
                    raise ValueError(f'Invalid candle at {stamp} in {url}: {r[:5]}')
                rows.append(Bar(stamp,o,h,l,c))
    if not rows: raise ValueError(f'No OHLCV rows in {url}')
    print('DOWNLOADED',url.rsplit('/',1)[-1],len(rows),rows[0].t.isoformat(),rows[-1].t.isoformat(),flush=True)
    return rows

def load_symbol(symbol:str)->list[Bar]:
    d={}
    for year,month in [(2025,11),(2025,12)]+[(2026,m) for m in range(1,11)]:
        fn=f'{symbol}-15m-{year:04d}-{month:02d}.zip'
        url=f'{BASE_URL}/monthly/klines/{symbol}/15m/{fn}'
        for b in archive_bars(url):
            if b.t in d and b!=d[b.t]: raise ValueError(f'Conflicting duplicate candle: {symbol} {b.t}')
            d[b.t]=b
    for day in range(1,6):
        fn=f'{symbol}-15m-2026-11-{day:02d}.zip'
        url=f'{BASE_URL}/daily/klines/{symbol}/15m/{fn}'
        for b in archive_bars(url):
            if b.t in d and b!=d[b.t]: raise ValueError(f'Conflicting duplicate candle: {symbol} {b.t}')
            d[b.t]=b
    bars=sorted((b for b in d.values() if datetime(2025,11,1,tzinfo=UTC)<=b.t<END),key=lambda x:x.t)
    gaps=[(a.t.isoformat(),b.t.isoformat()) for a,b in zip(bars,bars[1:]) if b.t-a.t!=INTERVAL]
    if gaps: raise ValueError(f'Incomplete 15m OHLCV for {symbol}: {len(gaps)} time gaps, e.g. {gaps[:5]}')
    expected=int((END-datetime(2025,11,1,tzinfo=UTC))/INTERVAL)
    if len(bars)!=expected: raise ValueError(f'Unexpected row count for {symbol}: {len(bars)} != {expected}')
    return bars

def kdj_cross(bars:list[Bar])->list[int]:
    """Chinese standard KDJ(9,3,3), seed K=D=50. Signal at close of bar i."""
    signs=[0]*len(bars)
    k=d=50.0
    prev_k=prev_d=None
    for i in range(8,len(bars)):
        window=bars[i-8:i+1]
        h=max(b.h for b in window); l=min(b.l for b in window)
        rsv=50.0 if h==l else (bars[i].c-l)/(h-l)*100.0
        k=(2*k+rsv)/3
        d=(2*d+k)/3
        if prev_k is not None:
            if prev_k<=prev_d and k>d: signs[i]=1
            elif prev_k>=prev_d and k<d: signs[i]=-1
        prev_k,prev_d=k,d
    return signs

def triggered(b:Bar,entry:float,side:int,tp:float=.01,sl:float=.005):
    """Returns (reason, mid exit price). Stop-first for same-bar high/low ambiguity."""
    target=entry*(1+side*tp)
    stop=entry*(1-side*sl)
    if side==1:
        if b.o<=stop: return ('stop_gap',b.o)
        if b.o>=target: return ('target',target)
        if b.l<=stop: return ('stop',stop)
        if b.h>=target: return ('target',target)
    else:
        if b.o>=stop: return ('stop_gap',b.o)
        if b.o<=target: return ('target',target)
        if b.h>=stop: return ('stop',stop)
        if b.l<=target: return ('target',target)
    return None

def simulate(bars:list[Bar],signal:list[int],fee_bps:float,slip_bps:float,save_trades:bool=False):
    fee=fee_bps/10_000; slip=slip_bps/10_000
    equity=10_000.0; peak=equity; worst_dd=0.0
    pos=None; trades=[]; marks={}; daily={}; monthly={}; used_signal=0; ignored_signal=0
    for i,b in enumerate(bars):
        if b.t<START: continue
        if b.t>=END: break
        prev=signal[i-1] if i>0 else 0
        if pos is None and prev:
            entry_exec=b.o*(1+prev*slip)
            pos={'side':prev,'entry_time':b.t,'entry_idx':i,'mid':b.o,
                 'entry':entry_exec,'start_equity':equity}
            used_signal+=1
        elif pos is not None and prev:
            ignored_signal+=1
        if pos is not None:
            s=pos['side']; entry=pos['entry']
            hit=triggered(b,entry,s)
            if hit:
                reason,mid=hit; exit_exec=mid*(1-s*slip)
                gross=s*(exit_exec-entry)/entry
                net=gross-fee*(1+exit_exec/entry)
                equity=pos['start_equity']*(1+net)
                trades.append({'side':'LONG' if s==1 else 'SHORT','entry_utc':pos['entry_time'].isoformat(),
                    'exit_utc':b.t.isoformat(),'entry_price':entry,'exit_price':exit_exec,
                    'bars_held':i-pos['entry_idx']+1,'reason':reason,
                    'gross_pct':round(gross*100,6),'net_pct':round(net*100,6),
                    'pnl_usdt':round(equity-pos['start_equity'],6),'equity':round(equity,6)})
                pos=None
                mark=equity
            else:
                theoretical_exit=b.c*(1-s*slip)
                floating=s*(theoretical_exit-entry)/entry-fee*(1+theoretical_exit/entry)
                mark=pos['start_equity']*(1+floating)
        else:
            mark=equity
        peak=max(peak,mark)
        worst_dd=min(worst_dd,mark/peak-1)
        daily[b.t.strftime('%Y-%m-%d')]=round(mark,6)
        monthly[b.t.strftime('%Y-%m')]=round(mark,6)
    lastmark=next(reversed(daily.values()))
    wins=[x for x in trades if x['net_pct']>0]
    losses=[x for x in trades if x['net_pct']<=0]
    sumwins=sum(x['net_pct'] for x in wins)
    sumlosses=sum(x['net_pct'] for x in losses)
    days=len(daily)
    day_end=list(daily.values())
    daily_rets=[b/a-1 for a,b in zip([10_000.0]+day_end[:-1],day_end)]
    if len(daily_rets)>1 and statistics.stdev(daily_rets)>0:
        sharpe=math.sqrt(365)*statistics.mean(daily_rets)/statistics.stdev(daily_rets)
    else: sharpe=None
    longs=[x for x in trades if x['side']=='LONG']
    shorts=[x for x in trades if x['side']=='SHORT']
    month_rets={}; prev_val=10_000.0
    for month,val in monthly.items():
        month_rets[month]=round((val/prev_val-1)*100,3);prev_val=val
    n=len(trades)
    def r(x):return round(x,4) if x is not None and math.isfinite(x) else None
    summary={
        'fee_bps_per_side':fee_bps,'slippage_bps_per_side':slip_bps,'candles':len(bars),
        'start_utc':START.isoformat(),'end_utc_exclusive':END.isoformat(),
        'closed_trades':n,'long_trades':len(longs),'short_trades':len(shorts),
        'long_win_pct':r(100*sum(x['net_pct']>0 for x in longs)/len(longs)) if longs else None,
        'short_win_pct':r(100*sum(x['net_pct']>0 for x in shorts)/len(shorts)) if shorts else None,
        'overall_win_pct':r(100*len(wins)/n) if n else None,
        'profit_factor':r(sumwins/-sumlosses) if sumlosses<0 else None,
        'avg_trade_net_pct':r(sum(x['net_pct'] for x in trades)/n) if n else None,
        'avg_win_pct':r(sumwins/len(wins)) if wins else None,
        'avg_loss_pct':r(sumlosses/len(losses)) if losses else None,
        'avg_hold_hours':r(sum(x['bars_held'] for x in trades)*.25/n) if n else None,
        'targets':sum(x['reason']=='target' for x in trades),
        'stops':sum(x['reason'].startswith('stop') for x in trades),
        'unclosed_position':pos is not None,
        'open_side':None if pos is None else ('LONG' if pos['side']==1 else 'SHORT'),
        'exit_signals_ignored_while_in_trade':ignored_signal,
        'signal_entries':used_signal,
        'equity_from_10000':r(lastmark),'closed_equity':r(equity),
        'return_pct':r((lastmark/10_000.0-1)*100),
        'max_drawdown_pct':r(worst_dd*100),
        'daily_sharpe_365':r(sharpe),
        'monthly_return_pct':month_rets,
    }
    return summary,trades,daily

def main():
    OUT.mkdir(exist_ok=True)
    all_results={
        'strategy':'KDJ(9,3,3): K/D crossing. Signals at candle close, enter at next 15m candle open. TP +1%, SL -0.5%, no reversal exit.',
        'market':'Binance USD-M perpetual futures, BTCUSDT and ETHUSDT',
        'equity':'Each symbol starts with 10,000 USDT, 1x notional of current equity, full compounding, only one position per symbol, no leverage/margin liquidation, funding excluded.',
        'ambiguity':'If TP and SL both touch within the same 15m bar, STOP is filled first; adverse gaps fill at open.',
        'data':'Binance Vision monthly archives 2025-11 to 2026-10 + daily 2026-11-01 to 2026-11-05, UTC.',
        'scenarios':{'ideal':{'fee_bps':0,'slippage_bps':0},
                     'base':{'fee_bps':5,'slippage_bps':2},
                     'stress':{'fee_bps':10,'slippage_bps':3}},'results':{}
    }
    for symbol in SYMBOLS:
        bars=load_symbol(symbol)
        signals=kdj_cross(bars)
        all_results['results'][symbol]={}
        for scenario,costs in all_results['scenarios'].items():
            summary,trades,daily=simulate(bars,signals,costs['fee_bps'],costs['slippage_bps'])
            all_results['results'][symbol][scenario]=summary
            if scenario=='base':
                with open(OUT/f'{symbol}_trades.csv','w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['side','entry_utc','exit_utc','entry_price','exit_price','bars_held','reason','gross_pct','net_pct','pnl_usdt','equity'])
                    w.writeheader();w.writerows(trades)
                with open(OUT/f'{symbol}_equity_daily.csv','w',newline='') as f:
                    w=csv.writer(f);w.writerow(['date_utc','equity_usdt'])
                    for k,v in daily.items():w.writerow([k,v])
            short={key:val for key,val in summary.items() if key not in ('monthly_return_pct',)}
            print('SCENARIO_SUMMARY',symbol,scenario,json.dumps(short,ensure_ascii=False,sort_keys=True),flush=True)
    with open(OUT/'summary.json','w') as f: json.dump(all_results,f,indent=2,ensure_ascii=False)
    print('KDJ_BACKTEST_SUMMARY_JSON='+json.dumps(all_results,ensure_ascii=False,separators=(',',':')),flush=True)

if __name__=='__main__': main()
