"""
NOVA SIGNALS BOT - Top 100 crypto, 5m/15m/1h/4h, 9 strategies, chart + SL/TP on Telegram
Install : pip install requests pandas numpy matplotlib
Run     : export TG_TOKEN="..." ; export TG_CHAT_ID="..." ; python nova_signals_bot.py
Data    : Binance Futures public API (no key needed)
NOT financial advice. Paper-trade first.
"""
import os, io, time, threading
from concurrent.futures import ThreadPoolExecutor
import requests, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TG_TOKEN = os.getenv("TG_TOKEN", "PASTE_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "PASTE_CHAT_ID")
TFS = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
TOP_N = 100
MIN_CONFLUENCE = 3      # kitni strategies agree karein tab signal aaye
SL_ATR, TP1_R, TP2_R = 1.5, 1.5, 3.0
FAPI = "https://fapi.binance.com"
SPOT = "https://data-api.binance.vision"
FUT = True              # False => spot data (funding/OI strategy off)
ONCE = False
last_sent = {}          # (symbol, tf) -> candle time

def get(path, base=None, **p):
    r = requests.get((base or (FAPI if FUT else SPOT)) + path, params=p, timeout=15)
    r.raise_for_status()
    return r.json()

def pick_source():
    global FUT
    try:
        requests.get(FAPI + "/fapi/v1/ping", timeout=10).raise_for_status()
        FUT = True
    except Exception:
        FUT = False     # 451/blocked => spot mirror
    print("data source:", "Binance Futures" if FUT else "Binance Spot (no funding/OI)")

def top_symbols():
    t = get("/fapi/v1/ticker/24hr" if FUT else "/api/v3/ticker/24hr")
    t = [x for x in t if x["symbol"].endswith("USDT") and "_" not in x["symbol"]]
    t.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [x["symbol"] for x in t[:TOP_N]]

def klines(sym, tf, n=300):
    k = get("/fapi/v1/klines" if FUT else "/api/v3/klines", symbol=sym, interval=tf, limit=n)
    df = pd.DataFrame(k, columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","i"])
    for c in "ohlcv": df[c] = df[c].astype(float)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    return df.iloc[:-1].reset_index(drop=True)   # sirf closed candles

# ---------- indicators ----------
def atr(df, n=14):
    pc = df.c.shift()
    tr = pd.concat([df.h-df.l, (df.h-pc).abs(), (df.l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100/(1 + up/(dn+1e-12))

def macd_hist(s):
    m = s.ewm(span=12).mean() - s.ewm(span=26).mean()
    return m - m.ewm(span=9).mean()

def vwap(df):
    tp = (df.h+df.l+df.c)/3
    day = df.t.dt.date
    return (tp*df.v).groupby(day).cumsum() / df.v.groupby(day).cumsum()

def pivots(df, k=3):
    hi, lo = [], []
    for i in range(k, len(df)-k):
        if df.h[i] == df.h[i-k:i+k+1].max(): hi.append((i, df.h[i]))
        if df.l[i] == df.l[i-k:i+k+1].min(): lo.append((i, df.l[i]))
    return hi, lo

def volume_profile(df, bins=40):
    lo, hi = df.l.min(), df.h.max()
    edges = np.linspace(lo, hi, bins+1)
    vol = np.zeros(bins)
    for _, r in df.iterrows():
        a = max(np.searchsorted(edges, r.l)-1, 0); b = min(np.searchsorted(edges, r.h)-1, bins-1)
        vol[a:b+1] += r.v/(b-a+1)
    poc = (edges[vol.argmax()] + edges[vol.argmax()+1]) / 2
    order = np.argsort(vol)[::-1]; tot = vol.sum(); acc = 0; used = []
    for i in order:
        acc += vol[i]; used.append(i)
        if acc >= 0.7*tot: break
    return poc, edges[min(used)], edges[max(used)+1]   # POC, VAL, VAH

# ---------- strategies: return "LONG"/"SHORT"/None ----------
def s_trend(df, ctx):
    hi, lo = ctx["hi"], ctx["lo"]
    if len(hi) < 2 or len(lo) < 2: return None
    if hi[-1][1] > hi[-2][1] and lo[-1][1] > lo[-2][1]: return "LONG"   # HH + HL
    if hi[-1][1] < hi[-2][1] and lo[-1][1] < lo[-2][1]: return "SHORT"  # LH + LL

def s_sweep(df, ctx):
    ph, pl = df.h[-21:-1].max(), df.l[-21:-1].min(); x = df.iloc[-1]
    if x.h > ph and x.c < ph: return "SHORT"   # high sweep + reversal
    if x.l < pl and x.c > pl: return "LONG"

def s_retest(df, ctx):
    a = ctx["atr"]; x = df.iloc[-1]
    for back in range(2, 7):
        lvl_h, lvl_l = df.h[-back-20:-back].max(), df.l[-back-20:-back].min()
        if df.c[-back] > lvl_h and abs(x.l-lvl_h) < 0.4*a and x.c > lvl_h: return "LONG"
        if df.c[-back] < lvl_l and abs(x.h-lvl_l) < 0.4*a and x.c < lvl_l: return "SHORT"

def s_fvg(df, ctx):
    x = df.iloc[-1]
    for i in range(len(df)-3, len(df)-30, -1):
        if df.l[i] > df.h[i-2] and df.h[i-2] <= x.l <= df.l[i] and x.c > x.o:
            ctx["fvg"] = (i-2, df.h[i-2], df.l[i], "bull"); return "LONG"
        if df.h[i] < df.l[i-2] and df.h[i] <= x.h <= df.l[i-2] and x.c < x.o:
            ctx["fvg"] = (i-2, df.h[i], df.l[i-2], "bear"); return "SHORT"

def s_sr(df, ctx):
    a, x = ctx["atr"], df.iloc[-1]
    sup = [p for _, p in ctx["lo"][-6:]]; res = [p for _, p in ctx["hi"][-6:]]
    if any(abs(x.l-p) < 0.5*a and x.c > p and x.c > x.o for p in sup): return "LONG"
    if any(abs(x.h-p) < 0.5*a and x.c < p and x.c < x.o for p in res): return "SHORT"

def s_rsimacd(df, ctx):
    r, h = ctx["rsi"], ctx["macd"]
    if h.iloc[-2] < 0 <= h.iloc[-1] and 40 < r.iloc[-1] < 65: return "LONG"
    if h.iloc[-2] > 0 >= h.iloc[-1] and 35 < r.iloc[-1] < 60: return "SHORT"

def s_vwap(df, ctx):
    v = ctx["vwap"]
    if df.c.iloc[-2] < v.iloc[-2] and df.c.iloc[-1] > v.iloc[-1]: return "LONG"
    if df.c.iloc[-2] > v.iloc[-2] and df.c.iloc[-1] < v.iloc[-1]: return "SHORT"

def s_vprofile(df, ctx):
    poc, val, vah = volume_profile(df.iloc[-120:-1]); ctx["poc"] = poc
    if df.c.iloc[-2] < val and df.c.iloc[-1] > val: return "LONG"    # wapas value area me
    if df.c.iloc[-2] > vah and df.c.iloc[-1] < vah: return "SHORT"

def s_funding_oi(df, ctx):
    f = ctx.get("funding"); oi = ctx.get("oi_chg")
    if f is None or oi is None: return None
    if f > 0.0005 and oi > 0: return "SHORT"    # crowded longs
    if f < -0.0003 and oi > 0: return "LONG"    # crowded shorts

STRATS = {"Trend HH/HL-LH/LL": s_trend, "Liquidity Sweep": s_sweep, "Breakout+Retest": s_retest,
          "ICT FVG": s_fvg, "Support/Resistance": s_sr, "RSI+MACD": s_rsimacd,
          "VWAP": s_vwap, "Volume Profile": s_vprofile, "Funding+OI": s_funding_oi}

# ---------- analysis ----------
def derivs(sym):
    if not FUT: return None, None
    try:
        f = float(get("/fapi/v1/premiumIndex", symbol=sym)["lastFundingRate"])
        h = get("/futures/data/openInterestHist", symbol=sym, period="1h", limit=6)
        oi = float(h[-1]["sumOpenInterest"]) / float(h[0]["sumOpenInterest"]) - 1
        return f, oi
    except Exception:
        return None, None

def analyze(sym, tf):
    df = klines(sym, tf)
    if len(df) < 150: return None
    hi, lo = pivots(df)
    ctx = {"hi": hi, "lo": lo, "atr": atr(df).iloc[-1], "rsi": rsi(df.c), "macd": macd_hist(df.c), "vwap": vwap(df)}
    if tf in ("1h", "4h"): ctx["funding"], ctx["oi_chg"] = derivs(sym)
    votes = {"LONG": [], "SHORT": []}
    for name, fn in STRATS.items():
        try:
            r = fn(df, ctx)
            if r: votes[r].append(name)
        except Exception:
            pass
    nl, ns = len(votes["LONG"]), len(votes["SHORT"])
    if nl >= MIN_CONFLUENCE and nl > ns + 1: side = "LONG"
    elif ns >= MIN_CONFLUENCE and ns > nl + 1: side = "SHORT"
    else: return None
    if ONCE and (pd.Timestamp.utcnow().tz_localize(None) - df.t.iloc[-1]).total_seconds() > TFS[tf] + 420:
        return None     # candle purani hai, pehle hi check ho chuki
    entry = df.c.iloc[-1]; risk = SL_ATR * ctx["atr"]
    m = 1 if side == "LONG" else -1
    return dict(sym=sym, tf=tf, side=side, entry=entry, sl=entry - m*risk,
                tp1=entry + m*risk*TP1_R, tp2=entry + m*risk*TP2_R,
                why=votes[side], df=df, ctx=ctx, t=df.t.iloc[-1])

# ---------- chart ----------
def draw(sig):
    df = sig["df"].iloc[-80:].reset_index(drop=True); ctx = sig["ctx"]
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0e1117"); ax.set_facecolor("#0e1117")
    for i, r in df.iterrows():
        col = "#26a69a" if r.c >= r.o else "#ef5350"
        ax.vlines(i, r.l, r.h, color=col, lw=1)
        ax.add_patch(Rectangle((i-0.3, min(r.o, r.c)), 0.6, abs(r.c-r.o) or 1e-9, color=col))
    ax.plot(ctx["vwap"].iloc[-80:].values, color="#ffd54f", lw=1, label="VWAP")
    if "fvg" in ctx:
        i0, lo_, hi_, kind = ctx["fvg"]; off = len(sig["df"]) - 80
        ax.add_patch(Rectangle((max(i0-off, 0), lo_), 80, hi_-lo_, color="#7e57c2", alpha=0.25))
    n = len(df)
    for y, c, lab in [(sig["entry"], "#90caf9", "ENTRY"), (sig["sl"], "#ef5350", "SL"),
                      (sig["tp1"], "#66bb6a", "TP1"), (sig["tp2"], "#00e676", "TP2")]:
        ax.axhline(y, color=c, ls="--", lw=1)
        ax.text(n+0.5, y, f"{lab} {y:.6g}", color=c, va="center", fontsize=9)
    ax.set_xlim(0, n+14); ax.tick_params(colors="#aaa"); ax.grid(alpha=0.1)
    ax.set_title(f"{sig['sym']} | {sig['tf']} | {sig['side']}", color="white")
    ax.legend(facecolor="#0e1117", labelcolor="white", loc="upper left")
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
    buf.seek(0); return buf

# ---------- telegram ----------
def send(sig):
    e = "🟢" if sig["side"] == "LONG" else "🔴"
    cap = (f"{e} {sig['side']} | {sig['sym']} | {sig['tf']}\n"
           f"Entry: {sig['entry']:.6g}\nSL: {sig['sl']:.6g}\nTP1: {sig['tp1']:.6g}\nTP2: {sig['tp2']:.6g}\n"
           f"Confluence ({len(sig['why'])}): " + ", ".join(sig["why"]) + "\n⚠️ Not financial advice")
    requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
                  data={"chat_id": TG_CHAT_ID, "caption": cap},
                  files={"photo": ("chart.png", draw(sig))}, timeout=30)

# ---------- main loop ----------
def scan(tf, syms):
    def one(sym):
        try:
            s = analyze(sym, tf)
            if s and last_sent.get((sym, tf)) != s["t"]:
                last_sent[(sym, tf)] = s["t"]; send(s); print("sent", sym, tf, s["side"])
        except Exception as ex:
            print("err", sym, tf, ex)
    with ThreadPoolExecutor(6) as ex: list(ex.map(one, syms))

def worker(tf, secs):
    while True:
        try:
            syms = top_symbols(); scan(tf, syms)
        except Exception as ex:
            print("loop err", tf, ex)
        time.sleep(secs - (time.time() % secs) + 5)   # next candle close + 5s

def due(tf, now):
    m, h = now.minute, now.hour
    return {"5m": True, "15m": m % 15 < 5, "1h": m < 5, "4h": h % 4 == 0 and m < 5}[tf]

if __name__ == "__main__":
    import sys
    pick_source()
    if "--once" in sys.argv:        # GitHub Actions mode: ek baar scan karke exit
        ONCE = True
        now = pd.Timestamp.utcnow()
        syms = top_symbols()
        for tf in TFS:
            if due(tf, now): scan(tf, syms)
        sys.exit(0)
    print("NOVA SIGNALS BOT started")
    for tf, secs in TFS.items(): threading.Thread(target=worker, args=(tf, secs), daemon=True).start()
    while True: time.sleep(3600)
