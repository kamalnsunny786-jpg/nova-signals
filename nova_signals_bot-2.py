"""
NOVA SIGNALS BOT - Top 100 crypto, 5m/15m/1h/4h, 9 strategies, chart + SL/TP on Telegram
Install : pip install requests pandas numpy matplotlib
Run     : export TG_TOKEN="..." ; export TG_CHAT_ID="..." ; python nova_signals_bot.py
Data    : Binance Futures public API (no key needed)
NOT financial advice. Paper-trade first.
"""
import os, io, time, threading, json, datetime
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
# ---- accuracy filters (research based) ----
MIN_GROUPS = {"5m": 4, "15m": 3, "1h": 3, "4h": 3}   # kitne ALAG type ke groups agree karein
HTF = {"5m": "1h", "15m": "1h", "1h": "4h", "4h": "1d"}   # bade timeframe ka trend signal ke saath hona chahiye
ADX_MIN = 20            # bade timeframe par trend itna strong ho
MIN_RVOL = 1.0          # signal candle ka volume 20-candle average se kam na ho
MIN_RISK_PCT = 0.30     # SL distance (%) isse chhota ho to fees profit kha jaati hain
COOLDOWN_CANDLES = 6    # same coin+tf+side ka signal itni candles tak dobara nahi
USE_BTC_FILTER = True   # alt coin ka LONG tab nahi jab BTC bearish ho (aur ulta)
SL_ATR, TP1_R, TP2_R = 1.5, 1.5, 3.0
GROUPS = {"Structure": ["Trend HH/HL-LH/LL", "Breakout+Retest"], "Smart Money": ["Liquidity Sweep", "ICT FVG"], "Levels": ["Support/Resistance", "Volume Profile"], "Momentum": ["RSI+MACD"], "VWAP": ["VWAP"], "Derivatives": ["Funding+OI"]}
FAPI = "https://fapi.binance.com"
SPOT = "https://data-api.binance.vision"
FUT = True              # False => spot data (funding/OI strategy off)
ONCE = False
last_sent = {}
DATA_DIR = "data"; SIG_FILE = f"{DATA_DIR}/signals.json"; STATE_FILE = f"{DATA_DIR}/state.json"
WEEKLY_HOUR_UTC = 15    # Sunday 15:00 UTC = 8:30 PM IST (is time ke baad pehle run par report)
RMULT = {"SL": -1.0, "TP1": TP1_R, "TP2": TP2_R, "EXP": 0.0}   # result in R (1R = risk of one trade)
EXPIRE = {"5m": 12*3600, "15m": 24*3600, "1h": 72*3600, "4h": 7*86400}
SIGS, STATE = [], {}
lock = threading.RLock()          # (symbol, tf) -> candle time

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

def adx(df, n=14):
    up = df.h.diff(); dn = -df.l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    pc = df.c.shift()
    tr = pd.concat([df.h-df.l, (df.h-pc).abs(), (df.l-pc).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100*plus.ewm(alpha=1/n, adjust=False).mean()/(a+1e-12)
    mdi = 100*minus.ewm(alpha=1/n, adjust=False).mean()/(a+1e-12)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi+1e-12)
    return dx.ewm(alpha=1/n, adjust=False).mean()

_HTF = {}
def htf_bias(sym, tf):
    """bade timeframe ka trend: bull / bear / neutral (EMA20>EMA50 + ADX strong)"""
    key = (sym, tf); c = _HTF.get(key)
    if c and time.time() - c[0] < 300: return c[1]
    res = "neutral"
    try:
        df = klines(sym, tf, 250)
        if len(df) > 100:
            e20 = df.c.ewm(span=20).mean().iloc[-1]; e50 = df.c.ewm(span=50).mean().iloc[-1]
            last = df.c.iloc[-1]; a = adx(df).iloc[-1]
            if last > e50 and e20 > e50 and a >= ADX_MIN: res = "bull"
            elif last < e50 and e20 < e50 and a >= ADX_MIN: res = "bear"
    except Exception:
        pass
    _HTF[key] = (time.time(), res)
    return res

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
    gl = [g for g, names in GROUPS.items() if any(n in votes["LONG"] for n in names)]
    gs = [g for g, names in GROUPS.items() if any(n in votes["SHORT"] for n in names)]
    need = MIN_GROUPS[tf]
    if len(gl) >= need and len(gl) >= len(gs) + 2: side, groups = "LONG", gl
    elif len(gs) >= need and len(gs) >= len(gl) + 2: side, groups = "SHORT", gs
    else: return None
    if ONCE and (pd.Timestamp.now("UTC").tz_localize(None) - df.t.iloc[-1]).total_seconds() > TFS[tf] + 420:
        return None     # candle purani hai, pehle hi check ho chuki
    # --- filter 1: volume ---
    rvol = df.v.iloc[-1] / (df.v.iloc[-21:-1].mean() + 1e-12)
    if rvol < MIN_RVOL: return None
    # --- stop loss: swing ke bahar, 1 se 2.5 ATR ke beech ---
    a = ctx["atr"]; entry = df.c.iloc[-1]; m = 1 if side == "LONG" else -1
    base = (df.l.iloc[-10:].min() - 0.2*a) if side == "LONG" else (df.h.iloc[-10:].max() + 0.2*a)
    risk = float(np.clip(abs(entry - base), 1.0*a, 2.5*a))
    risk_pct = 100*risk/entry
    # --- filter 2: SL bahut chhota ho to fees khaa jaati hain ---
    if risk_pct < MIN_RISK_PCT: return None
    # --- filter 3: cooldown (same signal baar baar nahi) ---
    with lock:
        recent = [x for x in SIGS if x["sym"] == sym and x["tf"] == tf and x["side"] == side]
    cur_ts = int(df.t.iloc[-1].tz_localize("UTC").timestamp()*1000) + TFS[tf]*1000     # is candle ka close time
    if any(0 <= cur_ts - x["ts"] < COOLDOWN_CANDLES*TFS[tf]*1000 for x in recent): return None
    # --- filter 4: bade timeframe ka trend ---
    want = "bull" if side == "LONG" else "bear"
    if htf_bias(sym, HTF[tf]) != want: return None
    # --- filter 5: BTC ka trend (alt coins ke liye) ---
    if USE_BTC_FILTER and sym != "BTCUSDT":
        b = htf_bias("BTCUSDT", HTF[tf])
        if b != "neutral" and b != want: return None
    why = [n for g in groups for n in GROUPS[g] if n in votes[side]]
    return dict(sym=sym, tf=tf, side=side, entry=entry, sl=entry - m*risk,
                tp1=entry + m*risk*TP1_R, tp2=entry + m*risk*TP2_R,
                why=why, groups=groups, rvol=rvol, risk_pct=risk_pct, htf=HTF[tf],
                df=df, ctx=ctx, t=df.t.iloc[-1])

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

# ---------- data (signals log) ----------
def load_data():
    global SIGS, STATE
    os.makedirs(DATA_DIR, exist_ok=True)
    try: SIGS = json.load(open(SIG_FILE))
    except Exception: SIGS = []
    try: STATE = json.load(open(STATE_FILE))
    except Exception: STATE = {}

def save_data():
    with lock:
        os.makedirs(DATA_DIR, exist_ok=True)
        cutoff = time.time()*1000 - 30*86400*1000          # 30 din se purane hata do
        SIGS[:] = [x for x in SIGS if x["ts"] >= cutoff]
        json.dump(SIGS, open(SIG_FILE, "w")); json.dump(STATE, open(STATE_FILE, "w"))

def log_signal(sig):
    key = f"{sig['sym']}|{sig['tf']}|{sig['t']}"
    with lock:
        if any(x["id"] == key for x in SIGS): return False      # pehle hi bhej chuke
        ts = int(sig["t"].tz_localize("UTC").timestamp()*1000) + TFS[sig["tf"]]*1000
        SIGS.append(dict(id=key, sym=sig["sym"], tf=sig["tf"], side=sig["side"], entry=float(sig["entry"]),
                         sl=float(sig["sl"]), tp1=float(sig["tp1"]), tp2=float(sig["tp2"]), ts=ts, status="open"))
    return True

# ---------- result tracking ----------
def judge(x, rows, now_ms):
    lg = x["side"] == "LONG"; st = x["status"]; exp_ms = x["ts"] + EXPIRE[x["tf"]]*1000
    for r in rows:
        if int(r[0]) < x["ts"]: continue
        if int(r[0]) > exp_ms: break
        h, l = float(r[2]), float(r[3])
        if st == "open":
            if (l <= x["sl"]) if lg else (h >= x["sl"]): st = "SL"; break      # same candle me dono => SL (conservative)
            if (h >= x["tp1"]) if lg else (l <= x["tp1"]):
                st = "tp1_hit"
                if (h >= x["tp2"]) if lg else (l <= x["tp2"]): st = "TP2"; break
        elif st == "tp1_hit":                                                  # TP1 ke baad SL entry par
            if (l <= x["entry"]) if lg else (h >= x["entry"]): st = "TP1"; break
            if (h >= x["tp2"]) if lg else (l <= x["tp2"]): st = "TP2"; break
    if st in ("open", "tp1_hit") and now_ms > exp_ms:
        st = "EXP" if st == "open" else "TP1"
    return st

def update_outcomes():
    now_ms = int(time.time()*1000)
    pend = [x for x in SIGS if x["status"] in ("open", "tp1_hit")]
    def one(x):
        try:
            iv = "5m" if x["tf"] in ("5m", "15m") else "15m"
            rows = get("/fapi/v1/klines" if FUT else "/api/v3/klines", symbol=x["sym"], interval=iv,
                       startTime=x["ts"], limit=1000)
            x["status"] = judge(x, rows, now_ms)
        except Exception as ex:
            print("upd err", x["sym"], ex)
    with ThreadPoolExecutor(6) as ex: list(ex.map(one, pend))

def agg(lst):
    w = sum(1 for x in lst if x["status"] in ("TP1", "TP2")); l = sum(1 for x in lst if x["status"] == "SL")
    r = sum(RMULT[x["status"]] for x in lst if x["status"] in RMULT)
    return dict(n=len(lst), w=w, l=l, o=len(lst)-w-l, r=r, wr=(100*w/(w+l) if w+l else 0))

def weekly_card(a, by_tf, start, end):
    fig = plt.figure(figsize=(8, 10), facecolor="#0e1117")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(.5, .94, "NOVASIGNALS", ha="center", color="white", fontsize=32, weight="bold")
    ax.text(.5, .895, "WEEKLY PERFORMANCE", ha="center", color="#82d2ff", fontsize=16)
    ax.text(.5, .86, f"{start:%d %b} - {end:%d %b %Y} (UTC)", ha="center", color="#999", fontsize=12)
    closed = a["w"] + a["l"]
    if closed == 0:
        ax.text(.5, .74, "WAITING", ha="center", color="#ffd54f", fontsize=54, weight="bold")
        ax.text(.5, .675, "Signals are still open - results coming soon", ha="center", color="#aaa", fontsize=12)
    else:
        col = "#26a69a" if a["wr"] >= 50 else "#ef5350"
        ax.text(.5, .74, f"{a['wr']:.0f}%", ha="center", color=col, fontsize=70, weight="bold")
        note = f"Win rate ({closed} closed signals)" + ("  -  small sample" if closed < 10 else "")
        ax.text(.5, .675, note, ha="center", color="#aaa", fontsize=12)
    for x, v, lab, c in [(.14, a["n"], "Signals", "white"), (.38, a["w"], "Wins", "#26a69a"),
                         (.62, a["l"], "Losses", "#ef5350"), (.86, a["o"], "Open/Exp", "#ffd54f")]:
        ax.text(x, .56, str(v), ha="center", color=c, fontsize=30, weight="bold")
        ax.text(x, .52, lab, ha="center", color="#aaa", fontsize=11)
    if closed == 0:
        ax.text(.5, .44, "Net result: pending", ha="center", fontsize=22, weight="bold", color="#ffd54f")
    else:
        ax.text(.5, .44, f"Net result: {a['r']:+.1f}R", ha="center", fontsize=22, weight="bold",
                color="#26a69a" if a["r"] >= 0 else "#ef5350")
    for i, tf in enumerate(TFS):
        y = .35 - i*.065; b = by_tf[tf]
        ax.text(.08, y, tf, color="white", fontsize=14, va="center", weight="bold")
        ax.add_patch(plt.Rectangle((.18, y-.012), .5, .024, color="#222a36"))
        if b["w"] + b["l"] == 0:
            ax.text(.71, y, "no signals" if b["n"] == 0 else f"waiting ({b['n']} open)", color="#888", fontsize=11, va="center")
        else:
            ax.add_patch(plt.Rectangle((.18, y-.012), .5*b["wr"]/100, .024, color="#26a69a"))
            ax.text(.71, y, f"{b['wr']:.0f}%  ({b['w']}W/{b['l']}L)", color="#ccc", fontsize=12, va="center")
    ax.text(.5, .075, "1R = risk per trade  |  SL = -1R, TP1 = +%.1fR, TP2 = +%.1fR" % (TP1_R, TP2_R),
            ha="center", color="#777", fontsize=10)
    ax.text(.5, .045, "Not financial advice. Past results do not guarantee future profit.", ha="center", color="#777", fontsize=10)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor()); plt.close(fig)
    return buf.getvalue()

def tg(method, files=None, **data):
    r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/{method}", data=data, files=files, timeout=30)
    try: return r.json()
    except Exception: return {}

def weekly_report():
    update_outcomes()
    now = datetime.datetime.utcnow(); start = now - datetime.timedelta(days=7)
    start_ms = int(start.replace(tzinfo=datetime.timezone.utc).timestamp()*1000)
    week = [x for x in SIGS if x["ts"] >= start_ms]
    STATE["last_weekly"] = now.strftime("%Y-%m-%d")
    if not week:
        print("weekly: is hafte koi signal nahi"); return
    a = agg(week); by_tf = {tf: agg([x for x in week if x["tf"] == tf]) for tf in TFS}
    card = weekly_card(a, by_tf, start, now)
    closed = [x for x in week if x["status"] in ("TP1", "TP2", "SL")]
    best = sorted([x for x in closed if x["status"] != "SL"], key=lambda x: -RMULT[x["status"]])[:5]
    worst = [x for x in closed if x["status"] == "SL"][:5]
    closed = a["w"] + a["l"]
    result_line = ("Results: waiting - signals still open" if closed == 0
                   else f"Win rate {a['wr']:.0f}% ({closed} closed) | Net {a['r']:+.1f}R")
    cap = (f"📊 Weekly report {start:%d %b} - {now:%d %b}\n✅ Wins {a['w']}  ❌ Losses {a['l']}  ⏳ Open/Exp {a['o']}\n"
           f"{result_line}\n⚠️ Not financial advice")
    lines = ["📋 Details", ""]
    for tf in TFS:
        b = by_tf[tf]
        lines.append(f"{tf}: {b['n']} signals | {b['w']}W/{b['l']}L | {b['r']:+.1f}R" if b["w"] + b["l"] else (f"{tf}: no signals" if b["n"] == 0 else f"{tf}: {b['n']} signals | waiting for results"))
    if best:
        lines += ["", "🏆 Best:"] + [f"✅ {x['sym']} {x['tf']} {x['side']} -> {x['status']}" for x in best]
    if worst:
        lines += ["", "⚠️ Stopped out:"] + [f"❌ {x['sym']} {x['tf']} {x['side']}" for x in worst]
    text = "\n".join(lines)
    for chat in [c.strip() for c in TG_CHAT_ID.split(",") if c.strip()]:
        res = tg("sendPhoto", files={"photo": ("weekly.png", card)}, chat_id=chat, caption=cap)
        mid = (res.get("result") or {}).get("message_id")
        if mid:
            old = STATE.setdefault("pins", {}).get(chat)
            if old: tg("unpinChatMessage", chat_id=chat, message_id=old)     # purani weekly pin hatao
            tg("pinChatMessage", chat_id=chat, message_id=mid, disable_notification="true")
            STATE["pins"][chat] = mid
        tg("sendMessage", chat_id=chat, text=text)

def weekly_due(now):
    return now.weekday() == 6 and now.hour >= WEEKLY_HOUR_UTC and STATE.get("last_weekly") != now.strftime("%Y-%m-%d")

# ---------- telegram ----------
def send(sig):
    e = "🟢" if sig["side"] == "LONG" else "🔴"
    g = len(sig["groups"]); grade = "A+" if g >= 5 else ("A" if g == 4 else "B")
    trend = "Bullish" if sig["side"] == "LONG" else "Bearish"
    cap = (f"{e} {sig['side']} | {sig['sym']} | {sig['tf']}  [Grade {grade}]\n"
           f"Entry: {sig['entry']:.6g}\nSL: {sig['sl']:.6g}  (risk {sig['risk_pct']:.2f}%)\n"
           f"TP1: {sig['tp1']:.6g}  (1:{TP1_R:g})\nTP2: {sig['tp2']:.6g}  (1:{TP2_R:g})\n"
           f"✅ {sig['htf']} trend: {trend} | Volume x{sig['rvol']:.1f}\n"
           f"Confirmations ({g} groups): " + ", ".join(sig["why"]) + "\n⚠️ Not financial advice")
    img = draw(sig).getvalue()
    for chat in [c.strip() for c in TG_CHAT_ID.split(",") if c.strip()]:   # ek se zyada chat/channel: comma se alag
        try:
            requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
                          data={"chat_id": chat, "caption": cap},
                          files={"photo": ("chart.png", img)}, timeout=30)
        except Exception as ex:
            print("send err", chat, ex)

# ---------- main loop ----------
def scan(tf, syms):
    def one(sym):
        try:
            s = analyze(sym, tf)
            if s and log_signal(s):
                send(s); print("sent", sym, tf, s["side"])
        except Exception as ex:
            print("err", sym, tf, ex)
    with ThreadPoolExecutor(6) as ex: list(ex.map(one, syms))

def worker(tf, secs):
    while True:
        try:
            syms = top_symbols(); scan(tf, syms); save_data()
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
        load_data()
        now = pd.Timestamp.now("UTC")
        syms = top_symbols()
        for tf in TFS:
            if due(tf, now): scan(tf, syms)
        if "--weekly" in sys.argv or weekly_due(datetime.datetime.utcnow()):
            weekly_report()
        save_data()
        sys.exit(0)
    load_data()
    print("NOVA SIGNALS BOT started")
    for tf, secs in TFS.items(): threading.Thread(target=worker, args=(tf, secs), daemon=True).start()
    while True: time.sleep(3600)
